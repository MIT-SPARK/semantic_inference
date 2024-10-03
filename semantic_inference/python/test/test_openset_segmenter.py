# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Unit tests for openset segmenter."""

import semantic_inference.models as models
import torch
import helpers
import pytest

import numpy as np


@helpers.parametrize_device
def test_nonempty_mask_correct(device):
    """Test that single non-empty mask works."""
    masks = torch.zeros((3, 6, 6))
    masks[0, :4, :3] = torch.ones(4, 3)
    masks[1] = torch.ones(6, 6)

    features = torch.ones((1, 3, 3, 5))
    for r in range(3):
        for c in range(3):
            features[0, r, c] = r * 3 + c

    results = models.pool_masked_features(features.to(device), masks.to(device))
    pooled_features = results[0].cpu()
    valid = torch.squeeze(results[1].cpu())

    expected_features = torch.ones((3, 5))
    expected_features[0] *= 2
    expected_features[1] *= 4
    expected_features[2] = 0
    assert pooled_features == pytest.approx(expected_features)
    assert (valid == torch.tensor([1, 1, 0], dtype=torch.bool)).all()


@pytest.mark.slow
@pytest.mark.skipif(not helpers.validate_modules("clip", "ultralytics"), reason="tbd")
@helpers.GPU_SKIP
def test_embeddings_correct(resource_dir):
    """Test that embeddings match old code."""
    test_pair = resource_dir / "clip_test_pairs" / "test_pair_000.npz"
    archive = np.load(str(test_pair))

    device = "cuda"
    model = models.OpensetSegmenter.construct(use_dense=False).to(device)

    img = torch.from_numpy(archive["img"]).to(device)
    masks = torch.from_numpy(archive["masks"]).to(device)
    bboxes = torch.from_numpy(archive["bboxes"]).to(device)
    with torch.no_grad():
        result = model.encode(img.to(device), masks.to(device), bboxes.to(device)).cpu()

    expected = torch.from_numpy(archive["features"])
    features = result.features
    dist = torch.nn.functional.cosine_similarity(
        expected.to(torch.float32), features.to(torch.float32)
    )
    assert dist == pytest.approx(torch.ones_like(dist), rel=1.0e-1)


@pytest.mark.slow
@pytest.mark.skipif(
    not helpers.validate_modules("clip", "f3rm", "ultralytics"), reason="tbd"
)
@helpers.GPU_SKIP
def test_dense_embeddings_correct(resource_dir):
    """Test that embeddings match old code (using f3rm)."""
    test_pair = resource_dir / "clip_test_pairs" / "dense_test_pair_000.npz"
    archive = np.load(str(test_pair))

    device = "cuda"
    model = models.OpensetSegmenter.construct(
        use_dense=True, use_dense_area_interpolation=True
    ).to(device)

    img = torch.from_numpy(archive["img"]).to(device)
    masks = torch.from_numpy(archive["masks"]).to(device)
    bboxes = torch.from_numpy(archive["bboxes"]).to(device)
    with torch.no_grad():
        result = model.encode(img.to(device), masks.to(device), bboxes.to(device)).cpu()

    expected = torch.from_numpy(archive["features"])
    features = result.features
    dist = torch.nn.functional.cosine_similarity(expected, features)
    assert dist == pytest.approx(torch.ones_like(dist), rel=1.0e-1)


def _make_masks(dims, bboxes):
    N = bboxes.shape[0]
    masks = torch.zeros((N,) + dims, dtype=torch.bool)
    for i in range(N):
        x1, y1, x2, y2 = bboxes[i]
        masks[i, y1:y2, x1:x2] = True

    return masks


def _get_pairwise_sim(X, Y):
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    N = X.shape[0]
    M = Y.shape[0]
    sims = torch.zeros(M, N)
    for i in range(N):
        scores = torch.nn.functional.cosine_similarity(X[i], Y)
        sims[:, i] = scores

    return sims


@torch.no_grad()
def _encode_images(model, *args):
    features = []
    for img in args:
        # kinda hacky way to get image vectors by forcing a single mask and bounding box
        h, w = img.shape[0], img.shape[1]
        bbox = torch.tensor([[0, 0, w, h]])
        mask = torch.ones((1, h, w), dtype=torch.bool)
        device = model.device
        ret = model.encode(img.to(device), mask.to(device), bbox.to(device))
        ret = ret.cpu()
        features.append(ret.features)

    return torch.cat(features)


@pytest.mark.slow
@pytest.mark.skipif(not helpers.validate_modules("clip", "ultralytics"), reason="tbd")
@helpers.GPU_SKIP
def test_get_image_masks_clip_features(resource_dir, image_factory):
    """Test getting clip features from masks."""
    rgb_cat = image_factory("cat.jpg")
    rgb_dog = image_factory("dog.jpg")
    rgb_tree = image_factory("tree.jpg")
    rgb_combined = image_factory("christmas.jpg")

    device = "cuda"
    model = models.OpensetSegmenter.construct(use_dense=False).to(device)

    clip_vectors = _encode_images(model, rgb_cat, rgb_dog, rgb_tree)

    cat_bbox = [97, 93, 147, 157]
    dog_bbox = [141, 50, 199, 169]
    tree_bbox = [10, 0, 129, 93]
    bboxes = torch.tensor([cat_bbox, dog_bbox, tree_bbox])
    masks = _make_masks(rgb_combined.shape[:2], bboxes)

    with torch.no_grad():
        ret = model.encode(rgb_combined.to(device), masks.to(device), bboxes.to(device))
        ret = ret.cpu()

    sims = _get_pairwise_sim(clip_vectors, ret.features)
    assert sims.shape == (3, 3)
    assert torch.argmax(sims[:, 0]) == 0
    assert torch.argmax(sims[:, 1]) == 1
    # TODO(nathan) think about figuring out why tree is bad
    # assert torch.argmax(sims[:, 1]) == 2
