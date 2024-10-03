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
"""Unit tests for mask refinement code."""

from semantic_inference.models import PatchExtractor, crop_to_bbox, center_crop
import torch
import helpers
import pytest


def _tile(img, stride):
    return torch.stack((img, img + stride, img + 2 * stride))


def test_crop_correct():
    """Test that cropping works as expected."""
    img = torch.arange(72, dtype=torch.uint8).reshape(3, 4, 6)

    b_xyxy = torch.tensor([0, 3, 1, 4], dtype=torch.int32)
    result = crop_to_bbox(img, b_xyxy)
    expected = torch.tensor([18, 42, 66], dtype=torch.uint8).reshape(3, 1, 1)
    assert (result == expected).all()

    b_xyxy = torch.tensor([4, 1, 6, 3], dtype=torch.int32)
    result = crop_to_bbox(img, b_xyxy)
    expected = torch.tensor(
        [[[10, 11], [16, 17]], [[34, 35], [40, 41]], [[58, 59], [64, 65]]],
        dtype=torch.uint8,
    )
    assert (result == expected).all()


def test_center_crop_correct():
    """Test that center crop works as expected."""
    img = torch.arange(24, dtype=torch.uint8).reshape(3, 2, 4)

    result = center_crop(img, 2)
    expected = torch.tensor(
        [[[1, 2], [5, 6]], [[9, 10], [13, 14]], [[17, 18], [21, 22]]],
        dtype=torch.uint8,
    )
    assert (result == expected).all()

    result = center_crop(img, 4)
    expected = torch.tensor(
        [
            [[0, 0, 0, 0], [0, 1, 2, 3], [4, 5, 6, 7], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [8, 9, 10, 11], [12, 13, 14, 15], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [16, 17, 18, 19], [20, 21, 22, 23], [0, 0, 0, 0]],
        ],
        dtype=torch.uint8,
    )
    assert (result == expected).all()

    result = center_crop(img, 6)
    expected = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 3, 0],
                [0, 4, 5, 6, 7, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 8, 9, 10, 11, 0],
                [0, 12, 13, 14, 15, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 16, 17, 18, 19, 0],
                [0, 20, 21, 22, 23, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.uint8,
    )
    assert (result == expected).all()


def test_no_masks_correct():
    """Test that if we pass in no masks, we get no images out."""
    model = PatchExtractor.construct(5)
    img = torch.zeros((4, 6, 3), dtype=torch.uint8)
    masks = torch.tensor([])
    bboxes = torch.tensor([])

    imgs = model.extract(img, bboxes=bboxes, masks=masks)[1]
    assert imgs.shape == (0, 3, 5, 5)


@helpers.parametrize_device
def test_no_normalization_no_resize(device, cleanup_torch):
    """Test that crop-only pipeline works."""
    # turn off normalization and resize
    model = PatchExtractor.construct(
        3, normalize=False, min_segment_area=1000, should_scale=False
    ).to(device)
    img = torch.arange(72, dtype=torch.uint8).reshape(4, 6, 3)
    masks = torch.ones((5, 4, 6), dtype=torch.bool)
    bboxes = torch.tensor(
        [[0, 0, 3, 3], [3, 0, 6, 3], [0, 2, 3, 5], [3, 2, 6, 5], [0, 0, 6, 4]]
    )

    results = model.extract(
        img.to(device), bboxes=bboxes.to(device), masks=masks.to(device)
    )
    imgs = results[1].cpu()
    expected = torch.zeros((5, 3, 3, 3))
    expected[0] = _tile(torch.tensor([[0, 3, 6], [18, 21, 24], [36, 39, 42]]), 1)
    expected[1] = _tile(torch.tensor([[9, 12, 15], [27, 30, 33], [45, 48, 51]]), 1)
    expected[2] = _tile(torch.tensor([[0, 0, 0], [36, 39, 42], [54, 57, 60]]), 1)
    expected[3] = _tile(torch.tensor([[0, 0, 0], [45, 48, 51], [63, 66, 69]]), 1)
    expected[4] = _tile(torch.tensor([[3, 6, 9], [21, 24, 27], [39, 42, 45]]), 1)
    # bounding box is smaller than tile size, so we pad by zero (which shifts center)
    # the floor behavior of center crop chooses to pick the top padding so we explicitly
    # zero the padding
    expected[2, :, 0, :] = 0
    expected[3, :, 0, :] = 0

    assert imgs == pytest.approx(expected.to(torch.float32))


@helpers.parametrize_device
def test_no_normalization(device, cleanup_torch):
    """Test that crop-only pipeline works."""
    # turn off normalization
    model = PatchExtractor.construct(
        3,
        normalize=False,
        min_segment_area=50,
        interpolation_mode="NEAREST",
        should_scale=False,
    ).to(device)
    # aspect ratio 2:1 makes resize a little bit more sane
    img = torch.arange(216, dtype=torch.uint8).reshape(3, 6, 12)
    masks = torch.zeros((2, 6, 12), dtype=torch.bool)
    # sets the diagonals of the mask for resized bbox
    masks[0, 0, 2] = True
    masks[0, 0, 6] = True
    masks[0, 2, 4] = True
    masks[0, 4, 2] = True
    masks[0, 4, 6] = True
    # set a different pattern for non-resized bbox (top corner)
    masks[1, 1, 3] = True
    masks[1, 1, 4] = True
    masks[1, 2, 3] = True
    bboxes = torch.tensor([[0, 0, 12, 6], [1, 0, 9, 6]])

    results = model(img.to(device), bboxes=bboxes.to(device), masks=masks.to(device))
    imgs = results[1].cpu()

    expected = torch.zeros((2, 3, 3, 3))
    # resized bbox should yield diagonals of 2 * arange(54).reshape(3, 3, 6)
    expected[0] = _tile(torch.tensor([[2, 4, 6], [26, 28, 30], [50, 52, 54]]), 72)
    e_mask = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    expected[0, :, e_mask] = 0

    expected[1] = _tile(torch.tensor([[15, 16, 17], [27, 28, 29], [39, 40, 41]]), 72)
    e_mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    expected[1, :, e_mask] = 0

    assert imgs == pytest.approx(expected.to(torch.float32))


@helpers.parametrize_device
def test_no_normalization_with_padding(device, cleanup_torch):
    """Test that crop-only pipeline works."""
    # turn off normalization
    model = PatchExtractor.construct(
        3,
        normalize=False,
        min_segment_area=50,
        interpolation_mode="NEAREST",
        should_scale=False,
        crop_padding=1,
    ).to(device)
    # aspect ratio 2:1 makes resize a little bit more sane
    img = torch.arange(216, dtype=torch.uint8).reshape(3, 6, 12)
    masks = torch.zeros((2, 6, 12), dtype=torch.bool)
    # sets the diagonals of the mask for resized bbox
    masks[0, 0, 2] = True
    masks[0, 0, 6] = True
    masks[0, 2, 4] = True
    masks[0, 4, 2] = True
    masks[0, 4, 6] = True
    # set a different pattern for non-resized bbox (top corner)
    masks[1, 1, 3] = True
    masks[1, 1, 4] = True
    masks[1, 2, 3] = True
    bboxes = torch.tensor([[1, 1, 11, 5], [2, 1, 8, 5]])

    results = model(img.to(device), bboxes=bboxes.to(device), masks=masks.to(device))
    imgs = results[1].cpu()

    expected = torch.zeros((2, 3, 3, 3))
    # resized bbox should yield diagonals of 2 * arange(54).reshape(3, 3, 6)
    expected[0] = _tile(torch.tensor([[2, 4, 6], [26, 28, 30], [50, 52, 54]]), 72)
    e_mask = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    expected[0, :, e_mask] = 0

    expected[1] = _tile(torch.tensor([[15, 16, 17], [27, 28, 29], [39, 40, 41]]), 72)
    e_mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    expected[1, :, e_mask] = 0

    assert imgs == pytest.approx(expected.to(torch.float32))


@helpers.parametrize_device
def test_normalize(device, suppress_torch, cleanup_torch):
    """Test that normalization makes sense."""
    # turn off normalization
    model = PatchExtractor.construct(
        3,
        normalize=True,
        min_segment_area=50,
        interpolation_mode="NEAREST",
        mean=[0.5, 0.5, 1.0],
        std=[10.0, 5.0, 5.0],
        should_scale=False,
    ).to(device)
    pixels = torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.uint8)
    # aspect ratio 2:1 makes resize a little bit more sane
    img = pixels.repeat(36).reshape(6, 12, 3)
    masks = torch.zeros((2, 6, 12), dtype=torch.bool)
    # sets the diagonals of the mask for resized bbox
    masks[0, 0, 2] = True
    masks[0, 0, 6] = True
    masks[0, 2, 4] = True
    masks[0, 4, 2] = True
    masks[0, 4, 6] = True
    # set a different pattern for non-resized bbox (top corner)
    masks[1, 1, 3] = True
    masks[1, 1, 4] = True
    masks[1, 2, 3] = True
    bboxes = torch.tensor([[0, 0, 12, 6], [1, 0, 9, 6]])

    results = model.extract(
        img.to(device), masks=masks.to(device), bboxes=bboxes.to(device)
    )
    imgs = results[1].cpu()

    # channel 0: {-0.3, 0.3}
    # channel: {-0.6, 0.6}
    # channel: {-0.5, 1.5}
    # diagonals of every other pixel
    expected = torch.zeros((2, 3, 3, 3))
    expected[0, 0] = torch.tensor([[-0.05, 0, -0.05], [0, -0.05, 0], [-0.05, 0, -0.05]])
    expected[0, 1] = torch.tensor([[0.10, 0, 0.10], [0, 0.10, 0], [0.10, 0, 0.10]])
    expected[0, 2] = torch.tensor([[0.20, 0, 0.20], [0, 0.20, 0], [0.20, 0, 0.20]])
    # upper l of every pixel
    expected[1, 0] = torch.tensor([[0.05, -0.05, 0], [0.05, 0, 0], [0, 0, 0]])
    expected[1, 1] = torch.tensor([[0.30, 0.10, 0], [0.30, 0, 0], [0, 0, 0]])
    expected[1, 2] = torch.tensor([[0.40, 0.20, 0], [0.40, 0, 0], [0, 0, 0]])

    assert imgs == pytest.approx(expected.to(torch.float32))
