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
"""Test various wrappers."""

import semantic_inference.models as models
import helpers
import pytest


@pytest.mark.skipif(not helpers.validate_modules("clip"), reason="tbd")
def test_clip_wrapper():
    """Test that input size works for open clip."""
    model = models.ClipWrapper.construct()
    assert model.input_size == 224


@pytest.mark.skipif(not helpers.validate_modules("open_clip"), reason="tbd")
def test_open_clip_wrapper():
    """Test that input size works for open clip."""
    model = models.OpenClipWrapper.construct()
    assert model.input_size == 224


@pytest.mark.slow
@pytest.mark.skipif(not helpers.validate_modules("ultralytics"), reason="tbd")
@helpers.parametrize_device
def test_fastsam(device, image_factory):
    """Test that fastsam is working as expected."""
    model = models.FastSAMSegmentation.construct()
    assert model is not None

    rgb_combined = image_factory("christmas.jpg")
    masks, boxes = model.forward(rgb_combined.numpy(), device=device)
    assert masks is not None
    assert boxes is not None
    assert len(masks) > 0
    assert len(boxes) > 0


@pytest.mark.slow
@pytest.mark.skipif(not helpers.validate_modules("segment_anything"), reason="tbd")
@helpers.parametrize_device
def test_sam(device, image_factory):
    """Test that sam is working as expected."""
    model = models.SAMSegmentation.construct()
    assert model is not None
    model.to(device)

    rgb_combined = image_factory("christmas.jpg")
    masks, boxes = model.forward(rgb_combined.numpy())
    assert masks is not None
    assert boxes is not None
    assert len(masks) > 0
    assert len(boxes) > 0
