"""Test various wrappers."""

import semantic_inference.models as models
import helpers
import pytest


@pytest.mark.skipif(not helpers.validate_modules("clip"), reason="tbd")
def test_clip_wrapper():
    """Test that input size works for open clip."""
    model = models.ClipVisionWrapper.construct()
    assert model.input_size == 224


@pytest.mark.skipif(not helpers.validate_modules("open_clip"), reason="tbd")
def test_open_clip_wrapper():
    """Test that input size works for open clip."""
    model = models.OpenClipVisionWrapper.construct()
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
