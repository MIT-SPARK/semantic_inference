"""YOLO BGR input and inference geometry contracts."""

from types import SimpleNamespace

import numpy as np
from torch import nn

from semantic_inference.models.wrappers import (
    YoloInstanceSegmenterBase,
    YoloInstanceSegmenterBaseConfig,
)


class InputModel(nn.Module):
    def forward(self, image, **kwargs):
        self.image = image
        self.image_size = kwargs["imgsz"]
        return [SimpleNamespace(masks=None)]


def test_yolo_preserves_bgr_without_resizing_source():
    """BGR channels are preserved while inference geometry is stride-aligned."""
    wrapper = YoloInstanceSegmenterBase(YoloInstanceSegmenterBaseConfig())
    wrapper.model = InputModel()
    wrapper._stride = 32
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    image[0, 0] = [3, 17, 255]
    wrapper(image)
    assert wrapper.model.image_size == [544, 960]
    assert wrapper.model.image.shape == image.shape
    np.testing.assert_array_equal(wrapper.model.image[0, 0], [3, 17, 255])
    np.testing.assert_array_equal(image[0, 0], [3, 17, 255])


def test_yolo_accepts_rotated_noncontiguous_bgr():
    """Rotation retains the correct color order and swaps inference dimensions."""
    wrapper = YoloInstanceSegmenterBase(YoloInstanceSegmenterBaseConfig())
    wrapper.model = InputModel()
    wrapper._stride = 32
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    image[0, 0] = [83, 29, 11]
    rotated = np.rot90(image)
    wrapper(rotated)
    assert wrapper.model.image_size == [960, 544]
    assert wrapper.model.image.shape == (960, 540, 3)
    assert wrapper.model.image.flags.c_contiguous
    np.testing.assert_array_equal(wrapper.model.image[-1, 0], [83, 29, 11])
