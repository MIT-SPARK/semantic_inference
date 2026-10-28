"""Regression tests for refinement, detection alignment and packed IDs."""

from types import SimpleNamespace

import helpers
import numpy as np
import pytest
import torch
from torch import nn

from semantic_inference.models.instance_segmenter import (
    InstanceSegmenter,
    InstanceSegmenterConfig,
)
from semantic_inference.models.segment_refinement import SegmentRefinement


class Detections(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, image):
        masks = torch.zeros((3, *image.shape[:2]), device=self.device, dtype=torch.bool)
        masks[0, 0, 0] = True
        masks[1, 1:6, 1:6] = True
        masks[2, 6:9, 6:9] = True
        return (
            torch.tensor([0, 1, 2]),
            masks,
            torch.zeros((3, 4)),
            torch.tensor([0.4, 0.7, 0.9]),
        )


@helpers.parametrize_device
@pytest.mark.parametrize("rotation", ["none", "ROTATE_90_CLOCKWISE"])
def test_refinement_keeps_track_alignment(device, rotation):
    """Removing a detection retains the remaining category and wrapped track IDs."""
    config = InstanceSegmenterConfig(use_refinement=True, rotation_type=rotation)
    config.instance_model = SimpleNamespace(create=lambda: Detections(device))
    config.instance_tracker = SimpleNamespace(
        create=lambda: SimpleNamespace(update=lambda *args: np.array([7, 32767, -1]))
    )
    config.refinement = SegmentRefinement.Config(dilate_masks=True, erode_only=True)
    model = InstanceSegmenter(config)
    result = model.segment(np.zeros((10, 12, 3), dtype=np.uint8))
    expected = np.zeros((10, 12) if rotation == "none" else (12, 10), dtype=np.uint32)
    expected[2:5, 2:5] = (1 << 16) | 2
    if rotation != "none":
        expected = np.rot90(expected)

    np.testing.assert_array_equal(result.instances, expected)
    np.testing.assert_array_equal(result.track_ids, [32767, -1])
    torch.testing.assert_close(result.categories, torch.tensor([1, 2]))
    assert result.boxes.device.type == "cpu"
    torch.testing.assert_close(
        result.boxes.cpu(), torch.tensor([[2.0, 2.0, 5.0, 5.0], [7.0, 7.0, 8.0, 8.0]])
    )


@helpers.parametrize_device
def test_equal_area_overlap_order(device):
    """Overlap ties retain the CPU ordering used by the original pipeline."""
    masks = torch.zeros((20, 7, 8), dtype=torch.bool)
    masks[:, 1:6, 1:7] = True
    expected_winner = torch.argsort(masks.sum((1, 2)), descending=True)[-1]
    result, boxes, kept = SegmentRefinement.construct().refine(
        masks.to(device), torch.zeros((20, 4), device=device)
    )
    assert kept.tolist() == [expected_winner.item()]
    torch.testing.assert_close(result.cpu(), masks[:1])
    torch.testing.assert_close(
        boxes.cpu(), torch.tensor([[1, 1, 7, 6]], dtype=torch.int32)
    )


@helpers.parametrize_device
def test_refinement_removes_every_detection(device):
    """A fully eroded frame still produces an empty packed image of the input size."""
    config = InstanceSegmenterConfig(use_refinement=True, use_instance_tracker=False)
    config.instance_model = SimpleNamespace(create=lambda: Detections(device))
    config.refinement = SegmentRefinement.Config(dilate_masks=True, erode_only=True)
    result = InstanceSegmenter(config).segment(np.zeros((2, 3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result.instances, np.zeros((2, 3), dtype=np.uint32))
    assert result.masks is None
    assert result.track_ids.size == 0
