"""Unit tests for mask refinement code."""

from semantic_inference.models import SegmentRefinement
import helpers
import torch


@helpers.parametrize_device
def test_no_masks_correct(device, cleanup_torch):
    """Test that if we pass in no masks, we get no masks."""
    masks = torch.tensor([])
    bboxes = torch.tensor([])

    model = SegmentRefinement.construct().to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    assert len(new_masks) == 0
    assert len(new_bboxes) == 0


@helpers.parametrize_device
def test_empty_masks_correct(device, cleanup_torch):
    """Test that if all masks are empty, we get no masks."""
    masks = torch.zeros((5, 4, 3))
    bboxes = torch.zeros((5, 4), dtype=torch.int32)

    model = SegmentRefinement.construct().to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    assert len(new_masks) == 0
    assert len(new_bboxes) == 0


@helpers.parametrize_device
def test_nonempty_mask_correct(device, cleanup_torch):
    """Test that single non-empty mask works."""
    masks = torch.zeros((5, 4, 3))
    masks[0] = torch.ones(4, 3)
    bboxes = torch.zeros((5, 4), dtype=torch.int32)
    bboxes[0] = torch.tensor([0, 0, 3, 4], dtype=torch.int32)

    model = SegmentRefinement.construct().to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    assert len(new_masks) == 1
    assert len(new_bboxes) == 1
    assert (new_masks[0] == masks[0]).all()
    assert (new_bboxes[0] == bboxes[0]).all()


@helpers.parametrize_device
def test_mask_suppression(device, cleanup_torch):
    """Test that we get the expected result."""
    masks = torch.zeros((5, 4, 3), dtype=torch.bool)
    masks[0] = torch.ones(4, 3, dtype=torch.bool)
    masks[1, :3, :3] = torch.ones(3, 3, dtype=torch.bool)
    masks[2, 2:, 1:] = torch.ones(2, 2, dtype=torch.bool)
    # note: this is for testing the result, not input (bboxes aren't used)
    bboxes = torch.tensor([[0, 3, 1, 4], [0, 0, 3, 3], [1, 2, 3, 4]], dtype=torch.int32)
    masks = masks.to(device)
    bboxes = bboxes.to(device)

    model = SegmentRefinement.construct().to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    expected_masks = torch.zeros((3, 4, 3), dtype=torch.bool)
    expected_masks[0, 3, 0] = 1
    expected_masks[1, :2, :3] = torch.ones((2, 3), dtype=torch.bool)
    expected_masks[1, 2, 0] = 1
    expected_masks[2, 2:, 1:] = torch.ones(2, 2, dtype=torch.bool)

    assert (new_masks == expected_masks).all()
    assert (new_bboxes == bboxes.cpu()).all()


@helpers.parametrize_device
def test_with_conv(device, cleanup_torch):
    """Test that we get the expected result with convolutions."""
    masks = torch.zeros((3, 6, 6), dtype=torch.bool)
    masks[0] = torch.ones(6, 6, dtype=torch.bool)
    masks[1, :4, :4] = torch.ones(4, 4, dtype=torch.bool)
    masks[2, 3:, 3:] = torch.ones(3, 3, dtype=torch.bool)
    # note: this is for testing the result, not input (bboxes aren't used)
    bboxes = torch.tensor([[0, 0, 4, 4], [3, 3, 6, 6]], dtype=torch.int32)

    model = SegmentRefinement.construct(dilate_masks=True).to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    expected_masks = torch.zeros((2, 6, 6), dtype=torch.bool)
    expected_masks[0, :3, :4] = 1
    expected_masks[0, 3, :3] = 1
    expected_masks[1, 3:, 3:] = 1

    assert (new_masks == expected_masks).all()
    assert (new_bboxes == bboxes).all()


@helpers.parametrize_device
def test_single_mask_correct(device, cleanup_torch):
    """Test that we don't squeeze all dimensions accidentally."""
    masks = torch.zeros((1, 6, 6), dtype=torch.bool)
    masks[0] = torch.ones(6, 6, dtype=torch.bool)
    # note: this is for testing the result, not input (bboxes aren't used)
    bboxes = torch.tensor([[0, 0, 6, 6]], dtype=torch.int32)

    model = SegmentRefinement.construct(dilate_masks=True).to(device)
    new_masks, new_bboxes = model(masks.to(device), bboxes.to(device))
    new_masks = new_masks.cpu()
    new_bboxes = new_bboxes.cpu()

    expected_masks = torch.ones((1, 6, 6), dtype=torch.bool)
    assert (new_masks == expected_masks).all()
    assert (new_bboxes == bboxes).all()
