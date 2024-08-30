"""Unit tests for mask application."""

from semantic_inference.models import ConstantMask, GaussianMask
import helpers
import pytest
import torch


def _compare_masked(masks, expected, result):
    expected = expected.permute(1, 0, 2, 3)
    result = result.permute(1, 0, 2, 3)
    assert (expected[:, masks] == result[:, masks]).all()


def _get_unmasked_stats(masks, result):
    result = result.permute(1, 0, 2, 3)
    unmasked = torch.logical_not(masks)
    return torch.mean(result[:, unmasked], dim=1), torch.std(result[:, unmasked], dim=1)


@helpers.parametrize_device
def test_constant_mask(device):
    """Test that constant mask works."""
    img = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    masks = torch.zeros(4, 2, 2, dtype=torch.bool)
    masks[0] = torch.ones(2, 2, dtype=torch.bool)
    masks[1] = torch.eye(2, dtype=torch.bool)
    masks[2] = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)

    model = ConstantMask().to(device)
    result = model(img.to(device), masks.to(device)).cpu()

    expected = torch.zeros(4, 3, 2, 2)
    expected[0] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    expected[1] = torch.tensor(
        [[[12, 0], [0, 15]], [[16, 0], [0, 19]], [[20, 0], [0, 23]]],
        dtype=torch.float32,
    )
    expected[2] = torch.tensor(
        [[[0, 25], [26, 0]], [[0, 29], [30, 0]], [[0, 33], [34, 0]]],
        dtype=torch.float32,
    )
    assert (result == expected).all()


@helpers.parametrize_device
def test_constant_mask_nonzero(device):
    """Test that constant mask works with non-default value."""
    img = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    masks = torch.zeros(4, 2, 2, dtype=torch.bool)
    masks[0] = torch.ones(2, 2, dtype=torch.bool)
    masks[1] = torch.eye(2, dtype=torch.bool)
    masks[2] = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)

    model = ConstantMask(value=[1, 2, 3]).to(device)
    result = model(img.to(device), masks.to(device)).cpu()

    expected = torch.zeros(4, 3, 2, 2)
    expected[0] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    expected[1] = torch.tensor(
        [[[12, 1], [1, 15]], [[16, 2], [2, 19]], [[20, 3], [3, 23]]],
        dtype=torch.float32,
    )
    expected[2] = torch.tensor(
        [[[1, 25], [26, 1]], [[2, 29], [30, 2]], [[3, 33], [34, 3]]],
        dtype=torch.float32,
    )
    expected[3] = torch.tensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
    assert result == pytest.approx(expected)


@helpers.parametrize_device
def test_gaussian_mask(device):
    """Test that gaussian mask works."""
    img = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    masks = torch.zeros(4, 2, 2, dtype=torch.bool)
    masks[0] = torch.ones(2, 2, dtype=torch.bool)
    masks[1] = torch.eye(2, dtype=torch.bool)
    masks[2] = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)

    model = GaussianMask().to(device)
    result = model(img.to(device), masks.to(device)).cpu()

    expected = torch.zeros(4, 3, 2, 2)
    expected[0] = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    expected[1] = torch.tensor(
        [[[12, 0], [0, 15]], [[16, 0], [0, 19]], [[20, 0], [0, 23]]],
        dtype=torch.float32,
    )
    expected[2] = torch.tensor(
        [[[0, 25], [26, 0]], [[0, 29], [30, 0]], [[0, 33], [34, 0]]],
        dtype=torch.float32,
    )
    _compare_masked(masks, expected, result)
    mean, std = _get_unmasked_stats(masks, result)
    # we aren't sampling that much, so hard for the moments to actually match
    assert (model.mean - mean).norm() <= 0.5
    assert (model.std - std).norm() <= 0.5
