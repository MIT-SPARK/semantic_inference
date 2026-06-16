"""Unit tests for image rotator."""

import numpy as np

from semantic_inference.image_rotator import ImageRotator, RotationType


def test_no_rotation():
    """Test that no rotation produces the same matrix."""
    A = np.arange(24).reshape((3, 4, 2))
    rotator = ImageRotator(RotationType.NONE)

    assert (rotator.rotate(A) == A).all()
    assert (rotator.derotate(A) == A).all()


def test_90cw_rotation():
    """Test that no rotation produces the same matrix."""
    A = np.arange(24).reshape((3, 4, 2))

    rotator = ImageRotator(RotationType.ROTATE_90_CLOCKWISE)
    rotated = rotator.rotate(A)
    expected = np.array(
        [
            [[16, 17], [8, 9], [0, 1]],
            [[18, 19], [10, 11], [2, 3]],
            [[20, 21], [12, 13], [4, 5]],
            [[22, 23], [14, 15], [6, 7]],
        ]
    )

    assert rotated.shape == (4, 3, 2)
    assert (rotated == expected).all()
    assert (rotator.derotate(rotated) == A).all()


def test_180_rotation():
    """Test that no rotation produces the same matrix."""
    A = np.arange(24).reshape((3, 4, 2))

    rotator = ImageRotator(RotationType.ROTATE_180)
    rotated = rotator.rotate(A)

    assert rotated.shape == A.shape
    assert (rotated != A).any()
    assert (rotator.derotate(rotated) == A).all()


def test_90ccw_rotation():
    """Test that no rotation produces the same matrix."""
    A = np.arange(24).reshape((3, 4, 2))

    rotator = ImageRotator(RotationType.ROTATE_90_COUNTERCLOCKWISE)
    rotated = rotator.rotate(A)
    expected = np.array(
        [
            [[6, 7], [14, 15], [22, 23]],
            [[4, 5], [12, 13], [20, 21]],
            [[2, 3], [10, 11], [18, 19]],
            [[0, 1], [8, 9], [16, 17]],
        ]
    )

    assert rotated.shape == (4, 3, 2)
    assert (expected == rotated).all()
    assert (rotator.derotate(rotated) == A).all()
