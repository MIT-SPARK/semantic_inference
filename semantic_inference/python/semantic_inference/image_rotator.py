"""Class for performing inference on a rotated image."""

import enum

import numpy as np


class RotationType(enum.Enum):
    """Type of rotation to apply to image."""

    ROTATE_90_CLOCKWISE = "ROTATE_90_CLOCKWISE"
    ROTATE_180 = "ROTATE_180"
    ROTATE_90_COUNTERCLOCKWISE = "ROTATE_90_COUNTERCLOCKWISE"
    NONE = "none"


class ImageRotator:
    """Class to apply and unapply image rotations."""

    def __init__(self, mode):
        """Construct an image rotator with the transformation to apply for inference."""
        self._mode = mode
        match self._mode:
            case RotationType.ROTATE_90_CLOCKWISE:
                self._forward_iters = 3
                self._reverse_iters = 1
            case RotationType.ROTATE_180:
                self._forward_iters = 2
                self._reverse_iters = 2
            case RotationType.ROTATE_90_COUNTERCLOCKWISE:
                self._forward_iters = 1
                self._reverse_iters = 3
            case _:
                pass

    @property
    def mode(self):
        """Get the rotation mode."""
        return self._mode

    def rotate(self, img):
        """Rotate an image to the correct orientation to run inference."""
        if self._mode == RotationType.NONE:
            return img

        return np.rot90(img, k=self._forward_iters)

    def derotate(self, img):
        """Unapply rotation to an image after running inference."""
        if self._mode == RotationType.NONE:
            return img

        return np.rot90(img, k=self._reverse_iters)
