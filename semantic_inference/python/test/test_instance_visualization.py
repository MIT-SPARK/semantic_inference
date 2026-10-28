"""Color contract for instance segmentation overlays."""

from types import SimpleNamespace

import numpy as np
import torch

from semantic_inference.visualization import get_semantic_overlay_img


def test_overlay_preserves_bgr_background_and_source():
    """Rendering uses BGR pixels without changing the input image."""
    image = np.full((40, 40, 3), [3, 17, 255], dtype=np.uint8)
    original = image.copy()
    masks = np.zeros((1, 40, 40), dtype=bool)
    masks[0, 20:25, 10:20] = True
    result = SimpleNamespace(
        masks=masks,
        boxes=torch.tensor([[10, 20, 20, 30]]),
        categories=torch.tensor([0]),
        confidences=torch.tensor([0.9]),
    )
    overlay = get_semantic_overlay_img(["car"], result, image)
    np.testing.assert_array_equal(overlay[-1, -1], [3, 17, 255])
    np.testing.assert_array_equal(image, original)
    assert np.any(overlay != image)
