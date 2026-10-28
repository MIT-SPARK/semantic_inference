"""RGB-only models at the BGR instance-pipeline boundary."""

import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from semantic_inference.models.instance_tracker import ClipReidAdapter
from semantic_inference.models.wrappers import GDSam2InstanceSegmenterWrapper


def test_gdsam_consumers_receive_rgb(monkeypatch):
    """Grounding DINO and SAM 2 both see RGB pixels from a BGR input."""

    def predict(**kwargs):
        torch.testing.assert_close(
            kwargs["image"][:, 0, 0], torch.tensor([255, 17, 3], dtype=torch.uint8)
        )
        return torch.tensor([[0.5, 0.5, 1.0, 1.0]]), torch.tensor([1.0]), ["object"]

    monkeypatch.setitem(
        sys.modules, "groundingdino.util.inference", SimpleNamespace(predict=predict)
    )

    class Sam:
        def set_image(self, image):
            np.testing.assert_array_equal(image[0, 0], [255, 17, 3])

        def predict(self, **kwargs):
            return np.ones((1, 1, 2, 2), dtype=bool), np.ones((1, 1)), None

    model = GDSam2InstanceSegmenterWrapper.__new__(GDSam2InstanceSegmenterWrapper)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(box_threshold=0.3, text_threshold=0.3)
    model.device = torch.device("cpu")
    model.grounding_model = object()
    model.text_prompt = "object."
    model.category_names = ["object"]
    model.preprocess_image = lambda image: torch.from_numpy(image.copy()).permute(
        2, 0, 1
    )
    model.sam2_predictor = Sam()
    model.multimask_output = False
    model.erosion = False
    categories, masks, boxes, confidences = model(
        np.full((2, 2, 3), [3, 17, 255], dtype=np.uint8)
    )
    assert masks.all()
    torch.testing.assert_close(categories, torch.tensor([0]))


def test_clip_reid_encodes_rgb_crops():
    """BGR tracker crops are converted to RGB before CLIP encoding."""

    class PixelClip(nn.Module):
        model = SimpleNamespace(visual=SimpleNamespace(output_dim=3))

        def _transform(self, image):
            return torch.from_numpy(np.array(image)[0, 0].copy()).float()

        def forward(self, batch):
            return batch

    adapter = ClipReidAdapter.__new__(ClipReidAdapter)
    adapter._clip = PixelClip()
    adapter._device = "cpu"
    features = adapter.get_features(
        np.array([[0, 0, 1, 1]]), np.array([[[3, 17, 255]]], dtype=np.uint8)
    )
    expected = np.array([[255, 17, 3]], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(features, expected)
