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
"""Tracking models for aggregating instance segmentations over time."""

import dataclasses
import logging
from typing import Any

import numpy as np
import torch
from spark_config import Config, register_config

Logger = logging.getLogger(__name__)


class InstanceTracker:
    """Interface: assigns a track id to each detection in the current frame."""

    def update(
        self,
        categories: torch.Tensor,
        boxes: torch.Tensor,
        confidences: torch.Tensor,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Update the tracker with the current frame's detections.

        Args:
            categories: (n,) category ids, one per detection.
            boxes: (n, 4) xyxy boxes, one per detection.
            confidences: (n,) detection confidences.
            frame: RGB image for this frame (H, W, 3), uint8.

        Returns:
            np.ndarray: (n,) int64 track id for every box (-1 means not associated)
        """
        raise NotImplementedError


class ClipReidAdapter:
    """Adapts `semantic_inference`'s CLIP wrapper to boxmot's ReID model contract."""

    def __init__(self, model_name: str = "ViT-B/16", device: str = "cpu"):
        """Load the CLIP visual encoder."""
        from semantic_inference.models.wrappers import ClipConfig, ClipWrapper

        self._clip = ClipWrapper(ClipConfig(model_name=model_name)).to(device)
        self._clip.eval()
        self._device = device

    @torch.no_grad()
    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Return one L2-normalized CLIP embedding per box, cropped from `img`."""
        from PIL import Image

        dim = self._clip.model.visual.output_dim
        if boxes is None or len(boxes) == 0:
            return np.empty((0, dim), dtype=np.float32)

        crops = []
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in boxes:
            xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
            xi2, yi2 = min(int(x2), w), min(int(y2), h)
            crop = img[yi1:yi2, xi1:xi2] if xi2 > xi1 and yi2 > yi1 else img
            crops.append(self._clip._transform(Image.fromarray(crop)))

        batch = torch.stack(crops).to(self._device)
        embeddings = self._clip(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return embeddings.float().cpu().numpy()


class BoxmotInstanceTracker(InstanceTracker):
    """Wraps a BoxMOT tracker to produce stable per-object IDs."""

    def __init__(self, config: "BoxmotInstanceTrackerConfig"):
        """Construct the underlying BoxMOT tracker from the registry by name."""
        import boxmot.trackers

        self.config = config
        if config.tracker_type == "botsort":
            self._tracker = boxmot.trackers.BotSort(
                track_buffer=config.track_buffer,
                frame_rate=config.frame_rate,
                with_reid=False,
                **config.tracker_args,
            )
        else:
            raise ValueError(f"Unimplemented type: '{config.tracker_type}'")

        Logger.info(f"Constructed BoxMOT tracker '{config.tracker_type}'")

    def update(self, categories, boxes, confidences, frame):
        """Run BoxMOT and map its output back to per-detection track ids."""
        n = 0 if boxes is None else boxes.shape[0]
        track_ids = np.full(n, -1, dtype=np.int64)

        if n == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            dets = np.column_stack(
                [
                    boxes.cpu().numpy(),
                    confidences.cpu().numpy(),
                    categories.cpu().numpy().astype(np.float32),
                ]
            ).astype(np.float32)

        tracks = self._tracker.update(dets, frame)

        # tracks: M x 8 [x1, y1, x2, y2, track_id, conf, cls, det_idx]. det_idx is the
        # index into the *input* dets array for whichever detection this track
        # associated to this frame
        for row in tracks:
            det_idx = int(row[7])
            if 0 <= det_idx < n:
                track_ids[det_idx] = int(row[4])

        return track_ids


@register_config("instance_tracker", name="boxmot", constructor=BoxmotInstanceTracker)
@dataclasses.dataclass
class BoxmotInstanceTrackerConfig(Config):
    """Config for the BoxMOT-backed instance tracker."""

    # Boxmot tracker method
    tracker_type: str = "botsort"
    # Frames a lost track is kept alive (no matching detection) before being dropped.
    track_buffer: int = 30
    # Frame rate used to scale `track_buffer` into a lost-track timeout in frames.
    frame_rate: int = 30
    # Keep tracks separate per semantic class.
    per_class: bool = False
    # Extra keyword arguments forwarded directly to the underlying boxmot tracker
    tracker_args: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Enable ReID (appearance) matching. Only meaningful for tracker types whose
    with_reid: bool = False
    reid_backend: str = "clip"
    reid_weights: str = "ViT-B/16"
    device: str = "cpu"
