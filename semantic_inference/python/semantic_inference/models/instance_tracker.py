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
"""Trackers that assign temporally-consistent instance ids to per-frame detections.

`InstanceSegmenter` calls a configured `InstanceTracker` after YOLO produces per-frame
boxes/masks, so a physical object keeps the same id across frames instead of getting a
fresh per-frame index. This is what lets khronos' `ExternalTracker` do exact-id
association instead of pixel-IoU re-projection (see `boxmot_integration_plan.md` at the
repo root).
"""

import dataclasses
import importlib
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
            np.ndarray: (n,) int64 track ids, aligned 1:1 with the input detections. A
            value of -1 means the tracker did not associate that detection to a track
            this frame (e.g. filtered out by the tracker's own confidence threshold).
        """
        raise NotImplementedError


class NullInstanceTracker(InstanceTracker):
    """Passthrough tracker: assigns a fresh per-frame index to every detection.

    Reproduces the pre-BoxMOT `instance_id = i + 1` behavior verbatim (ids are stable
    only within a single frame, not across frames). Selecting this via
    `instance_tracker.type: null` keeps the old, temporally-inconsistent behavior
    reachable without touching call sites.
    """

    def __init__(self, config: "NullInstanceTrackerConfig"):
        """Store config (unused; present for constructor-signature consistency)."""
        self.config = config

    def update(self, categories, boxes, confidences, frame):
        """Return `[0, n)` as the track ids for this frame."""
        n = 0 if boxes is None else boxes.shape[0]
        return np.arange(n, dtype=np.int64)


@register_config("instance_tracker", name="null", constructor=NullInstanceTracker)
@dataclasses.dataclass
class NullInstanceTrackerConfig(Config):
    """Config for `NullInstanceTracker` (no fields: nothing to configure)."""


class ClipReidAdapter:
    """Adapts `semantic_inference`'s CLIP wrapper to boxmot's ReID model contract.

    boxmot's appearance-matching path (`resolve_batch_embeddings`, called from any
    tracker constructed with `with_reid=True`) expects a model exposing
    `get_features(boxes, img) -> np.ndarray`, one embedding row per box. This wraps a
    "vanilla" CLIP visual encoder -- no ReID-specific fine-tuning -- as that model: the
    same idea the sibling DAAAM project's `export_vanilla_clip_engine.py` builds (base
    OpenAI CLIP, default-initialized bottleneck, no domain fine-tuning), just run here
    as plain PyTorch instead of exported to TensorRT. This is a materially better domain
    match than boxmot's own ReID model zoo, whose every downloadable checkpoint is fine-
    tuned on person re-identification datasets (Market1501/DukeMTMC/MSMT17) -- see
    boxmot_integration_plan.md for the full comparison. Uses
    `semantic_inference.models.wrappers.ClipWrapper`, which wraps the `clip` package
    already installed in `spark_env` for openset segmentation -- no new dependency.
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = "cpu"):
        """Load the CLIP visual encoder."""
        # Imported lazily (not at module scope) so importing this module doesn't require
        # the `clip` package unless the "clip" reid_backend is actually selected.
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

        h, w = img.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
            xi2, yi2 = min(int(x2), w), min(int(y2), h)
            # Degenerate box (can happen right at the image border): fall back to the
            # full frame rather than handing the CLIP transform a zero-size crop.
            crop = img[yi1:yi2, xi1:xi2] if xi2 > xi1 and yi2 > yi1 else img
            crops.append(self._clip._transform(Image.fromarray(crop)))

        batch = torch.stack(crops).to(self._device)
        embeddings = self._clip(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return embeddings.float().cpu().numpy()


class BoxmotInstanceTracker(InstanceTracker):
    """Wraps a BoxMOT tracker (default: BotSort, no ReID) to produce stable per-object
    ids."""

    def __init__(self, config: "BoxmotInstanceTrackerConfig"):
        """Construct the underlying BoxMOT tracker from the registry by name."""
        from boxmot.trackers.registry import get_tracker_definition

        self.config = config
        definition = get_tracker_definition(config.tracker_type)
        module_name, class_name = definition.class_path.rsplit(".", 1)
        tracker_cls = getattr(importlib.import_module(module_name), class_name)

        kwargs = {
            "track_buffer": config.track_buffer,
            "frame_rate": config.frame_rate,
        }
        if definition.accepts_per_class:
            kwargs["per_class"] = config.per_class

        # `needs_reid` is a registry hint that this tracker type *supports* appearance
        # matching, not that it requires it -- e.g. BotSort's own `with_reid` defaults
        # True but works fine with with_reid=False (motion + CMC only, no model needed).
        # So gate on `needs_reid` for which kwarg names to pass at all, and on
        # `config.with_reid` for whether to actually build a model.
        if definition.needs_reid:
            kwargs["with_reid"] = config.with_reid
            if config.with_reid:
                kwargs["reid_model"] = self._build_reid_model(config)
        elif config.with_reid:
            Logger.warning(
                f"with_reid=true but tracker_type '{config.tracker_type}' does not "
                "support ReID; ignoring."
            )

        # Passthrough for tracker-specific tuning (track_high_thresh, new_track_thresh,
        # match_thresh, proximity_thresh, appearance_thresh, cmc_method, use_cmc,
        # fuse_first_associate, min_conf, track_thresh, ...). Applied last so it can
        # also override the fields above if the caller explicitly asks.
        kwargs.update(config.tracker_args)

        self._tracker = tracker_cls(**kwargs)
        Logger.info(
            f"Constructed BoxMOT tracker '{config.tracker_type}' "
            f"({tracker_cls}): {kwargs}"
        )

    @staticmethod
    def _build_reid_model(config: "BoxmotInstanceTrackerConfig"):
        """Build the ReID model selected by `config.reid_backend`."""
        if config.reid_backend == "clip":
            return ClipReidAdapter(model_name=config.reid_weights, device=config.device)

        if config.reid_backend == "boxmot":
            from boxmot.models.reid import ReIDModel

            if not config.reid_weights:
                raise ValueError(
                    "reid_backend='boxmot' requires reid_weights to be set."
                )
            return ReIDModel(
                config.reid_weights, device=config.device, half=config.half
            ).model

        raise ValueError(f"Unknown reid_backend '{config.reid_backend}'")

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

        # Always call update(), even with zero detections: BoxMOT's Kalman predictions
        # and lost-track ageing must advance every frame, not just frames with
        # detections.
        tracks = self._tracker.update(dets, frame)

        # tracks: M x 8 [x1, y1, x2, y2, track_id, conf, cls, det_idx]. det_idx is the
        # index into the *input* dets array for whichever detection this track
        # associated to this frame -- use it directly rather than re-matching by IoU
        # (see daaam's TrackingService/_process_tracks for the same pattern).
        for row in tracks:
            det_idx = int(row[7])
            if 0 <= det_idx < n:
                track_ids[det_idx] = int(row[4])

        return track_ids


@register_config("instance_tracker", name="boxmot", constructor=BoxmotInstanceTracker)
@dataclasses.dataclass
class BoxmotInstanceTrackerConfig(Config):
    """Config for the BoxMOT-backed instance tracker."""

    # Any name registered in boxmot.trackers.registry.TRACKER_DEFINITIONS, e.g.
    # "botsort" (motion + camera-motion-compensation, our deployed default),
    # "bytetrack"/"ocsort" (motion+IoU only, no CMC), or "strongsort"/"deepocsort"
    # (ReID-only / ReID+motion).
    tracker_type: str = "botsort"

    # Frames a lost track is kept alive (no matching detection) before being dropped.
    track_buffer: int = 30

    # Frame rate used to scale `track_buffer` into a lost-track timeout in frames. Our
    # node processes frames at whatever rate the camera delivers minus
    # ImageWorkerConfig's queue_size=1 drops, not a fixed rate -- see
    # boxmot_integration_plan.md ("Frame pacing").
    frame_rate: int = 30

    # Keep tracks separate per semantic class. Deliberately False by default: a YOLO
    # class flip on an otherwise-stable detection (e.g. chair/bench) should not fragment
    # the track -- that is exactly the fragmentation problem this tracker exists to fix.
    # khronos' ExternalTracker complements this by matching on the track-id bits only,
    # so a class flip there doesn't mint a new khronos track either. Ignored for tracker
    # types whose registry entry marks accepts_per_class=False (e.g. strongsort).
    per_class: bool = False

    # Enable ReID (appearance) matching. Only meaningful for tracker types whose
    # registry entry marks needs_reid=True (botsort, strongsort, deepocsort, hybridsort,
    # boosttrack, occluboost) -- ignored (with a warning) otherwise. Deliberately False
    # in the deployed _awcd overlays: VRAM budget rules out a ReID network in the live
    # pipeline. Set True for offline experiments (see boxmot_integration_plan.md).
    with_reid: bool = False

    # Which ReID model to build when with_reid=True:
    # - "clip" (default): a vanilla, non-fine-tuned CLIP visual encoder via
    #   `ClipReidAdapter`, domain-appropriate for our open-vocab object classes.
    #   `reid_weights` is interpreted as the CLIP model_name (e.g. "ViT-B/16") for this
    #   backend.
    # - "boxmot": boxmot's own ReID model zoo via `boxmot.models.reid.ReIDModel`;
    #   `reid_weights` is a weights path/name from that zoo (e.g.
    #   "osnet_x0_25_msmt17.pt"). Kept for completeness / a future person-tracking use
    #   case -- every downloadable checkpoint there is person-ReID fine-tuned, not
    #   domain-appropriate for our object classes.
    reid_backend: str = "clip"
    reid_weights: str = "ViT-B/16"
    device: str = "cpu"
    half: bool = False

    # Extra keyword arguments forwarded directly to the underlying boxmot tracker
    # constructor (e.g. track_high_thresh, new_track_thresh, match_thresh,
    # proximity_thresh, appearance_thresh, cmc_method, use_cmc, fuse_first_associate,
    # min_conf, track_thresh). Kept as a passthrough rather than enumerated fields so
    # this wrapper doesn't have to track every tracker type's schema; unrecognized keys
    # are the caller's explicit choice, not a wrapper bug.
    tracker_args: dict[str, Any] = dataclasses.field(default_factory=dict)
