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
"""Model to segment an image and encode segments with CLIP embeddings."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from spark_config import Config, config_field
from torch import nn

from semantic_inference.image_rotator import ImageRotator, RotationType


def _map_opt(values, f):
    return {k: v if v is None else f(v) for k, v in values.items()}


@dataclass
class Results:
    """Openset Segmentation Results."""

    # Expecting all tensor to be on cpu
    masks: torch.Tensor  # (n, H, W), torch.bool
    boxes: torch.Tensor  # (n, 4) xyxy format, torch.float32
    categories: torch.Tensor  # (n,), torch.float32/int64 (doesn't matter)
    confidences: torch.Tensor  # (n,), torch.float32
    instances: np.ndarray  # (H, W, c) np.uint32 (result image)
    # (n,) int64 raw (pre-wrap) tracker ids, aligned to
    # masks/boxes/categories/confidences. -1 means the tracker didn't associate that
    # detection to a track this frame. Debug/visualization only (see
    # visualization.get_semantic_overlay_img) -- `instances` above already carries the
    # wrapped id actually used downstream; not part of that format.
    track_ids: np.ndarray = None

    def cpu(self):
        """Move results to CPU."""
        values = dataclasses.asdict(self)
        return Results(**_map_opt(values, lambda v: v.cpu()))

    def to(self, *args, **kwargs):
        """Forward to to all tensors."""
        values = dataclasses.asdict(self)
        return Results(**_map_opt(values, lambda v: v.to(*args, **kwargs)))


@dataclass
class InstanceSegmenterConfig(Config):
    """
    Main config for instance segmenter.

    Attributes:
        instance_model: Configuration for underlying instance segmentation model.
        instance_tracker: Configuration for the tracker assigning temporally-consistent
            instance ids (see semantic_inference.models.instance_tracker). Defaults to a
            BoxMOT-backed tracker; set type "null" to reproduce the legacy per-frame
            index ids.
        rotation_type: Amount of rotation to apply (0, 90 c/ccw, 180).
        label_offset: Fixed offset to apply to labels
    """

    # relevant configs (model path, model weights) for the model
    instance_model: Any = config_field("instance_model", default="yolo-seg")
    instance_tracker: Any = config_field("instance_tracker", default="boxmot")
    rotation_type: str = "none"
    category_offset: int = 1


class InstanceSegmenter(nn.Module):
    """Module to segment and encode an image."""

    def __init__(self, config):
        """Construct an instance segmenter."""
        super().__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self._rotator = ImageRotator(RotationType(config.rotation_type))
        self.segmenter = self.config.instance_model.create()
        self.instance_tracker = self.config.instance_tracker.create()

    def eval(self):
        """
        Override eval to avoid issues with certain models
        """
        self.segmenter.eval()

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = InstanceSegmenterConfig()
        config.update(kwargs)
        return cls(config)

    @torch.no_grad()
    def segment(self, rgb_img, is_rgb_order=True):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarry): uint8 image of shape (R, C, 3) in rgb order
            is_rgb_order (bool): whether the image is rgb order or not

        Returns:
            Encoded image
        """
        img = rgb_img if is_rgb_order else rgb_img[:, :, ::-1].copy()
        return self(img)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    @property
    def category_names(self):
        """Get category names."""
        return self.segmenter.category_names

    def forward(self, rgb_img):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarray): uint8 image of shape (R, C, 3) in rgb order

        Returns:
            Encoded image
        """
        rotated = self._rotator.rotate(rgb_img)
        categories, masks, boxes, confidences = self.segmenter(rotated)

        if categories is None:
            # No detections this frame. Still feed the tracker an empty update: BoxMOT's
            # Kalman predictions and lost-track ageing must advance every frame, not
            # just frames with detections (see
            # semantic_inference.models.instance_tracker).
            track_categories = torch.empty(0, dtype=torch.int64)
            track_boxes = torch.empty((0, 4), dtype=torch.float32)
            track_confidences = torch.empty(0, dtype=torch.float32)
        else:
            track_categories, track_boxes, track_confidences = (
                categories,
                boxes,
                confidences,
            )
        track_ids = self.instance_tracker.update(
            track_categories,
            track_boxes,
            track_confidences,
            np.ascontiguousarray(rotated),
        )

        if masks is None:
            instances = np.zeros(rgb_img.shape[:2])
        else:
            instances = np.zeros(masks[0].shape, dtype=np.uint32)
            masks = masks.cpu().numpy()
            category_ids = categories.cpu().numpy()
            for i in range(masks.shape[0]):
                raw_id = int(track_ids[i])
                if raw_id < 0:
                    # Tracker did not associate this detection to a track this frame;
                    # leave its pixels unlabeled (id 0) rather than mint a fake/unstable
                    # id for it.
                    continue

                category_id = int(category_ids[i]) + self.config.category_offset
                # Wrap into 16 bits, reserving 0 (0 stays "no instance"/background
                # downstream). NOTE: the wire's upper 16 bits land in a *signed* 16-bit
                # channel downstream (InstanceSubscriber::fillInput unpacks `original >>
                # 16` into a CV_16SC1 mat, hydra_ros/src/input/image_receiver.cpp), so
                # the usable range is [1, 32767], not [1, 65535] -- an id above 32767
                # wraps negative on that side.
                instance_id = (raw_id % 32767) + 1
                # combine into single uint32
                combined_id = (instance_id << 16) | category_id
                instances[masks[i, ...] > 0] = combined_id

            instances = self._rotator.derotate(instances)

        return Results(
            masks=masks,
            boxes=boxes,
            categories=categories,
            confidences=confidences,
            instances=instances,
            track_ids=track_ids,
        )
