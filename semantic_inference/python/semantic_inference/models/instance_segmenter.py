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

    @property
    def instance_seg_img(self):
        """
        Convert segmentation results to instance segmentation image.
        Each pixel value encodes both category id and instance id.
        First 16 bits are category id, last 16 bits are instance id.
        """
        masks = self.masks.cpu().numpy()
        category_ids = self.categories.cpu().numpy()
        img = np.zeros(masks[0].shape, dtype=np.uint32)
        for i in range(masks.shape[0]):
            category_id = int(category_ids[i])  # category id are 0-indexed
            instance_id = i + 1  # instance ids are 1-indexed
            combined_id = (
                category_id << 16
            ) | instance_id  # combine into single uint32
            img[masks[i, ...] > 0] = combined_id

        return img

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
    """Main config for instance segmenter."""

    # relevant configs (model path, model weights) for the model
    instance_model: Any = config_field("instance_model", default="yolov11")


class InstanceSegmenter(nn.Module):
    """Module to segment and encode an image."""

    def __init__(self, config):
        """Construct an instance segmenter."""
        super().__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.segmenter = self.config.instance_model.create()

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
        categories, masks, boxes, confidences = self.segmenter(rgb_img)

        return Results(
            masks=masks, boxes=boxes, categories=categories, confidences=confidences
        )
