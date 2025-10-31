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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from spark_config import Config, config_field
from torch import nn

from semantic_inference.models.mask_functions import ConstantMask
from semantic_inference.models.patch_extractor import (
    PatchExtractor,
    center_crop,
    default_normalization_parameters,
    get_image_preprocessor,
)
from semantic_inference.models.segment_refinement import SegmentRefinement

def _map_opt(values, f):
    return {k: v if v is None else f(v) for k, v in values.items()}


@dataclass
class Results:
    """Openset Segmentation Results."""

    masks: torch.Tensor
    boxes: torch.Tensor # bounding boxes for the masks
    categories: torch.Tensor
    confidences: torch.Tensor   

    @property
    def instances(self):
        """Get instance image (if it exists)."""
        if self.masks.shape[0] == 0:
            return None

        np_masks = self.masks.numpy()
        img = np.zeros(np_masks[0].shape, dtype=np.uint16)
        for i in range(self.masks.shape[0]):
            # instance ids are 1-indexed
            img[np_masks[i, ...] > 0] = i + 1

        # TODO: 16 + 16 int for instance id and category id
        
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

    instance_model: Any = config_field("instance_model", default="yolov11")
    # relevant configs (model path, model weights) for the model


class InstanceSegmenter(nn.Module):
    """Module to segment and encode an image."""

    def __init__(self, config):
        """Construct an instance segmenter."""
        super().__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.segmenter = self.config.instance_model.create()
        # self.segment_refinement = SegmentRefinement(config.refinement) # might be useful clean up inprecise edges
    
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

    def forward(self, rgb_img):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarray): uint8 image of shape (R, C, 3) in rgb order

        Returns:
            Encoded image
        """
        categories, masks, boxes, confidences = self.segmenter(rgb_img)
        
        # img = torch.from_numpy(rgb_img).to(self.device)
        # return self.encode(img, masks, boxes)
        # TODO: return the results of the actual instance segmentation model here
        return Results(masks=masks, boxes=boxes, categories=categories, confidences=confidences)
