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

from semantic_inference.config import Config, config_field
from semantic_inference.models.segment_refinement import SegmentRefinement
from semantic_inference.models.mask_functions import ConstantMask
from semantic_inference.models.patch_extractor import PatchExtractor
from semantic_inference.models.patch_extractor import get_image_preprocessor
from semantic_inference.models.patch_extractor import default_normalization_parameters
from semantic_inference.models.patch_extractor import center_crop

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from typing import Any
import dataclasses
from dataclasses import dataclass, field


def _default_extractor():
    return PatchExtractor.Config(crop_padding=4)


def _map_opt(values, f):
    return {k: v if v is None else f(v) for k, v in values.items()}


def pool_masked_features(embeddings, masks, use_area=False):
    """
    Compute averaged features where masked elements are valid.

    Args:
        embeddings (torch.Tensor): Tensor of shape [1, H, W, L] where L is feature size
        masks (torch.Tensor): Tensor of shape [N, H, W] where N is number of masks

    Returns:
        torch.Tensor: Pooled features of shape [N, L]
    """
    target_size = (embeddings.size(1), embeddings.size(2))
    masks = masks.type(torch.uint8).unsqueeze(1)
    if use_area:
        downscaled = F.interpolate(
            masks.to(torch.float16), size=target_size, mode="area"
        ).squeeze()
        downscaled = (downscaled >= 0.5).to(torch.uint8)
    else:
        downscaled = F.interpolate(masks, size=target_size, mode="nearest").squeeze()

    downscaled = downscaled.unsqueeze(3)
    num_valid = torch.sum(downscaled, dim=(1, 2))
    valid = num_valid > 0

    features = downscaled * embeddings
    features = torch.sum(features, dim=(1, 2))
    num_valid[num_valid == 0] = 1
    features /= num_valid
    return features, valid


@dataclass
class Results:
    """Openset Segmentation Results."""

    masks: torch.Tensor
    boxes: torch.Tensor
    features: torch.Tensor
    boxed_patches: torch.Tensor
    masked_patches: torch.Tensor
    image_embedding: torch.Tensor

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
class OpensetSegmenterConfig(Config):
    """Main config for openset segmenter."""

    clip_model: Any = config_field("clip", default="clip")
    segmentation: Any = config_field("segmentation", default="fastsam")
    use_dense: bool = False
    dense_ratio: float = 0.9
    use_dense_area_interpolation: bool = False
    refinement: SegmentRefinement.Config = field(
        default_factory=SegmentRefinement.Config
    )
    patches: PatchExtractor.Config = field(default_factory=_default_extractor)


class OpensetSegmenter(nn.Module):
    """Module to segment and encode an image."""

    def __init__(self, config):
        """Construct an openset segmenter."""
        super(OpensetSegmenter, self).__init__()
        # for detecting model device
        self._canary_param = nn.Parameter(torch.empty(0))

        self.config = config
        self.segmenter = self.config.segmentation.create()
        self.segment_refinement = SegmentRefinement(config.refinement)
        self.encoder = self.config.clip_model.create()

        # previous code normalized after masking, so make sure we "normalize" 0
        # to be consistent
        mean, std = default_normalization_parameters()
        mask_value = -mean / std
        self.patch_extractor = PatchExtractor(
            self.encoder.input_size,
            config.patches,
            mask_function=ConstantMask(mask_value),
        )
        self.preprocess = get_image_preprocessor(self.encoder.input_size)

        self.dense_encoder = None
        if config.use_dense:
            self.dense_encoder = self.encoder.get_dense_encoder()

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpensetSegmenterConfig()
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
        # TODO(nathan) tensor if we can get around FastSAM
        return self(img)

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def encode(self, img, masks, boxes):
        """Compute language embeddings for each segment."""
        img = img.permute((2, 0, 1))

        masks_to_use = masks if self.dense_encoder is None else None
        patches = self.patch_extractor(img, bboxes=boxes, masks=masks_to_use)
        boxed_features = self.encoder(patches[0])
        img = self.preprocess(img)

        # dense clip doesn't use center-crop, so we have to apply it ourselves
        clip_img = center_crop(img, self.encoder.input_size)
        img_embedding = torch.squeeze(self.encoder(clip_img.unsqueeze(0)))

        if self.dense_encoder is None:
            assert patches[1] is not None
            masked_features = self.encoder(patches[1])
            features = (boxed_features + masked_features) / 2.0
        else:
            dense_embeddings = self.dense_encoder(img)
            dense_features, valid = pool_masked_features(
                dense_embeddings,
                masks,
                use_area=self.config.use_dense_area_interpolation,
            )
            ratios = self.config.dense_ratio * valid
            features = (1.0 - ratios) * boxed_features + ratios * dense_features

        return Results(masks, boxes, features, patches[0], patches[1], img_embedding)

    def forward(self, rgb_img):
        """
        Segment image and compute language embeddings for each mask.

        Args:
            img (np.ndarray): uint8 image of shape (R, C, 3) in rgb order

        Returns:
            Encoded image
        """
        masks, boxes = self.segmenter(rgb_img, device=self.device)
        masks, boxes = self.segment_refinement(masks, boxes)

        img = torch.from_numpy(rgb_img).to(self.device)
        return self.encode(img, masks, boxes)
