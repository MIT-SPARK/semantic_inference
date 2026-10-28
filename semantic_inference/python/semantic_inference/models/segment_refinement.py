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
"""Torch module for refining SAM masks."""

from dataclasses import dataclass

import torch
import torchvision
from spark_config import Config
from torch import nn


@dataclass
class SegmentRefinementConfig(Config):
    """Configuration for segment refinement."""

    dilate_masks: bool = False
    kernel_size: int = 3
    kernel_tolerance: float = 1.0e-3
    passes: int = 1
    erode_only: bool = False


class SegmentRefinement(nn.Module):
    """Segment refining module."""

    Config = SegmentRefinementConfig

    def __init__(self, config):
        """Initialize the module with the provided config."""
        super().__init__()
        self.config = config

    @classmethod
    def construct(cls, *args, **kwargs):
        """See config."""
        config = SegmentRefinementConfig(*args, **kwargs)
        return cls(config)

    def forward(self, masks, bboxes):
        """
        Refine computed masks.

        Bounding boxes are assumed to each be [min_x, min_y, max_x, max_y]

        Args:
            masks (torch.Tensor): bool tensor of N image masks of shape (N, R, C)
            bboxes (torch.Tensor): int tensor of N bounding boxes of shape (N, 4)

        Returns:
            masks, bboxes as tensors
        """
        masks, bboxes, _ = self.refine(masks, bboxes)
        return masks, bboxes

    def refine(self, masks, bboxes):
        """
        Refine computed masks, also returning which input masks were kept.

        Masks that end up empty are dropped

        Args:
            masks (torch.Tensor): bool tensor of N image masks of shape (N, R, C)
            bboxes (torch.Tensor): int tensor of N bounding boxes of shape (N, 4)

        Returns:
            masks, bboxes, and the int64 indices (M,) of the kept input masks
        """
        masks = masks.to(torch.bool)
        device = masks.device

        num_masks = masks.size(0)
        if num_masks == 0:
            return masks, bboxes, torch.zeros(0, dtype=torch.int64, device=device)

        refined = self._resolve_overlaps(masks)
        if self.config.dilate_masks:
            refined = self._apply_morphology(refined)

        valid = refined.flatten(1).any(dim=1).nonzero().flatten()
        refined = torch.index_select(refined, 0, valid)
        boxes = torchvision.ops.masks_to_boxes(refined).to(torch.int32)
        boxes[:, 2:] += 1
        return refined, boxes, valid

    @staticmethod
    def _resolve_overlaps(masks):
        device = masks.device
        mask_sizes = torch.sum(masks, dim=(1, 2))
        # CPU sorting preserves the original tie order for either device.
        sorted_idx = torch.argsort(mask_sizes.cpu(), descending=True)
        height, width = masks.shape[1:]
        img = torch.zeros((height, width), dtype=torch.int64, device=device)
        for idx in sorted_idx.tolist():
            img[masks[idx]] = idx + 1

        mask_ids = torch.arange(1, masks.size(0) + 1, device=device)
        return img.unsqueeze(0) == mask_ids[:, None, None]

    def _apply_morphology(self, masks):
        device = masks.device
        N = self.config.kernel_size
        masks = masks[:, None, :, :].to(torch.float32)
        W = torch.ones((1, 1, N, N), dtype=masks.dtype, device=device)

        # erode mask
        for _ in range(self.config.passes):
            masks = nn.functional.conv2d(masks, W, padding="same")
            masks[torch.abs(masks - N**2) > self.config.kernel_tolerance] = 0

        # dilate mask
        if not self.config.erode_only:
            for _ in range(self.config.passes):
                masks = nn.functional.conv2d(masks, W, padding="same")

        return torch.squeeze(masks, 1).to(torch.bool)
