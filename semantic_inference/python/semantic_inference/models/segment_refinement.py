"""Torch module for refining SAM masks."""

from semantic_inference.config import Config
from dataclasses import dataclass

import torch
import torchvision
from torch import nn


@dataclass
class SegmentRefinementConfig(Config):
    """Configuration for segment refinement."""

    dilate_masks: bool = False
    kernel_size: int = 3
    kernel_tolerance: float = 1.0e-3
    dilation_passes: int = 1


class SegmentRefinement(nn.Module):
    """Segment refining module."""

    Config = SegmentRefinementConfig

    def __init__(self, config):
        """Initialize the module with the provided config."""
        super(SegmentRefinement, self).__init__()
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
        masks = masks.to(torch.bool)
        device = masks.device

        num_masks = masks.size(0)
        if num_masks == 0:
            return masks, bboxes

        # order masks
        mask_sizes = torch.sum(masks, dim=(1, 2))
        sorted_idx = torch.argsort(mask_sizes, descending=True)

        dims = masks.size()
        img = torch.zeros((dims[1], dims[2], 1), dtype=sorted_idx.dtype, device=device)
        # think about broadcasting this somehow
        for idx in sorted_idx:
            img[masks[idx]] = idx + 1

        m_new = torch.zeros_like(masks)
        for idx in sorted_idx:
            nms_indices = (img == idx + 1)[:, :, 0]
            m_new[idx, nms_indices] = 1

        # filter out small border artifacts
        if self.config.dilate_masks:
            N = self.config.kernel_size
            m_new = m_new[:, None, :, :].to(torch.float32)
            W = torch.ones((1, 1, N, N), dtype=m_new.dtype, device=device)

            # erode mask
            for _ in range(self.config.dilation_passes):
                m_new = nn.functional.conv2d(m_new, W, padding="same")
                m_new[torch.abs(m_new - N**2) > self.config.kernel_tolerance] = 0

            # dilate mask
            for _ in range(self.config.dilation_passes):
                m_new = nn.functional.conv2d(m_new, W, padding="same")

            m_new = torch.squeeze(m_new, 1).to(torch.bool)

        valid = torch.squeeze(torch.argwhere(torch.sum(m_new, dim=(1, 2))))
        m_new = torch.index_select(m_new, 0, valid)
        b_new = torchvision.ops.masks_to_boxes(m_new).to(torch.int32)
        b_new[:, 2:] += 1
        return m_new, b_new
