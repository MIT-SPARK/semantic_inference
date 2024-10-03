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
"""Torch module preparing batch of images for CLIP."""

from semantic_inference.config import Config
from semantic_inference.models.mask_functions import ConstantMask
from semantic_inference.misc import Logger

from typing import Optional, List
from dataclasses import dataclass
import torch

from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
import torchvision.ops


class ToFloat(nn.Module):
    """Replacement for ToDtype."""

    def __init__(self, dtype=torch.float32, should_scale=True):
        """Initialize the module."""
        super(ToFloat, self).__init__()
        self.dtype = dtype
        self.scale = 255.0 if should_scale else 1.0

    def forward(self, img):
        """Convert an image."""
        return img.to(self.dtype) / self.scale


def default_normalization_parameters():
    """Get default mean and standard deviation (for CLIP)."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    return mean, std


def get_image_preprocessor(size, interpolation_mode="BICUBIC"):
    """Get composed (v2) transform for converting image to CLIP input."""
    mode_name = interpolation_mode.upper()
    try:
        mode = torchvision.transforms.InterpolationMode[mode_name]
    except KeyError:
        Logger.warning(f"invalid mode '{mode_name}'! defaulting to 'BICUBIC'")
        mode = torchvision.transforms.InterpolationMode.BICUBIC

    mean, std = default_normalization_parameters()

    resize = v2.Resize(size, interpolation=mode, antialias=True)
    # crop = v2.CenterCrop(size)
    normalization = v2.Normalize(mean, std)
    return v2.Compose([resize, ToFloat(), normalization])


def crop_to_bbox(img, b_xyxy):
    """
    Crop an image to a specified bounding box.

    Assumes bounding box is fully inside image!

    Arguments:
        img (torch.Tensor): Image with [C, H, W] order
        b_xyxy (torch.Tensor): Bounding box

    Returns:
        Crop of image
    """
    x_min, y_min, x_max, y_max = torch.squeeze(b_xyxy)
    return img[..., y_min:y_max, x_min:x_max]


def center_crop(img, size, value=0):
    """
    Crop an image (from center) to be a square patch.

    Derived from torchvision center_crop and custom opencv
    version.

    Arguments:
        img (torch.Tensor): Image with [C, H, W] order
        size (int): Crop size
    """
    missing = size - torch.tensor(img.size()[-2:])
    missing = torch.maximum(missing, torch.tensor([0]))
    # prefer larger padding for odd missing values
    padding = torch.ceil(missing / 2.0)
    p_x = int(padding[1])
    p_y = int(padding[0])
    img = F.pad(img, (p_x, p_x, p_y, p_y), "constant", value=value)
    # we don't take non-integer sizes, so default to lower quadrant for center
    h, w = img.size()[-2:]
    y = (h - size) // 2
    x = (w - size) // 2
    return img[..., y : y + size, x : x + size]


def _extract_patch(img, b_xyxy, size, resize=False, **kwargs):
    b_img = crop_to_bbox(img, b_xyxy)

    # TODO(nathan) this disables any chance at using vmap, but also
    # most torch functionals don't support vmap either...
    if resize:
        b_img = v2.functional.resize(b_img, size, **kwargs)

    b_img = center_crop(b_img, size)
    return b_img


def _extract_mask_patch(mask, b_xyxy, size, resize=False):
    b_img = crop_to_bbox(mask, b_xyxy)

    # TODO(nathan) this disables any chance at using vmap, but also
    # most torch functionals don't support vmap either...
    if resize:
        b_img = b_img.to(torch.float32).unsqueeze(0)
        dims = torch.tensor(b_img.size()[-2:])
        ratio = size / torch.min(dims)
        h, w = torch.round(ratio * dims)
        new_dims = (int(h), int(w))
        b_img = F.interpolate(b_img, new_dims, mode="nearest")
        b_img = b_img.to(torch.bool)

    b_img = center_crop(b_img, size)
    return b_img


@dataclass
class PatchExtractorConfig(Config):
    """Configuration for patch extraction."""

    interpolation_mode: str = "bicubic"
    normalize: bool = True
    crop_padding: int = 0
    min_segment_area: int = 0
    min_mask_size: int = 0
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    should_scale: bool = True


class PatchExtractor(nn.Module):
    """Module to handle extracting image patches from bounding boxes."""

    Config = PatchExtractorConfig

    def __init__(self, size, config, mask_function=None):
        """Make the segment refinement."""
        super(PatchExtractor, self).__init__()
        self.size = size
        self.config = config

        mean_default, std_default = default_normalization_parameters()
        mean = mean_default if config.mean is None else torch.tensor(config.mean)
        std = std_default if config.std is None else torch.tensor(config.std)

        if mask_function is None:
            self.mask_function = ConstantMask()
        else:
            self.mask_function = mask_function

        mode_name = self.config.interpolation_mode.upper()
        try:
            self.mode = torchvision.transforms.InterpolationMode[mode_name]
        except KeyError:
            Logger.warning(f"invalid mode '{mode_name}'! defaulting to 'BICUBIC'")
            self.mode = torchvision.transforms.InterpolationMode.BICUBIC

        self._scale = ToFloat(should_scale=config.should_scale)
        self._normalize = v2.Normalize(mean, std)

    @classmethod
    def construct(cls, size, mask_function=None, *args, **kwargs):
        """Make a patch extractor."""
        config = PatchExtractorConfig(*args, **kwargs)
        return cls(size, config, mask_function=mask_function)

    def extract_masks(self, masks, bboxes):
        """
        Get resized masks for each patch.

        Bounding boxes are assumed to each be [min_x, min_y, max_x, max_y].

        Args:
            masks (torch.Tensor): bool tensor of masks of shape (N, R, C, 3)
            bboxes (torch.Tensor): int tensor of N bounding boxes of shape (N, 4)

        Returns:
            (torch.Tensor): (N, 3, S, S)
        """
        masks = masks.to(torch.bool)
        mask_sizes = torch.sum(masks, dim=(1, 2))

        new_dims = (masks.size(0), self.size, self.size)
        b_masks = torch.zeros(new_dims, dtype=masks.dtype, device=masks.device)

        areas = torchvision.ops.box_area(bboxes)
        needs_resize = torch.argwhere(areas >= self.config.min_segment_area)
        needs_crop = torch.argwhere(areas < self.config.min_segment_area)

        for idx in needs_resize:
            b_masks[idx] = _extract_mask_patch(
                masks[idx], bboxes[idx], self.size, resize=True
            )

        for idx in needs_crop:
            b_masks[idx] = _extract_mask_patch(
                masks[idx], bboxes[idx], self.size, resize=False
            )

        # disable masks for images that are too small
        b_masks[mask_sizes < self.config.min_mask_size, :, :] = torch.ones(
            (self.size, self.size), dtype=masks.dtype, device=masks.device
        )
        return b_masks

    def extract_patches(self, img, bboxes):
        """
        Get patches of image from bboxes.

        Bounding boxes are assumed to each be [min_x, min_y, max_x, max_y].

        Args:
            img (torch.Tensor): uint8 image tensor of shape (C, H, W)
            bboxes (torch.Tensor): int tensor of N bounding boxes of shape (N, 4)

        Returns:
            (torch.Tensor): (N, 3, S, S)
        """
        areas = torchvision.ops.box_area(bboxes)
        needs_resize = torch.argwhere(areas >= self.config.min_segment_area)
        needs_crop = torch.argwhere(areas < self.config.min_segment_area)

        new_dims = (bboxes.size(0), 3, self.size, self.size)
        imgs = torch.zeros(new_dims, dtype=img.dtype, device=img.device)

        resize_kwargs = {"resize": True, "interpolation": self.mode, "antialias": True}
        for idx in needs_resize:
            imgs[idx] = _extract_patch(img, bboxes[idx], self.size, **resize_kwargs)

        for idx in needs_crop:
            imgs[idx] = _extract_patch(img, bboxes[idx], self.size, resize=False)

        imgs = self._scale(imgs)
        if self.config.normalize:
            imgs = self._normalize(imgs)

        return imgs

    def forward(self, img, bboxes, masks=None):
        """
        Extract image patches for bounding boxes and optional masks.

        Bounding boxes are assumed to each be [min_x, min_y, max_x, max_y].

        Args:
            img (torch.Tensor): uint8 image tensor of shape (C, H, W)
            bboxes (torch.Tensor): int tensor of N bounding boxes of shape (N, 4)
            masks (Optional[torch.Tensor]): bool tensor of N masks of shape (N, R, C)

        Returns:
            (torch.Tensor): (N, 3, S, S)
        """
        bboxes = bboxes.to(torch.int32)
        N = bboxes.size(0)
        if N == 0:
            empty = torch.tensor([]).reshape(0, 3, self.size, self.size)
            return empty, None if masks is None else empty

        if self.config.crop_padding > 0:
            h, w = img.shape[-2:]
            bboxes[..., :2] -= self.config.crop_padding
            bboxes[..., 2:] += self.config.crop_padding
            bboxes[..., 0].clamp_(min=0, max=w)
            bboxes[..., 1].clamp_(min=0, max=h)
            bboxes[..., 2].clamp_(min=0, max=w)
            bboxes[..., 3].clamp_(min=0, max=h)

        imgs = self.extract_patches(img, bboxes)

        if masks is None:
            return imgs, None

        masks = self.extract_masks(masks, bboxes)
        return imgs, self.mask_function(imgs, masks)

    def extract(self, img, bboxes, masks=None):
        """
        Extract image patches for bounding boxes and optional masks.

        Takes image in [H, W, C] order, see forward for details.
        """
        return self(img.permute((2, 0, 1)), bboxes, masks=masks)
