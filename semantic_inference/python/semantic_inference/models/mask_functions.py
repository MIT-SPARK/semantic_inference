"""Torch module preparing batch of images for CLIP."""

import torch

from torch import nn


class NoMask(nn.Module):
    """Module that does not actually change unmaksed pixels."""

    def __init__(self):
        """Make the module."""
        super(NoMask, self).__init__()

    def forward(self, img, mask):
        """Return the same img as supplied."""
        return img


class ConstantMask(nn.Module):
    """Module to clear unmasked pixels with a constant value."""

    def __init__(self, value=None):
        """
        Construct a constant mask module.

        Args:
            value (Optional[torch.Tensor]): color to apply of shape (3)
        """
        super(ConstantMask, self).__init__()
        if value is None:
            value = torch.zeros((3, 1), dtype=torch.float32)
        else:
            if isinstance(value, torch.Tensor):
                # avoid torch warning about copying tensors
                value = value.clone().detach()
            else:
                value = torch.tensor(value)

            value = value.reshape((3, 1)).to(torch.float32)

        self.register_buffer("value", value)

    def forward(self, img, mask):
        """
        Set unmasked pixels to a constant value.

        Args:
            img (torch.Tensor): image tensor of N images of shape (N, 3, R, C)
            masks (torch.Tensor): bool tensor of N image masks of shape (N, R, C)

        Returns:
            (torch.Tensor): masked images of (N, 3, R, C)
        """
        masked = img.clone()
        masked = masked.permute(1, 0, 2, 3)
        masked[:, torch.logical_not(mask)] = self.value
        return masked.permute(1, 0, 2, 3)


class GaussianMask(nn.Module):
    """
    Segment refining module to clear unmasked pixels.

    Applies a value sampled from a normal distribution.

    Args:
        mean (Optional[torch.Tensor]: distribution mean
        std (Optional[torch.Tensor]: distribution standard deviation
        generator: random generator to use
    """

    def __init__(self, mean=None, std=None, generator=None):
        """Make the segment refinement."""
        super(GaussianMask, self).__init__()
        self.mean = (
            torch.tensor(mean)
            if mean is not None
            else torch.tensor([0.48145466, 0.4578275, 0.40821073])
        )
        self.std = (
            torch.tensor(std)
            if std is not None
            else torch.tensor([0.26862954, 0.26130258, 0.27577711])
        )
        self.gen = generator

    def forward(self, img, mask):
        """
        Set unmasked pixels to a constant value.

        Args:
            img (torch.Tensor): image tensor of N images of shape (N, 3, R, C)
            masks (torch.Tensor): bool tensor of N image masks of shape (N, R, C)

        Returns:
            (torch.Tensor): masked images of (N, 3, R, C)
        """
        img = img.clone()
        # more convenient to deal with channels first instead of batch
        img = img.permute(1, 0, 2, 3)
        # unclear if normal_ broadcasts mean and std so we go by channel
        r_mean, g_mean, b_mean = self.mean
        r_std, g_std, b_std = self.std
        mask = torch.logical_not(mask)
        img[0, mask] = img[0, mask].normal_(mean=r_mean, std=r_std, generator=self.gen)
        img[1, mask] = img[1, mask].normal_(mean=g_mean, std=g_std, generator=self.gen)
        img[2, mask] = img[2, mask].normal_(mean=b_mean, std=b_std, generator=self.gen)
        return img.permute(1, 0, 2, 3)
