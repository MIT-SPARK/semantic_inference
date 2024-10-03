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
"""Model wrappers for image segmentation."""

import dataclasses

import einops
import torch
import torch.nn as nn
import torchvision
from semantic_inference import root_path
from semantic_inference.config import Config, register_config


def models_path():
    """Get path to pre-trained weight storage."""
    return root_path().parent.parent / "models"


class FastSAMSegmentation(nn.Module):
    """Fast SAM wrapper."""

    def __init__(self, config, verbose=False):
        """Load Fast SAM."""
        super(FastSAMSegmentation, self).__init__()
        from ultralytics import FastSAM

        self.config = config
        self.verbose = verbose
        self.sam = FastSAM(config.model_name)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = FastSAMConfig()
        config.update(kwargs)
        return cls(config)

    def train(self, mode):
        """Don't pass train to underlying model."""
        pass

    def forward(self, img, device=None):
        """Segment image."""
        # TODO(nathan) resize?
        results = self.sam(
            source=img,
            device=device,
            retina_masks=True,
            imgsz=self.config.output_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=self.verbose,
        )

        return results[0].masks.data.to(torch.bool), results[0].boxes.xyxy.to(
            torch.int32
        )


@register_config("segmentation", name="fastsam", constructor=FastSAMSegmentation)
@dataclasses.dataclass
class FastSAMConfig(Config):
    """Configuration for FastSAM."""

    model_name: str = "FastSAM-x.pt"
    confidence: float = 0.55
    iou: float = 0.85
    output_size: int = 736

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class SAMSegmentation(nn.Module):
    """SAM wrapper."""

    def __init__(self, config):
        """Load SAM."""
        super(SAMSegmentation, self).__init__()
        import segment_anything as sam

        self.config = config

        weight_path = models_path() / config.model_name
        model = sam.sam_model_registry["vit_h"](checkpoint=str(weight_path))
        self.sam = sam.SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            crop_n_layers=self.config.crop_n_layers,
            min_mask_region_area=self.config.min_mask_region_area,
        )

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = SAMConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, img):
        """
        Segment image.

        Args:
            img (np.ndarray): uint8 image in [H, W, C] order

        Returns
            Tuple[torch.Tensor, torch.Tensor]: Masks and bounding boxes
        """
        results = self.sam.generate(img)
        N = len(results)
        masks = torch.zeros((N, img.shape[0], img.shape[1]), dtype=torch.bool)
        b_xywh = torch.zeros((N, 4), dtype=torch.float32)
        for idx, r in enumerate(results):
            masks[idx] = torch.from_numpy(r["segmentation"])
            b_xywh[idx] = torch.tensor(r["bbox"])

        return masks, torchvision.ops.box_convert(b_xywh, "xywh", "xyxy")


@register_config("segmentation", name="sam", constructor=SAMSegmentation)
@dataclasses.dataclass
class SAMConfig(Config):
    """Configuration for FastSAM."""

    model_name = "sam_vit_h_4b8939.pth"
    points_per_side: int = 12
    points_per_batch: int = 144
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    min_mask_region_area: int = 100

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DenseFeatures(nn.Module):
    """Module to compute dense features per mask."""

    def __init__(self, model_name):
        """Load f3rm module."""
        super(DenseFeatures, self).__init__()
        from f3rm.features.clip import clip as f3rm_clip

        self.model_name = model_name
        self.model, self.preprocess = f3rm_clip.load(model_name)
        print(self.preprocess)

    def get_output_dims(self, h_in, w_in):
        """Compute output dimensions."""
        # from https://github.com/f3rm/f3rm/blob/main/f3rm/features/clip_extract.py
        if self.model_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
            return h_out, w_out

        if self.model_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            return int(h_out), int(w_out)

        raise ValueError(f"unknown clip model: {self.model_name}")

    def forward(self, img):
        """Compute dense clip embeddings for image."""
        embeddings = self.model.get_patch_encodings(img.unsqueeze(0))
        h_in, w_in = img.size()[-2:]
        h_out, w_out = self.get_output_dims(h_in, w_in)
        return einops.rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)


class ClipWrapper(nn.Module):
    """Quick wrapper around clip to simplifiy interface for encoding images."""

    def __init__(self, config):
        """Load the visual encoder for CLIP."""
        super(ClipWrapper, self).__init__()
        import clip

        self.config = config
        self.model, self._transform = clip.load(config.model_name)
        self._canary_param = nn.Parameter(torch.empty(0))
        self._tokenize = clip.tokenize

    @torch.no_grad()
    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        # TODO(nathan) think about validation
        return self.model.visual(imgs.to(self.model.dtype))

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = ClipConfig()
        config.update(kwargs)
        return cls(config)

    @property
    def model_name(self):
        """Get current model name."""
        return self.config.model_name

    @property
    def input_size(self):
        """Get input patch size for clip."""
        return self.model.visual.input_resolution

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def get_dense_encoder(self):
        """Get corresponding dense encoder."""
        return DenseFeatures(self.model_name)

    @torch.no_grad()
    def embed_text(self, text):
        """Encode text."""
        tokens = self._tokenize(text).to(self.device)
        return self.model.encode_text(tokens)


@register_config("clip", name="clip", constructor=ClipWrapper)
@dataclasses.dataclass
class ClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-L/14"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class OpenClipWrapper(nn.Module):
    """Quick wrapper around openclip to simplifiy image encoding interface."""

    def __init__(self, config):
        """Load the visual encoder for OpenCLIP."""
        super(OpenClipWrapper, self).__init__()
        import open_clip

        self.config = config
        self.model, _, self._transform = open_clip.create_model_and_transforms(
            config.model_name, pretrained=config.pretrained
        )
        self._canary_param = nn.Parameter(torch.empty(0))
        # TODO(nathan) load tokenize function

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenClipConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, imgs):
        """Encode multiple images (without transformation)."""
        return self.visual.model(imgs)

    @property
    def input_size(self):
        """Get input patch size for clip."""
        return self.model.visual.image_size[0]

    @property
    def model_name(self):
        """Get current model name."""
        return self.config.model_name

    @property
    def device(self):
        """Get current model device."""
        return self._canary_param.device

    def get_dense_encoder(self):
        """Get corresponding dense encoder."""
        return None

    @torch.no_grad()
    def embed_text(self, text):
        """Encode text."""
        tokens = self._tokenize(text).to(self.device)
        return self.model.encode_text(tokens)


@register_config("clip", name="open_clip", constructor=OpenClipWrapper)
@dataclasses.dataclass
class OpenClipConfig(Config):
    """Configuration for OpenCLIP."""

    model_name: str = "ViT-H/14"
    pretrained: str = "laion2B_s32B_b79K"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
