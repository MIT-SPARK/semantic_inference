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
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
from spark_config import Config, register_config
from torchvision.ops import box_convert

from semantic_inference import root_path


def models_path():
    """Get path to pre-trained weight storage."""
    return root_path().parent.parent / "models"


def path_to_dot_semantic_inference():
    """Get path to ~/.semantic_inference directory."""
    return os.getenv("HOME") + "/.semantic_inference"


class FastSAMSegmentation(nn.Module):
    """Fast SAM wrapper."""

    def __init__(self, config, verbose=False):
        """Load Fast SAM."""
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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


class Yolov11InstanceSegmenterWrapper(nn.Module):
    """Yolov11 instance segmentation wrapper."""

    def __init__(self, config):
        """Load Yolov11 model."""
        super().__init__()
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.model_name)

    def eval(self):
        """override eval to avoid issues with yolo model"""
        self.model.model.eval()

    @property
    def category_names(self):
        """Get category names."""
        return self.model.names

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = Yolov11InstanceSegmenterConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, img):
        """Segment image."""
        result = self.model(img)[0]  # assume batch size 1
        if result.masks is None:
            return None, None, None, None

        categories = result.boxes.cls.cpu()  # int8
        masks = result.masks.data.to(torch.bool).cpu()  #
        boxes = result.boxes.xyxy.cpu()  # float32
        confidences = result.boxes.conf.cpu()  # float32
        # assume the instance id is the index in the result?
        return categories, masks, boxes, confidences


@register_config(
    "instance_model", name="yolov11", constructor=Yolov11InstanceSegmenterWrapper
)
@dataclasses.dataclass
class Yolov11InstanceSegmenterConfig(Config):
    """Configuration for Yolov11 instance segmenter."""

    model_name: str = "yolo11n-seg.pt"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class GDSam2InstanceSegmenterWrapper(nn.Module):
    """Grounded SAM 2 instance segmentation wrapper."""

    def __init__(self, config):
        """Load Grounded SAM 2 model."""
        super().__init__()
        from groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_prompt = config.text_prompt
        self.multimask_output = config.multimask_output
        self.erosion = config.erosion
        self.erosion_kernel_size = config.erosion_kernel_size

        # hydra config pkg: only need relative path to the pkg installation dir
        sam2_model_config_path = os.path.join(
            "configs/sam2.1", config.sam2_model_config
        )
        sam2_checkpoint_path = os.path.join(
            path_to_dot_semantic_inference(), config.sam2_checkpoint
        )
        grounding_dino_config_path = os.path.join(
            path_to_dot_semantic_inference(),
            "gdsam2_config",
            config.grounding_dino_config,
        )
        grounding_dino_checkpoint_path = os.path.join(
            path_to_dot_semantic_inference(), config.grounding_dino_checkpoint
        )

        # build SAM2 image predictor
        self.sam2_model = build_sam2(sam2_model_config_path, sam2_checkpoint_path)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_checkpoint_path,
            device=self.device,
        )

        # convert text prompt to category names
        self.category_names = self.text_prompt.lower().split(". ")
        self.category_names = [
            cat.strip() for cat in self.category_names if len(cat.strip()) > 0
        ]
        self.category_names[-1] = self.category_names[-1].rstrip(
            "."
        )  # remove the last dot if any

    def preprocess_image(self, img):
        """Preprocess image for Grounded SAM 2.
        Input:
            - img np.ndarray (H, W, C) uint8
        Output:
            - image_transformed torch.Tensor (C, H, W) float32
        """
        import groundingdino.datasets.transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        pil_img = torchvision.transforms.ToPILImage()(img)
        image_transformed, _ = transform(pil_img, None)
        return image_transformed

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = GDSam2InstanceSegmenterConfig()
        config.update(kwargs)
        return cls(config)

    def forward(self, img):
        """Segment image."""
        from groundingdino.util.inference import predict

        # preprocess img
        img_transformed = self.preprocess_image(img)

        # gdino prediction
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=img_transformed,
            caption=self.text_prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.device,
        )

        # if nothing detected
        if boxes.shape[0] == 0:
            return None, None, None, None

        # process the box prompt for SAM 2
        h, w, _ = img.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        """
        Below is the comment from the official sam2 demo:
        FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type=self.device.type, dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs
            (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        """

        # SAM 2 predicts mask
        self.sam2_predictor.set_image(img)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=self.multimask_output,
        )

        # Sample best according to scores if multimask output
        if self.multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]

        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # If needed, apply erosion to masks
        if self.erosion:
            kernel = np.ones((self.config.erosion_kernel_size, self.config.erosion_kernel_size), np.uint8)
            for i in range(masks.shape[0]):
                masks[i] = cv2.erode(masks[i].astype(np.uint8), kernel, iterations=1)
                
        # convert string labels to indexes based on the text prompt
        categories = []
        for label in labels:
            label_str = label.lower()
            if label_str in self.category_names:
                label_indx = self.category_names.index(label_str)
            else:
                label_indx = -1  # unknown
            categories.append(label_indx)
        categories = torch.tensor(categories)

        # convert masks to boolean
        masks = masks.astype(bool)
        masks = torch.tensor(masks)

        # use xyxy boxes
        boxes = torch.tensor(input_boxes)

        return categories, masks, boxes, confidences


@register_config(
    "instance_model", name="gdsam2", constructor=GDSam2InstanceSegmenterWrapper
)
@dataclasses.dataclass
class GDSam2InstanceSegmenterConfig(Config):
    """Configuration for Grounded SAM 2 instance segmenter."""

    text_prompt: str = "car. tire."
    sam2_checkpoint: str = "sam2.1_hiera_large.pt"
    sam2_model_config: str = "sam2.1_hiera_l.yaml"
    grounding_dino_config: str = "GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    multimask_output: bool = False
    erosion: bool = False
    erosion_kernel_size: int = 5

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
