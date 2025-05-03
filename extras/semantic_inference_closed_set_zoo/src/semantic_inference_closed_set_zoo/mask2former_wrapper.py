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
"""Run inference with oneformer."""

import pathlib

import detectron2.data.transforms as T
import spark_config as sc
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from semantic_inference_closed_set_zoo.third_party.mask2former import (
    add_maskformer2_config,
)


def _get_config_path(dataset, model):
    curr_path = pathlib.Path(__file__).absolute().parent
    config_path = curr_path / "third_party" / "mask2former" / "config"
    return config_path / dataset / "semantic-segmentation" / f"{model}.yaml"


class Mask2Former:
    """Wrapper around Mask2Former model."""

    def __init__(self, config):
        model_config = _get_config_path(config.dataset, config.model)
        if not model_config.exists():
            raise ValueError(f"Model config does not exist at '{model_config}'")

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_list(
            [
                "MODEL.MASK_FORMER.TEST.INSTANCE_ON",
                False,
                "MODEL.MASK_FORMER.TEST.PANOPTIC_ON",
                False,
            ]
        )
        cfg.merge_from_file(model_config)
        cfg.freeze()

        self.model = build_model(cfg)
        self.model.eval()

        DetectionCheckpointer(self.model).load(config.weight_file)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    @torch.no_grad()
    def segment(self, img):
        """Run inference."""
        height, width = img.shape[:2]
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        return self.model([inputs])[0]["sem_seg"].argmax(dim=0).cpu().numpy()


@sc.register_config("closed_set_model", name="Mask2Former", constructor=Mask2Former)
class Mask2FormerConfig:
    """Configuration for Mask2Former."""

    dataset: str = "ade20k"
    model: str = "swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640"
    weight_file: str = ""
