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
from dataclasses import dataclass

import detectron2.data.transforms as T
import numpy as np
import requests
import spark_config as sc
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from ruamel.yaml import YAML
from tqdm import tqdm

from semantic_inference_closed_set_zoo.third_party.mask2former import (
    add_maskformer2_config,
)

yaml = YAML(typ="safe", pure=True)


def _get_config_path(dataset, model):
    curr_path = pathlib.Path(__file__).absolute().parent
    config_path = curr_path / "third_party" / "mask2former" / "config"
    return config_path / dataset / "semantic-segmentation" / f"{model}.yaml"


def _download(url, filepath, block_size=1024, verbose=True):
    print(f"Downloading model from {url} to {filepath}...")

    r = requests.get(url, stream=True)
    bar_args = {
        "total": int(r.headers.get("content-length", 0)),
        "unit": "B",
        "unit_scale": True,
    }
    with tqdm(**bar_args) as bar, filepath.open("wb") as fout:
        for chunk in r.iter_content(chunk_size=block_size):
            fout.write(chunk)
            bar.update(len(chunk))

    print("Finished download!")


class Mask2Former:
    """Wrapper around Mask2Former model."""

    def __init__(self, config):
        weight_dir = pathlib.Path("~/.semantic_inference/mask2former").expanduser()
        weight_dir.mkdir(exist_ok=True, parents=True)
        weight_path = weight_dir / config.weight_name
        if not weight_path.exists():
            url_path = pathlib.Path(__file__).absolute().parent / "model_weights.yaml"
            urls = yaml.load(url_path)
            model_url = urls[config.dataset][config.model]
            _download(model_url, weight_path)

        model_config = _get_config_path(config.dataset, config.model)
        if not model_config.exists():
            raise ValueError(f"Model config does not exist at '{model_config}'")

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(model_config)
        cfg.freeze()

        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(str(weight_path))
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    @torch.no_grad()
    def segment(self, img):
        """Run inference. img should be in RGB order"""
        height, width = img.shape[:2]
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return (
            self.model([inputs])[0]["sem_seg"]
            .argmax(dim=0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )


@dataclass
@sc.register_config("closed_set_model", name="Mask2Former", constructor=Mask2Former)
class Mask2FormerConfig(sc.Config):
    """Configuration for Mask2Former."""

    dataset: str = "ade20k"
    model: str = "swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640"

    @property
    def weight_name(self):
        return f"{self.dataset}-{self.model.replace('/', '-')}.pkl"
