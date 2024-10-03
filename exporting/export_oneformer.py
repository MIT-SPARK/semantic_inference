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
"""Script to export oneformer to onnx."""

import click
import cv2
import pathlib
import functools
import numpy as np
import onnxruntime as ort

import torch
from torch import nn
import torch.onnx as onnx
from torch.nn import functional as F

import detectron2
import detectron2.checkpoint
import detectron2.projects.deeplab


def _get_oneformer(path_to_repo):
    import importlib
    import sys

    module_path = path_to_repo / "oneformer" / "__init__.py"
    spec = importlib.util.spec_from_file_location("oneformer", module_path)
    oneformer = importlib.util.module_from_spec(spec)
    sys.modules["oneformer"] = oneformer
    spec.loader.exec_module(oneformer)
    return oneformer


def _setup_cfg(oneformer, config_path):
    cfg = detectron2.config.get_cfg()
    detectron2.projects.deeplab.add_deeplab_config(cfg)
    oneformer.add_common_config(cfg)
    oneformer.add_swin_config(cfg)
    oneformer.add_dinat_config(cfg)
    oneformer.add_convnext_config(cfg)
    oneformer.add_oneformer_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.IS_TRAIN", "False"])
    cfg.freeze()
    return cfg


def resize_image(img, image_size: int):
    """Get image of correct size."""
    # resize to match input size
    dims = torch.tensor(img.size()[-2:])
    ratio = image_size / torch.min(dims)
    ndims = torch.round(ratio * dims).to(torch.int32)
    return F.interpolate(img, [int(ndims[0]), int(ndims[1])], mode="bilinear")


def pad_image(img, grid_size: int):
    """Pad image to token size."""
    # pads to make divisible by 32
    dims = torch.tensor(img.size()[-2:])
    padded_dims = torch.ceil(dims / grid_size) * grid_size
    paddings = padded_dims - dims
    padding_size = (0, int(paddings[1]), 0, int(paddings[0]))
    return F.pad(img, padding_size)


class OneFormer(nn.Module):
    """OneFormer custom model."""

    def __init__(
        self,
        path_to_repo,
        config_path,
    ):
        """Construct the model."""
        super().__init__()

        path_to_repo = pathlib.Path(path_to_repo).expanduser().absolute()
        oneformer = _get_oneformer(path_to_repo)
        cfg = _setup_cfg(oneformer, path_to_repo / "configs" / config_path)
        cfg = cfg.clone()

        self.backbone = detectron2.modeling.build_backbone(cfg)
        self.sem_seg_head = detectron2.modeling.build_sem_seg_head(
            cfg, self.backbone.output_shape()
        )

        self.task_mlp = (
            oneformer.modeling.transformer_decoder.oneformer_transformer_decoder.MLP(
                cfg.INPUT.TASK_SEQ_LEN,
                cfg.MODEL.ONE_FORMER.HIDDEN_DIM,
                cfg.MODEL.ONE_FORMER.HIDDEN_DIM,
                2,
            )
        )

        self.tokenizer = oneformer.data.tokenizer.Tokenize(
            oneformer.data.tokenizer.SimpleTokenizer(),
            max_seq_len=cfg.INPUT.TASK_SEQ_LEN,
        )

        self.image_size = cfg.INPUT.MIN_SIZE_TEST
        self.grid_size = cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY
        if self.grid_size < 0:
            # use backbone size_divisibility if not set
            self.grid_size = self.backbone.size_divisibility

        def _get_view(arr):
            return torch.Tensor(arr).view(-1, 1, 1)

        self.register_buffer("pixel_mean", _get_view(cfg.MODEL.PIXEL_MEAN), False)
        self.register_buffer("pixel_std", _get_view(cfg.MODEL.PIXEL_STD), False)

    @torch.no_grad()
    def transform(self, img):
        """Convert image to input format model expects."""
        img = img.to(self.pixel_mean.device)
        img = resize_image(img, self.image_size)
        img = (img - self.pixel_mean) / self.pixel_std
        img = pad_image(img, self.grid_size)
        return img

    @torch.no_grad()
    def init_task(self):
        """Initialize task encoding."""
        task = "The task is semantic"
        task = self.tokenizer(task).to(self.pixel_mean.device).unsqueeze(0)
        task = self.task_mlp(task.float())
        self.register_buffer("task", task, False)

    def forward(self, img):
        """Run inference."""
        # preprocess image

        # run model
        features = self.backbone(img)
        outputs = self.sem_seg_head(features, self.task)

        # upsample masks
        mask_cls = outputs["pred_logits"][0]
        mask_pred = F.interpolate(
            outputs["pred_masks"],
            size=(img.shape[2], img.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # get predictions
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        r = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return r.argmax(dim=0)


def _export_legacy(model, img, output_path, opset_version=17):
    script_model = torch.jit.script(model)
    onnx.export(
        script_model,
        img,
        output_path,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
        opset_version=opset_version,
    )


def _compare(predictions):
    predictions = predictions.astype(np.int16)
    gt_pred = np.squeeze(np.load("/home/ubuntu/test_img.npy"))
    size = (gt_pred.shape[1], gt_pred.shape[0])

    predictions = cv2.resize(predictions, size, interpolation=cv2.INTER_NEAREST)
    total = functools.reduce(lambda x, y: x * y, predictions.shape, 1)
    diff = np.sum(predictions != gt_pred)
    ratio = diff / total
    print(f"Difference: {ratio:%}")


@click.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path())
@click.argument("weights_path", type=click.Path(exists=True))
@click.argument("input_path", type=click.Path(exists=True))
def main(repo_path, config_path, weights_path, input_path):
    """Run OneFormer."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = OneFormer(repo_path, config_path)
    model.eval()
    # model = model.to("cuda")

    checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
    checkpointer.load(weights_path)
    model.init_task()

    image = cv2.imread(input_path)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).unsqueeze(0)

    model_dir = pathlib.Path(__file__).absolute().parent.parent / "models"
    model_path = model_dir / "oneformer.onnx"

    # note: assumes RGB order input
    with torch.no_grad():
        # get padded and resized image
        img = model.transform(image)
        # export onnx model
        _export_legacy(model, img, str(model_path))

        predictions = model(img).cpu().numpy()
        _compare(predictions)

    del model

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    ort_predictions = session.run(None, {"input": img.cpu().numpy()})[0]
    print(ort_predictions)
    _compare(ort_predictions)


if __name__ == "__main__":
    main()
