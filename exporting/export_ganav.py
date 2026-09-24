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
"""Export GA-Nav traversability models.

Must be run from a python environment with GA-Nav (mmseg 0.21) installed, e.g.
the environment described in the GA-Nav README. The exported model expects an
RGB image of a fixed size normalized with mean [0.485, 0.456, 0.406] and stddev
[0.229, 0.224, 0.225] and returns GA-Nav group labels at the input resolution.

GA-Nav's test pipeline resizes the image (keeping the aspect ratio) and zero pads
it to the network size. mmseg then upsamples the padded logits straight to the
input size, which misaligns the labels for any input that does not share the
network's aspect ratio (e.g., 16:9 cameras). The exported model reproduces the
resize and pad, but crops the padding before upsampling.
"""

import pathlib
import sys

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx as onnx

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDDEV = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _rescale_size(input_size, model_size):
    """Get the keep-ratio size matching mmcv.imrescale for (height, width) sizes."""
    scale = min(
        max(model_size) / max(input_size), min(model_size) / min(input_size)
    )
    return tuple(int(x * scale + 0.5) for x in input_size)


class ExportModel(nn.Module):
    """GA-Nav segmentor for a fixed input size."""

    def __init__(self, segmentor, model_size, input_size):
        """Construct the model."""
        super().__init__()
        self.model = segmentor
        self.model_size = model_size
        self.input_size = input_size
        self.scaled_size = _rescale_size(input_size, model_size)

    def forward(self, img):
        """Run inference."""
        # GA-Nav's class attention has a fixed feature map size, so the network
        # input has to be the (padded) size it was built with
        h, w = self.scaled_size
        x = F.interpolate(img, size=self.scaled_size, mode="bilinear")
        x = F.pad(x, (0, self.model_size[1] - w, 0, self.model_size[0] - h))
        ret = self.model.encode_decode(x, None)
        ret = F.interpolate(ret[:, :, :h, :w], size=self.input_size, mode="bilinear")
        return torch.argmax(ret, dim=1).to(torch.int32)


def _load_segmentor(ganav_root, config, checkpoint):
    sys.path.insert(0, str(ganav_root))
    from mmcv.cnn.utils import revert_sync_batchnorm
    from mmseg.apis import init_segmentor

    segmentor = init_segmentor(str(config), str(checkpoint), device="cpu")
    segmentor = revert_sync_batchnorm(segmentor)
    segmentor.eval()
    return segmentor


def _model_size(segmentor):
    # size of the padded test input, e.g. (300, 375) for RUGD
    for step in segmentor.cfg.data.test.pipeline[1]["transforms"]:
        if step["type"] == "Pad":
            return tuple(step["size"])

    raise ValueError("unable to determine network input size from test pipeline")


def _to_tensor(rgb):
    img = (rgb.astype(np.float32) / 255.0 - MEAN) / STDDEV
    return torch.as_tensor(img.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _reference(model, rgb):
    """Run GA-Nav's own preprocessing (mmcv resize on uint8), cropping the padding."""
    import mmcv

    h, w = model.scaled_size
    scaled = mmcv.imresize(rgb, (w, h))
    padded = np.zeros((*model.model_size, 3), dtype=np.uint8)
    padded[:h, :w] = scaled
    img = _to_tensor(padded)
    # native pads after normalizing, so padding is zero in normalized space
    img[:, :, h:, :] = 0.0
    img[:, :, :, w:] = 0.0
    with torch.no_grad():
        ret = model.model.encode_decode(img, None)[:, :, :h, :w]
        ret = F.interpolate(ret, size=model.input_size, mode="bilinear")
        return torch.argmax(ret, dim=1).to(torch.int32).numpy()


def _validate(model, model_path, image_paths, num_classes):
    import cv2
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    export_agree = []
    reference_agree = []
    for path in image_paths:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != model.input_size:
            print(f"skipping {path.name}: size {rgb.shape[:2]} != {model.input_size}")
            continue

        img = _to_tensor(rgb)
        with torch.no_grad():
            expected = model(img).numpy()

        result = session.run(["output"], {"input": img.numpy()})[0]
        export_agree.append(np.mean(expected == result))
        reference_agree.append(np.mean(_reference(model, rgb) == result))

        hist = np.bincount(result.flatten(), minlength=num_classes)
        hist = hist / hist.sum()
        hist_str = " ".join(f"{x:.2f}" for x in hist)
        print(
            f"{path.name}: onnx_vs_torch={export_agree[-1]:.4f} "
            f"onnx_vs_ganav={reference_agree[-1]:.4f} classes=[{hist_str}]"
        )

    if len(export_agree) == 0:
        raise click.ClickException("no validation images matched the input size")

    ratio = np.mean(export_agree)
    print(f"overall onnx vs torch: {ratio:.5f}")
    print(f"overall onnx vs ganav preprocessing: {np.mean(reference_agree):.5f}")
    return ratio


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--ganav-root", default="~/mit/tmp/GANav-offroad")
@click.option("--output", default="~/.semantic_inference/ganav_rugd_group6.onnx")
@click.option("--width", default=640, help="camera image width")
@click.option("--height", default=360, help="camera image height")
@click.option("--validate", "images", type=click.Path(exists=True), multiple=True)
@click.option("--min-agreement", default=0.99)
def main(config, checkpoint, ganav_root, output, width, height, images, min_agreement):
    """Run export."""
    ganav_root = pathlib.Path(ganav_root).expanduser().absolute()
    config = pathlib.Path(config).absolute()
    checkpoint = pathlib.Path(checkpoint).absolute()
    segmentor = _load_segmentor(ganav_root, config, checkpoint)

    model = ExportModel(segmentor, _model_size(segmentor), (height, width))
    model.eval()
    print(
        f"classes: {segmentor.CLASSES}, input: {model.input_size}, "
        f"scaled: {model.scaled_size}, network: {model.model_size}"
    )

    model_path = pathlib.Path(output).expanduser().absolute()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    img = torch.randn(1, 3, height, width, dtype=torch.float32)
    with torch.no_grad():
        onnx.export(
            model,
            img,
            str(model_path),
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            opset_version=17,
            dynamo=False,
        )

    import onnx as onnx_lib

    onnx_lib.checker.check_model(str(model_path))
    print(f"exported model to '{model_path}'")

    if len(images) == 0:
        return

    image_paths = []
    for path in images:
        path = pathlib.Path(path)
        image_paths += sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
        if path.is_file():
            image_paths.append(path)

    ratio = _validate(model, model_path, image_paths, len(segmentor.CLASSES))
    if ratio < min_agreement:
        raise click.ClickException(f"agreement {ratio:.5f} < {min_agreement}")


if __name__ == "__main__":
    main()
