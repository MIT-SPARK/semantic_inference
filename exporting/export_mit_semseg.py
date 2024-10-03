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
"""Download and export pretrained models."""

import click
import pathlib
import subprocess
import tempfile
import importlib
import warnings
import yaml
import sys

import torch
import torch.nn as nn
import torch.onnx as onnx


def _normalized_path(filepath):
    return pathlib.Path(filepath).expanduser().absolute()


def _check_checkpoint_path(filepath):
    filepath = pathlib.Path(filepath).expanduser().absolute()
    if not filepath.exists():
        click.secho("Invalid filepath: '{filepath}'", fg="red")
        sys.exit(1)


def _default_path():
    return _normalized_path(tempfile.gettempdir()) / "sr_model_weights"


def _config_path(repo_path):
    return _normalized_path(repo_path) / "config"


def _model_path():
    return pathlib.Path(__file__).absolute().parent.parent / "models"


def _load_model_info(config_path):
    with config_path.open("r") as fin:
        config = yaml.safe_load(fin.read())
        # model name, checkpoint
        return config["DIR"].split("/")[1], config["TEST"]["checkpoint"]


class ExportableSegmentationModel(nn.Module):
    """Wrapper around base model to hide some things."""

    def __init__(self, repo_path, config_path, weights_path, dims):
        """Pass-through constructor."""
        module_path = _normalized_path(repo_path) / "mit_semseg" / "__init__.py"
        spec = importlib.util.spec_from_file_location("mit_semseg", module_path)
        mit_semseg = importlib.util.module_from_spec(spec)
        sys.modules["mit_semseg"] = mit_semseg
        spec.loader.exec_module(mit_semseg)

        from mit_semseg.models import ModelBuilder
        from mit_semseg.config import cfg

        super(ExportableSegmentationModel, self).__init__()

        weights_path = _normalized_path(weights_path)

        cfg.merge_from_file(config_path)
        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
        cfg.MODEL.weights_encoder = str(weights_path / "encoder.pth")
        _check_checkpoint_path(cfg.MODEL.weights_encoder)
        cfg.MODEL.weights_decoder = str(weights_path / "decoder.pth")
        _check_checkpoint_path(cfg.MODEL.weights_decoder)

        self.encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder,
        )
        self.decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True,
        )

    def forward(self, img):
        """Pass-through forward pass grabbing the size from the input data."""
        x = self.encoder(img, return_feature_maps=True)
        scores = self.decoder(x, segSize=img.shape[-2:])
        _, pred = torch.max(scores, dim=1)
        return pred


def _export_legacy(model, img, output_path, opset_version=None):
    onnx.export(
        model,
        img,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
        do_constant_folding=False,
        opset_version=opset_version,
    )


def _export_dynamo(model, img, output_path, opset_version=None):
    opts = onnx.ExportOptions(dynamic_shapes=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*running_var.*")
        warnings.filterwarnings("ignore", ".*running_mean.*")
        warnings.filterwarnings("ignore", ".*only implements opset.*")
        warnings.filterwarnings("ignore", ".*no underlying reference*")
        prog = onnx.dynamo_export(model, img, export_options=opts)
    with output_path.open("wb") as fout:
        prog.save(fout)


def _export(
    repo_path, output_path, image_dim, dry_run=False, opset_version=None, device=None
):
    device = "cuda" if device is None else device
    configs = [x for x in _config_path(repo_path).glob("*.yaml")]
    configs = {_load_model_info(x)[0]: x for x in configs}

    models = [x for x in output_path.iterdir() if x.is_dir()]
    for model_path in models:
        model_output = _model_path() / f"{model_path.stem}.onnx"
        config_path = configs[model_path.stem]
        click.secho(f"Exporting {model_path.stem}...", fg="green")

        model = ExportableSegmentationModel(
            repo_path, config_path, model_path, image_dim
        )
        model = model.to(device)
        model = model.eval()

        with torch.no_grad():
            image_height, image_width = image_dim
            img = torch.randn(1, 3, image_height, image_width).to(device)
            try:
                _export_legacy(model, img, model_output, opset_version=opset_version)
                # _export_dynamo(model, img, model_output, opset_version=opset_version)
                click.secho(f"Exported {model_path.stem}!", fg="green")
            except Exception as e:
                click.secho(f"Export failed: {e}", fg="red")


def _download(repo_path, output_path, dry_run=False):
    base_url = "http://sceneparsing.csail.mit.edu/model/pytorch"
    click.echo(f"Downloading from {base_url}")

    configs = [x for x in _config_path(repo_path).iterdir()]

    output_path.mkdir(parents=True, exist_ok=True)
    for config_path in configs:
        model_name, checkpoint = _load_model_info(config_path)
        encoder_url = f"{base_url}/{model_name}/encoder_{checkpoint}"
        decoder_url = f"{base_url}/{model_name}/decoder_{checkpoint}"
        model_out = output_path / model_name
        model_out.mkdir(exist_ok=True)
        encoder_args = ["wget", "-O", f"{model_out / 'encoder.pth'}", encoder_url]
        decoder_args = ["wget", "-O", f"{model_out / 'decoder.pth'}", decoder_url]

        if dry_run:
            click.secho(" ".join(encoder_args), fg="green")
            click.secho(" ".join(decoder_args), fg="green")
        else:
            subprocess.run(encoder_args)
            subprocess.run(decoder_args)


@click.group(chain=True)
@click.option("--output", "-o", default=None)
@click.option("--dry-run", "-d", is_flag=True)
@click.pass_context
def main(ctx, output, dry_run):
    """Entry point for commands."""
    output_path = _default_path() if output is None else _normalized_path(output)

    ctx.ensure_object(dict)
    ctx.obj["output"] = output_path
    ctx.obj["dry_run"] = dry_run


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.pass_context
def download(ctx, repo_path):
    """Download all pretrained models."""
    output_path = ctx.obj.get("output", _default_path)
    dry_run = ctx.obj.get("dry_run", False)
    _download(repo_path, output_path, dry_run=dry_run)


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--height", default=360, help="image input height", type=int)
@click.option("--width", default=640, help="image input width", type=int)
@click.option("--opset_version", "-v", default=17, type=int)
@click.pass_context
def export(ctx, repo_path, height, width, opset_version):
    """Load and export model."""
    output_path = ctx.obj.get("output", _default_path)
    dry_run = ctx.obj.get("dry_run", False)
    _export(
        repo_path,
        output_path,
        (height, width),
        dry_run=dry_run,
        opset_version=opset_version,
    )


if __name__ == "__main__":
    main()
