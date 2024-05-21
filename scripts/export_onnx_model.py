"""Export models via onnx."""

import pathlib
import click
import sys

import torch
import torch.nn as nn
import torch.onnx as onnx

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.config import cfg


class ExportableSegmentationModule(SegmentationModule):
    """Wrapper around base model to hide some things."""

    def __init__(self, *args, **kwargs):
        """Pass-through constructor."""
        super(ExportableSegmentationModule, self).__init__(*args, **kwargs)

    def set_size(self, seg_size):
        """Cache segmentation size."""
        self.seg_size = seg_size

    def forward(self, img):
        """Pass-through forward pass grabbing the size from the input data."""
        scores = self.decoder(
            self.encoder(img, return_feature_maps=True), segSize=self.seg_size
        )

        _, pred = torch.max(scores, dim=1)
        return pred.squeeze(0)


def export(model, gpu, image_dim, output, opset_version=None):
    """Export a model."""
    model.eval()
    image_height, image_width = image_dim

    model.set_size(image_dim)

    with torch.no_grad():
        fake_img = torch.randn(1, 3, image_height, image_width).to(gpu)

        print("Exporting")
        onnx.export(model, fake_img, output, opset_version=opset_version)
        print("Exported")


def _check_checkpoint_path(filepath):
    filepath = pathlib.Path(filepath).expanduser().absolute()
    if not filepath.exists():
        click.secho("Invalid filepath: '{filepath}'", fg="red")
        sys.exit(1)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("weights_dir", type=click.Path(exists=True))
@click.option("--checkpoint", "-c", default="epoch_30.pth")
@click.option("--output", "-o", default="model.onnx")
@click.option("--height", default=360, help="image input height", type=int)
@click.option("--width", default=640, help="image input width", type=int)
@click.option("--opset_version", "-v", default=None, type=int)
@click.option("--gpu", "-g", default=0, type=int)
def main(
    config_path,
    weights_dir,
    checkpoint,
    output,
    height,
    width,
    opset_version,
    gpu,
):
    """Load and export model."""
    cfg.merge_from_file(config_path)
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    weights_dir = pathlib.Path(weights_dir).expanduser().absolute()
    cfg.MODEL.weights_encoder = str(weights_dir / f"encoder_{checkpoint}")
    _check_checkpoint_path(cfg.MODEL.weights_encoder)
    cfg.MODEL.weights_decoder = str(weights_dir, f"decoder_{checkpoint}")
    _check_checkpoint_path(cfg.MODEL.weights_decoder)

    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
    )
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True,
    )

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = ExportableSegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()

    export(segmentation_module, gpu, (height, width), output)


if __name__ == "__main__":
    main()
