import os
import argparse

import torch
import torch.nn as nn
import torch.onnx as onnx

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.lib.nn import async_copy_to
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


def export(segmentation_module, gpu, image_dim, output, opset_version=12):
    segmentation_module.eval()
    image_height, image_width = image_dim

    segmentation_module.set_size(image_dim)

    with torch.no_grad():
        fake_img = torch.randn(1, 3, image_height, image_width)
        fake_img_gpu = async_copy_to(fake_img, gpu)

        print("Exporting")
        onnx.export(
            segmentation_module,
            fake_img_gpu,
            output,
            opset_version=opset_version,
        )
        print("Exported")


def main(cfg, gpu, image_dim, output):
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

    export(segmentation_module, gpu, image_dim, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx exporter for mit semseg models")
    parser.add_argument("config", metavar="FILE", help="path to config file", type=str)
    parser.add_argument(
        "weights_dir", metavar="WEIGHTS_DIR", help="path to weights", type=str
    )
    parser.add_argument("--weights_checkpoint", "-w", default="epoch_30.pth")
    parser.add_argument(
        "--output", "-o", default="model.onnx", help="onnx model output path"
    )
    parser.add_argument("--height", default=360, help="image input height", type=int)
    parser.add_argument("--width", default=640, help="image input width", type=int)
    parser.add_argument(
        "--opset_version", "-v", default=12, help="onnx opset version to use", type=int
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    cfg.MODEL.weights_encoder = os.path.join(
        args.weights_dir, "encoder_{}".format(args.weights_checkpoint)
    )
    cfg.MODEL.weights_decoder = os.path.join(
        args.weights_dir, "decoder_{}".format(args.weights_checkpoint)
    )

    assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(
        cfg.MODEL.weights_decoder
    ), "checkpoint does not exitst!"

    main(cfg, 0, (args.height, args.width), args.output)
