import os
import argparse

import torch
import torch.nn as nn
import torch.onnx as onnx

from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import setup_logger
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


def export(segmentation_module, gpu, image_dim):
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
            "model.onnx",
            opset_version=12,
        )
        print("Exported")


def main(cfg, gpu, image_dim):
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

    export(segmentation_module, gpu, image_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx exporter for mit semseg models")
    parser.add_argument("--height", default=360, help="image input height", type=int)
    parser.add_argument("--width", default=640, help="image input width", type=int)

    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, "encoder_" + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, "decoder_" + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(
        cfg.MODEL.weights_decoder
    ), "checkpoint does not exitst!"

    main(cfg, 0, (args.height, args.width))
