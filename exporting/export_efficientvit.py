"""Export efficientvit models."""

import efficientvit
import efficientvit.seg_model_zoo

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import click
import imageio.v3

import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.nn.functional as F


class ExportModel(nn.Module):
    """OneFormer custom model."""

    def __init__(self, weight_path, name="l2", dataset="ade20k"):
        """Construct the model."""
        super().__init__()
        self.model = efficientvit.seg_model_zoo.create_seg_model(
            name=name, dataset=dataset, weight_url=weight_path
        )

    def forward(self, img):
        """Run inference."""
        img = F.interpolate(img, size=(512, 512), mode="bilinear")
        ret = self.model(img)
        ret = F.interpolate(ret, size=(img.shape[2], img.shape[3]), mode="bilinear")
        return torch.argmax(ret, dim=1)


def _export_legacy(model, img, output_path, opset_version=17):
    onnx.export(
        model,
        img,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
        do_constant_folding=True,
        opset_version=opset_version,
    )


@click.command()
@click.argument("weight_path", type=click.Path(exists=True))
@click.option("--name", default="l2")
@click.option("--dataset", default="ade20k")
@click.option("--input-path", type=click.Path(exists=True), default=None)
def main(weight_path, name, dataset, input_path):
    """Run export."""
    model = ExportModel(weight_path, name=name, dataset=dataset)
    model.eval()

    if input_path is None:
        img = torch.randn(1, 3, 480, 640, dtype=torch.float32)
    else:
        img = imageio.v3.imread(input_path)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).unsqueeze(0)
        img /= 255.0

    print(img.shape)

    model_dir = pathlib.Path(__file__).absolute().parent.parent / "models"
    model_path = model_dir / f"{dataset}-efficientvit_seg_{name}.onnx"

    with torch.no_grad():
        _export_legacy(model, img, str(model_path))

        pred = model(img).cpu().numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("RGB")
    ax[0].imshow(img.squeeze(0).numpy().transpose(1, 2, 0) / 255.0)
    ax[1].set_title("Classes")
    ax[1].imshow(np.squeeze(pred))
    plt.show()


if __name__ == "__main__":
    main()
