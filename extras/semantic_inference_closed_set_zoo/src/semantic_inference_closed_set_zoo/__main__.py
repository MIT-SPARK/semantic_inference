"""Test entry point for testing maskformer2."""

import pathlib

import click
import distinctipy
import imageio.v3 as iio
import numpy as np

from semantic_inference_closed_set_zoo import Mask2Former, Mask2FormerConfig


@click.command()
@click.argument("images", type=click.Path(exists=True), nargs=-1)
def main(images):
    """Test maskformers2."""
    config = Mask2FormerConfig()
    model = Mask2Former(config)
    images = [pathlib.Path(x).expanduser().absolute() for x in images]

    colors = distinctipy.get_colors(
        150, exclude_colors=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    )
    colors = (255 * np.array(colors)).astype(np.uint8).T

    for image_path in images:
        img = iio.imread(image_path)
        labels = model.segment(img)
        output = np.take(colors, labels, mode="raise", axis=1)
        output = np.transpose(output, [1, 2, 0])
        iio.imwrite(image_path.parent / f"{image_path.stem}_labels.png", output)


if __name__ == "__main__":
    main()
