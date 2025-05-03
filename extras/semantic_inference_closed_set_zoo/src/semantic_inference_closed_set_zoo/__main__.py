"""Test entry point for testing maskformer2."""

import click

from semantic_inference_closed_set_zoo import Mask2Former, Mask2FormerConfig


@click.command()
@click.argument("weights_file", type=click.Path(exists=True))
def main(weights_file):
    """Test maskformers2."""
    config = Mask2FormerConfig()
    config.weights_file = weights_file
    model = Mask2Former(config)
    print(model)


if __name__ == "__main__":
    main()
