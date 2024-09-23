"""Entry points for semantic inference."""

import click

import semantic_inference.commands.color as color
import semantic_inference.commands.labelspace as labelspace
import semantic_inference.commands.model as model


@click.group()
def cli():
    """Toolkit for manipulating and examining labelspaces and models."""
    pass


cli.add_command(color.cli)
cli.add_command(labelspace.cli)
cli.add_command(model.cli)


if __name__ == "__main__":
    cli()
