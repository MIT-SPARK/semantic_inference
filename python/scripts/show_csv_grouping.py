#!/usr/bin/env python3
"""Read in a color csv and show mapping between label and names."""
import pandas as pd
import pathlib
import pprint
import click
import sys


@click.command()
@click.argument("color_file")
def main(color_file):
    """Run everything."""
    color_path = pathlib.Path(color_file).expanduser().absolute()
    if not color_path.exists():
        click.secho(f"[FATAL]: {color_path} does not exist", fg="red")
        sys.exit(1)

    label_df = pd.read_csv(color_file)
    unique_labels = sorted(label_df["id"].unique().tolist())
    print(f"Labels: {unique_labels}\n")

    for label in unique_labels:
        names = label_df.loc[label_df["id"] == label, "name"].tolist()
        print(f"label {label}:")
        pprint.pprint(names)


if __name__ == "__main__":
    main()
