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
"""Subcommands for handling colormaps."""

import csv
import pathlib
import pprint
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from semantic_inference.visualization import (
    make_colormap_legend,
    make_palette,
    verify_palette,
)


def _normalize_path(path):
    return pathlib.Path(path).expanduser().absolute()


@click.group(name="color")
def cli():
    """Subcommands for color-related operations."""
    pass


@cli.command()
@click.argument("categories", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-s", "--seed", default=None, type=int, help="random seed")
@click.option("-n", "--config-name", default=None, help="output config name")
def create(categories, output, seed):
    """Create a color map from CATEGORIES and output it to OUTPUT."""
    categories = _normalize_path(categories)
    output = _normalize_path(output)

    with categories.open("r") as fin:
        config = yaml.load(fin.read(), Loader=yaml.SafeLoader)

    base_colors = []
    if "palette" in config:
        for color in config["palette"]:
            base_colors.append([int(x) for x in color])

    # TODO(nathan) convert label config to a better input to make_palette
    colors = make_palette(config, base_palette=base_colors, seed=seed)
    if not verify_palette(colors):
        click.secho("[FATAL]: color collision in color palette", fg="red")
        sys.exit(1)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    with output.open("w") as fout:
        fout.write("name,red,green,blue,alpha,id\r\n")
        for idx, group in enumerate(config["groups"]):
            name = group["name"]
            color = colors["name"]
            fout.write(f"{name},{color[0]},{color[1]},{color[2]},255,{idx}\r\n")


@cli.command()
@click.argument("colormap", type=click.Path(exists=True))
def show(colormap):
    """Show a COLORMAP csv file by label."""
    color_path = pathlib.Path(colormap).expanduser().absolute()
    label_df = pd.read_csv(color_path)
    unique_labels = sorted(label_df["id"].unique().tolist())
    print(f"Labels: {unique_labels}\n")

    for label in unique_labels:
        names = label_df.loc[label_df["id"] == label, "name"].tolist()
        print(f"label {label}:")
        pprint.pprint(names)


@cli.command()
@click.argument("colormap", type=click.Path(exists=True))
@click.option("-l", "--labelspace", type=click.Path(exists=True), help="label to names")
@click.option("-o", "--output", default=None, help="output filepath")
@click.option("-p", "--plot", is_flag=True, help="plot legend")
@click.option("--skip-header", is_flag=True, help="skip first csv row")
@click.option("-r", "--num-rows", type=int, default=None, help="number of grid rows")
@click.option("-c", "--num-cols", type=int, default=5, help="number of grid columns")
@click.option("-d", "--dpi", type=float, default=600.0, help="output DPI")
@click.option("-a", "--alpha", type=float, default=1.0, help="visualizer alpha")
@click.option("-w", "--cell-width", type=float, default=600, help="cell width (px)")
@click.option("-h", "--cell-height", type=float, default=200, help="cell height (px)")
@click.option("-f", "--fontsize", type=int, default=12, help="text fontsize")
@click.option("-t", "--wrap", type=int, default=14, help="characters before wrapping")
@click.option("--offset", type=float, default=10, help="pixel offset for text")
def legend(
    colormap,
    labelspace,
    output,
    plot,
    skip_header,
    num_rows,
    num_cols,
    dpi,
    alpha,
    cell_width,
    cell_height,
    fontsize,
    wrap,
    offset,
):
    """Create a legend for COLORMAP showing correspondences between names and colors."""
    label_to_name = {}
    label_to_color = {}

    def _read_color(line):
        return np.squeeze(np.array([float(x) for x in line[1:4]]) / 255.0)

    with _normalize_path(colormap).open("r") as fin:
        reader = csv.reader(fin)
        next(reader)
        for line in reader:
            name = line[0]
            label = int(line[-1])
            label_to_name[label] = name
            label_to_color[label] = _read_color(line)

    if labelspace is not None:
        label_to_name = {}
        with _normalize_path(labelspace).open("r") as fin:
            contents = yaml.safe_load(fin.read())
            for x in contents["label_names"]:
                label_to_name[x["label"]] = x["name"]

    name_to_color = {v: label_to_color[k] for k, v in label_to_name.items()}
    fig = make_colormap_legend(
        name_to_color,
        nrows=num_rows,
        ncols=num_cols,
        dpi=dpi,
        alpha=alpha,
        cell_width=cell_width,
        cell_height=cell_height,
        fontsize=fontsize,
        offset=offset,
        wrapsize=wrap,
    )

    if plot:
        fig.show()
        plt.show(block=True)

    if output is not None:
        fig.savefig(_normalize_path(output), dpi=dpi)
