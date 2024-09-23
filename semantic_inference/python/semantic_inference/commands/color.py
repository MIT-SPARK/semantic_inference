"""Subcommands for handling colormaps."""

import csv
import math
import pathlib
import pprint
import random
import sys

import click
import distinctipy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle

EXPANDED_COLORS = {
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "blue": [0, 0, 225],
    "orange": [255, 140, 0],
    "green": [0, 200, 0],
    "red": [225, 0, 0],
    "purple": [120, 0, 120],
    "brown": [139, 69, 19],
    "grey": [128, 128, 128],
    "yellow": [225, 225, 0],
    "magenta": [225, 0, 225],
    "cyan": [0, 255, 255],
    "lightgrey": [220, 220, 220],
    "darkgrey": [100, 100, 100],
    "lightblue": [135, 206, 235],
    "darkblue": [0, 0, 128],
    "lightgreen": [0, 255, 0],
    "darkgreen": [0, 108, 0],
    "lightred": [255, 120, 120],
    "darkred": [128, 0, 0],
    "lightpurple": [186, 85, 211],
    "darkpurple": [75, 0, 130],
}


def _read_label_config(category_groups_path):
    with category_groups_path.open("r") as fin:
        config = yaml.load(fin.read(), Loader=yaml.SafeLoader)
    return config


def _int_color(color):
    return [int(math.floor(255.0 * x)) for x in color[:3]]


def _colors_equal(c1, c2):
    c1_int = _int_color(c1)
    c2_int = _int_color(c2)
    return c1_int == c2_int


def _verify_palette(colors_by_group):
    invalid_groups = set([])
    for group, color in colors_by_group.items():
        for other_group, other_color in colors_by_group.items():
            if group == other_group:
                continue

            if not _colors_equal(color, other_color):
                continue

            invalid_groups.add(group)
            invalid_groups.add(other_group)
            click.secho(
                f"[ERROR]: {group} and {other_group} share color {_int_color(color)}",
                fg="red",
            )

    return len(invalid_groups) == 0


def _get_specified_color(group_color):
    if group_color["source"] == "raw":
        return [float(x) for x in group_color["color"][:3]]

    if group_color["source"] == "raw_int":
        return [float(x) / 255.0 for x in group_color["color"][:3]]

    color_name = group_color["color"]
    if group_color["source"] == "expanded":
        if color_name not in EXPANDED_COLORS:
            click.secho(f"failed to find {color_name} in EXPANDED_COLORS", fg="red")
            return None

        return [x / 255.0 for x in EXPANDED_COLORS[color_name][:3]]

    if group_color["source"] == "mpl":
        if color_name not in mcolors.CSS4_COLORS:
            click.secho(f"failed to find {color_name} in CSS4_COLORS", fg="red")
            return None

        return [x for x in mcolors.to_rgb(mcolors.CSS4_COLORS[color_name])]


def _get_palette(label_config, palette, seed, force_compatible=False):
    colors_by_group = {
        group_color["name"]: _get_specified_color(group_color)
        for group_color in label_config["specified_colors"]
    }
    N_groups = len(label_config["groups"])
    num_to_assign = N_groups if force_compatible else N_groups - len(colors_by_group)

    base_colors = []
    if "palette" in label_config:
        for color in label_config["palette"]:
            base_colors.append([float(x) for x in color])

    if len(base_colors) < num_to_assign:
        if "palette" in label_config:
            click.secho(
                "[WARN]: invalid palette size ({len(base_colors)} < {num_to_assign}",
                fg="yellow",
            )

        distinct = label_config.get("distinct", False)
        if not distinct:
            base_colors = sns.color_palette(palette, num_to_assign)
        else:
            group_colors = [v for _, v in colors_by_group.items()]
            chosen_colors = base_colors + group_colors
            num_to_find = N_groups - len(chosen_colors)
            # we also don't want white and black as colors
            new_colors = distinctipy.get_colors(
                num_to_find,
                exclude_colors=chosen_colors + [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            )
            base_colors += new_colors

    random.seed(seed)
    random.shuffle(base_colors)

    color_index = 0
    for group in label_config["groups"]:
        group_name = group["name"]

        if group_name in colors_by_group:
            if force_compatible:
                color_index += 1
            continue

        colors_by_group[group_name] = [float(x) for x in base_colors[color_index][:3]]
        color_index += 1

    return colors_by_group


def _write_config(label_config, group_colors, output_dir, config_name, csv_shift=0):
    csv_output = output_dir / f"{config_name}.csv"
    with csv_output.open("w") as fout:
        fout.write("name,red,green,blue,alpha,id\r\n")
        for index, group in enumerate(label_config["groups"]):
            name = group["name"]
            color = _int_color(group_colors[name])
            idx = index + csv_shift
            fout.write(f"{name},{color[0]},{color[1]},{color[2]},255,{idx}\r\n")


# TODO(nathan) use better implementation
# adapted from https://matplotlib.org/stable/gallery/color/named_colors.html
def _plot_colortable(
    color_dict,
    name_dict,
    sort_colors=True,
    desiredcols=4,
    emptycols=0,
    cell_width=205,
    cell_height=22,
    swatch_width=22,
    margin=12,
    dpi=300,
    alpha=0.8,
):
    N = len(color_dict)
    ncols = desiredcols - emptycols
    nrows = N // ncols + int(N % ncols > 0)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    labels = sorted([x for x in color_dict])
    for i, label in enumerate(labels):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            f"{name_dict[label]}",
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        color = color_dict[label].tolist() + [alpha]
        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=color,
                edgecolor="k",
            )
        )

    plt.tight_layout()
    return fig


@click.group(name="color")
def cli():
    """Subcommands for color-related operations."""
    pass


@cli.command()
@click.argument("categories", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=True))
@click.option("-p", "--palette", default="husl", help="fallback color palette")
@click.option("-s", "--seed", default=0, type=int, help="random seed")
@click.option("-n", "--config-name", default=None, help="output config name")
@click.option("--csv-shift", type=int, default=0, help="amount to add to color csv")
@click.option(
    "-f",
    "--force-compatible",
    default=False,
    is_flag=True,
    help="use same palette size as original scripts",
)
def create(
    categories,
    output,
    palette,
    seed,
    config_name,
    csv_shift,
    force_compatible,
):
    """Create a color map from CATEGORIES and output it to OUTPUT."""
    category_groups_path = pathlib.Path(categories).expanduser().absolute()
    output_path = pathlib.Path(output).expanduser().absolute()

    label_config = _read_label_config(category_groups_path)
    colors = _get_palette(
        label_config, palette, seed, force_compatible=force_compatible
    )
    if not _verify_palette(colors):
        click.secho("[FATAL]: color collision in color palette", fg="red")
        sys.exit(1)

    name_to_use = (
        config_name if config_name is not None else label_config.get("name", "test")
    )
    if name_to_use == "test":
        click.secho(
            "[WARN]: config name not specifying, defaulting to test", fg="yellow"
        )
    _write_config(
        label_config,
        colors,
        output_path,
        name_to_use,
        csv_shift=csv_shift,
    )


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
@click.option("-c", "--num-cols", type=int, default=4)
@click.option("-d", "--dpi", type=float, default=100.0)
@click.option("-o", "--output", default=None)
@click.option("-a", "--alpha", type=float, default=0.8)
def plot(colormap, num_cols, dpi, output, alpha):
    """Plot COLORMAP showing correspondences between names and colors."""
    color_dict = {}
    name_dict = {}

    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    csv_path = pathlib.Path(colormap)
    with csv_path.open("r") as fin:
        reader = csv.reader(fin)
        next(reader)

        for line in reader:
            label = int(line[-1])
            name_dict[label] = line[0]
            color_dict[label] = np.squeeze(
                np.array([float(x) for x in line[1:4]]) / 255
            )

    _plot_colortable(color_dict, name_dict, desiredcols=num_cols, dpi=dpi, alpha=alpha)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight", dpi=300)
