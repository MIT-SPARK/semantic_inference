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
"""Various visualization utilities."""

import functools
import logging
import math
import random

import distinctipy
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# useful colornames
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


def _int_color(color):
    return [int(math.floor(255.0 * x)) for x in color[:3]]


def _float_color(color):
    return [float(x) / 255.0 for x in color[:3]]


def _colors_equal(c1, c2):
    c1_int = _int_color(c1)
    c2_int = _int_color(c2)
    return c1_int == c2_int


def _line_length(x):
    return len(x) + sum([len(w) for w in x])


def _wrap(label, width=15):
    words = label.split(" ")
    lines = []

    line = []
    for word in words:
        if len(line) == 0:
            line.append(word)
            continue

        curr_length = len(word) + _line_length(line)
        if curr_length > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)

    if len(line) > 0:
        lines.append(" ".join(line))

    return "\n".join(lines)


def make_color(color_type, color_info):
    """
    Make an RGB color from provided information.

    Accepts various color types:
      - `raw`: `color_info` is a 3-channel RGB value between 0 and 1
      - 'raw_int`: `color_info` is a 3-channel integer RGB value between 0 and 255
      - `expanded`: `color_info` is a color name from the EXPANDED_COLORS map
      - `mpl`: `color_info` is a color name in the CSS4_COLORS palette

    Args:
        color_type (str): Color "factory" to use
        color_info (Union[List[Union[int, float]], str]): Raw RGB values or color name

    Returns:
        List[float]: Three-channel RGB value between 0 and 1.
    """
    if color_type == "raw":
        return [float(x) for x in color_info[:3]]

    if color_type == "raw_int":
        return _float_color(color_info)

    if color_type == "expanded":
        color = EXPANDED_COLORS.get(color_info)
        if color is None:
            logging.warning(f"failed to find '{color_info}' in EXPANDED_COLORS")
            return None

        return [x / 255.0 for x in color]

    if color_type == "mpl":
        color = matplotlib.colors.CSS4_COLORS.get(color_info)
        if color is not None:
            logging.warning(f"failed to find {color_info} in CSS4_COLORS")
            return None

        return [x for x in matplotlib.colors.to_rgb(color)]

    logging.warning(f"invalid color type '{color_type}'")
    return None


def make_palette(label_config, base_colors=None, seed=None):
    """Make a color palette for the provided label space."""
    colors_by_group = {
        group["name"]: make_color(group.get("source"), group.get("color"))
        for group in label_config["specified_colors"]
    }
    colors_by_group = {k: v for k, v in colors_by_group.items() if v is not None}
    N_groups = len(label_config["groups"])
    N_new = N_groups - len(colors_by_group)

    palette = [_float_color(c) for c in base_colors] if base_colors is not None else []
    if len(base_colors) < N_new:
        if base_colors is not None:
            logging.warning("invalid palette size ({len(base_colors)} < {N_new}")

        group_colors = [v for _, v in colors_by_group.items()]
        chosen_colors = base_colors + group_colors
        num_to_find = N_groups - len(chosen_colors)
        # we also don't want white and black as colors
        new_colors = distinctipy.get_colors(
            num_to_find,
            exclude_colors=chosen_colors + [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        )
        base_colors += new_colors

    if seed is not None:
        random.seed(seed)
        random.shuffle(palette)

    colors = [_int_color(x) for x in palette]

    color_index = 0
    for group in label_config["groups"]:
        group_name = group["name"]

        if group_name in colors_by_group:
            continue

        colors_by_group[group_name] = [float(x) for x in colors[color_index][:3]]
        color_index += 1

    return colors_by_group


def verify_palette(color_palette):
    """Check that every label has a distinct color."""
    invalid = set([])
    for name, color in color_palette.items():
        for other, other_color in color_palette.items():
            if name == other:
                continue

            if not _colors_equal(color, other_color):
                continue

            invalid.add(name)
            invalid.add(other)
            logging.error(f"[ERROR]: {name} and {other} share color {color}")

    return len(invalid) == 0


def make_colormap_legend(
    colormap,
    nrows=None,
    ncols=4,
    cell_width=60,
    cell_height=22,
    dpi=300,
    fontsize=8,
    wrapsize=14,
    offset=1,
    alpha=1.0,
    sort=False,
):
    """
    Create a color legend using.

    Args:
        colormap (Dict[str, Tuple[float, float, float]]): Map between names and colors
        nrows (int): Number of rows to force in the grid
        ncols (Optional[int]): Number of columns to try to force in the grid
        cell_width (int): Width of the background rectangle
        cell_height (int): Height of the background rectangle
        dpi (int): Resolution of the legend
        fontsize (int): Text size of each label
        wrapsize (int): Maximum characters per line before wrapping takes effect
        offset (int): Offset of label from edge of cell
        alpha (int): Overall alpha of the colormap
        sort (bool): Sort cell order alphabetically
    """
    N = len(colormap)
    if nrows is None or nrows * ncols < N:
        nrows = N // ncols + int(N % ncols > 0)

    width = cell_width * ncols
    height = cell_height * nrows
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.subplots()
    ax.set_axis_off()

    fig.subplots_adjust(0.0, 0.0, 1.0, 1.0)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)

    labels = sorted([x for x in colormap]) if sort else [x for x in colormap]
    for i, label in enumerate(labels):
        # figure out index in grid
        col = i % ncols
        row = i // ncols
        # compute cell coordinates
        start_x = cell_width * col
        y = row * cell_height

        # get background and appropriate foreground colors
        color = colormap[label][:3]
        prod = (1 / alpha) * functools.reduce(lambda x, y: x * y, color, 1)
        textcolor = (0.3, 0.3, 0.3) if prod > 0.1 else (1.0, 1.0, 1.0)

        text = _wrap(label, width=wrapsize)
        ax.text(
            start_x + offset,
            y,
            text,
            fontsize=fontsize,
            horizontalalignment="left",
            verticalalignment="center",
            color=textcolor,
        )

        ax.add_patch(
            Rectangle(
                xy=(start_x, y - cell_height / 2),
                width=cell_width,
                height=cell_height,
                facecolor=color,
                alpha=alpha,
            )
        )

    return fig
