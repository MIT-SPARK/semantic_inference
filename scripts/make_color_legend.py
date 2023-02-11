"""Show colors from csv."""
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import click
import csv


# adapted from https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_colortable(
    color_dict,
    name_dict,
    sort_colors=True,
    emptycols=0,
    cell_width=212,
    cell_height=22,
    swatch_width=48,
    margin=12,
):
    """Plot a color table."""
    N = len(color_dict)
    ncols = 4 - emptycols
    nrows = N // ncols + int(N % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * 4)
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
            f"{label}: {name_dict[label]}",
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=color_dict[label],
                edgecolor="0.7",
            )
        )

    return fig


@click.command()
@click.argument("csv_file", type=click.Path(exists=True))
def main(csv_file):
    """Plot labels and colors."""
    color_dict = {}
    name_dict = {}

    csv_path = pathlib.Path(csv_file)
    with csv_path.open("r") as fin:
        reader = csv.reader(fin)
        next(reader)

        for line in reader:
            label = int(line[-1])
            name_dict[label] = line[0]
            color_dict[label] = np.squeeze(
                np.array([float(x) for x in line[1:4]]) / 255
            )

    plot_colortable(color_dict, name_dict)
    plt.show()


if __name__ == "__main__":
    main()
