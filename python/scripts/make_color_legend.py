"""Show colors from csv."""
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib
import click
import csv


# adapted from https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_colortable(
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
    """Plot a color table."""
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


@click.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.option("-c", "--num-cols", type=int, default=4)
@click.option("-d", "--dpi", type=float, default=100.0)
@click.option("-o", "--output", default=None)
@click.option("-a", "--alpha", type=float, default=0.8)
def main(csv_file, num_cols, dpi, output, alpha):
    """Plot labels and colors."""
    color_dict = {}
    name_dict = {}

    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

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

    plot_colortable(color_dict, name_dict, desiredcols=num_cols, dpi=dpi, alpha=alpha)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
