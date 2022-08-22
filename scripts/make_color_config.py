#!/usr/bin/env python3
"""Script to make color config."""
import matplotlib.colors as mcolors
import seaborn as sns
import pathlib
import random
import click
import math
import yaml
import sys


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
    else:
        base_colors = sns.color_palette(palette, num_to_assign)

    if len(base_colors) < num_to_assign:
        if "palette" in label_config:
            click.secho(
                "[WARN]: invalid palette size ({len(base_colors)} < {num_to_assign}",
                fg="yellow",
            )
        base_colors = sns.color_palette(palette, num_to_assign)

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


def _write_configs(label_config, group_colors, output_dir, config_name):
    output_path = pathlib.Path(output_dir).resolve().absolute()
    yaml_output = output_path / f"{config_name}.yaml"
    csv_output = output_dir / f"{config_name}.csv"

    default_color = [0.0, 0.0, 0.0]
    if "default_color" in label_config:
        default_color = [float(x) for x in label_config["default_color"]]

    config = {"classes": [], "default_color": default_color}

    for index, group in enumerate(label_config["groups"]):
        group_name = group["name"]
        config["classes"].append(index)
        config_prefix = "class_info/{}".format(index)
        config[f"{config_prefix}/color"] = group_colors[group_name]
        # network and csv are off by 1
        config[f"{config_prefix}/labels"] = [x - 1 for x in group["labels"]]

    with yaml_output.open("w") as fout:
        fout.write(yaml.dump(config))

    with csv_output.open("w") as fout:
        fout.write("name,red,green,blue,alpha,id\r\n")
        for index, group in enumerate(label_config["groups"]):
            name = group["name"]
            color = _int_color(group_colors[name])
            fout.write(f"{name},{color[0]},{color[1]},{color[2]},255,{index}\r\n")


@click.command()
@click.argument("category_groups_file")
@click.argument("output_dir")
@click.option("-p", "--palette", default="husl", help="fallback color palette")
@click.option("-s", "--seed", default=0, type=int, help="random seed")
@click.option("-n", "--config-name", default=None, help="output config name")
@click.option(
    "-f",
    "--force-compatible",
    default=False,
    is_flag=True,
    help="use same palette size as original scripts",
)
def main(
    category_groups_file, output_dir, palette, seed, config_name, force_compatible
):
    """Export a config."""
    category_groups_path = pathlib.Path(category_groups_file).expanduser().absolute()
    if not category_groups_path.exists():
        click.secho(
            f"[FATAL]: invalid category group file {category_groups_path}", fg="red"
        )
        sys.exit(1)

    output_path = pathlib.Path(output_dir).expanduser().absolute()
    if not output_path.exists():
        click.secho(f"[FATAL]: invalid output path {output_path}", fg="red")
        sys.exit(1)

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
    _write_configs(label_config, colors, output_path, name_to_use)


if __name__ == "__main__":
    main()
