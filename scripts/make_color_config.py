#!/usr/bin/env python3
"""Script to make color config."""
import seaborn as sns
import pathlib
import random
import click
import math
import yaml
import sys


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

            invalid_groups.insert(group)
            invalid_groups.insert(other_group)
            click.secho(
                f"[ERROR]: {group} and {other_group} share color {_int_color(color)}",
                fg="red",
            )

    return len(invalid_groups) == 0


def _get_palette(label_config, palette, seed, force_compatible=False):
    colors_by_group = {
        group_color["name"]: [float(x) for x in group_color["color"]]
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
        fout.write("name,red,green,blue,alpha,id\n")
        for index, group in enumerate(label_config["groups"]):
            name = group["name"]
            color = _int_color(group_colors[name])
            fout.write(f"{name},{color[0]},{color[1]},{color[2]},255,{index}\n")


@click.command()
@click.argument("category_groups_file")
@click.argument("output_dir")
@click.option("-p", "--palette", default="husl", help="fallback color palette")
@click.option("-s", "--seed", default=0, type=int, help="random seed")
@click.option("-n", "--config-name", default="test_config", help="output config name")
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
        click.echo("[FATAL]: color collision in color palette")
        sys.exit(1)

    _write_configs(label_config, colors, output_path, config_name)


if __name__ == "__main__":
    main()
