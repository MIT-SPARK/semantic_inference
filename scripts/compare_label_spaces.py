#!/usr/bin/env python3
"""Show label to name mappings."""
import click
import pathlib
import pprint
import yaml
import csv


def _load_catmap(filepath, cat_index=0, name_index=-1):
    with filepath.open("r") as fin:
        delimiter = "\t" if filepath.suffix == ".tsv" else ","
        catreader = csv.reader(fin, delimiter=delimiter)
        catmap = {}

        have_first = False
        for row in catreader:
            if not have_first:
                have_first = True
                continue

            catmap[int(row[cat_index])] = row[name_index]

    return catmap


def _load_groups(filename):
    with filename.expanduser().absolute().open("r") as fin:
        contents = yaml.load(fin.read(), Loader=yaml.SafeLoader)

    names = {index + 1: group["name"] for index, group in enumerate(contents["groups"])}
    return names


@click.command()
@click.argument("category_mapping_file")
@click.argument("grouping_config_file")
@click.option("-n", "--name-index", default=-1, type=int, help="index for name column")
@click.option("-l", "--label-index", default=0, type=int, help="index for label column")
def main(category_mapping_file, grouping_config_file, name_index, label_index):
    category_mapping_path = pathlib.Path(category_mapping_file).resolve()
    if not category_mapping_path.exists():
        click.secho(f"[FATAL]: {category_mapping_path} does not exist!", fg="red")

    grouping_config_path = pathlib.Path(grouping_config_file).resolve()
    if not grouping_config_path.exists():
        click.secho(f"[FATAL]: {grouping_config_path} does not exist!", fg="red")

    groups = _load_groups(grouping_config_path)
    catmap = _load_catmap(
        category_mapping_path, cat_index=label_index, name_index=name_index
    )

    print("{:<39}||{:<39}".format("orig: label → name", "grouping: label → name"))
    print("-" * 80)
    N_max = max(max(groups), max(catmap))
    for i in range(N_max):
        orig_str = f"{i} → {catmap[i]}" if i in catmap else "n/a"
        new_str = f"{i} → {groups[i]}" if i in groups else "n/a"
        print(f"{orig_str:<39}||{new_str:<39}")

    matches = {}
    for index, group in groups.items():
        for label, name in catmap.items():
            if group == name:
                matches[index] = label
                break

    print("")
    print("Matches")
    print("-" * 80)
    for index, group in groups.items():
        if index in matches:
            click.echo(f" - {group}: {index} → {matches[index]}")
        else:
            click.secho(f" - {group}: {index} → ?", fg="red")


if __name__ == "__main__":
    main()
