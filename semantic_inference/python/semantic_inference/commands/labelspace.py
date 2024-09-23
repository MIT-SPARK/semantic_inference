"""Show label to name mappings."""

import csv
import pathlib

import click
import yaml


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


def _show_labels(groups, catmap):
    print("{:<39}||{:<39}".format("orig: label → name", "grouping: label → name"))
    print("-" * 80)
    N_max = max(max(groups), max(catmap))
    for i in range(N_max):
        orig_str = f"{i} → {catmap[i]}" if i in catmap else "n/a"
        new_str = f"{i} → {groups[i]}" if i in groups else "n/a"
        print(f"{orig_str:<39}||{new_str:<39}")


def _get_match_str(name, index, matches):
    if index in matches:
        match_str = f" - {name}: {index} → {matches[index]}"
        return f"{match_str:<39}"

    match_str = f" - {name}: {index} → ?"
    match_str = f"{match_str:<39}"
    return click.style(match_str, fg="red")


def _show_matches(groups, catmap, group_matches, cat_matches):
    print(
        "{:<39}||{:<39}".format(
            "orig name: label → match", "grouping name: label → match"
        )
    )
    print("-" * 80)
    N_max = max(max(groups), max(catmap))
    for i in range(N_max):
        orig_str = _get_match_str(catmap[i], i, cat_matches) if i in catmap else "n/a"
        new_str = _get_match_str(groups[i], i, group_matches) if i in groups else "n/a"
        print(f"{orig_str}||{new_str}")


@click.group(name="labelspace")
def cli():
    """Subcommands dealing with labelspace configuration."""
    pass


@click.command()
@click.argument("labelspace", type=click.Path(exists=True))
@click.argument("grouping", type=click.Path(exists=True))
@click.option("-n", "--name-index", default=-1, type=int, help="index for name column")
@click.option("-l", "--label-index", default=0, type=int, help="index for label column")
def compare(labelspace, grouping, name_index, label_index):
    """Compare original LABELSPACE to label grouping in GROUPING."""
    labelspace_path = pathlib.Path(labelspace).resolve()
    grouping_config_path = pathlib.Path(grouping).resolve()

    groups = _load_groups(grouping_config_path)
    catmap = _load_catmap(labelspace_path, cat_index=label_index, name_index=name_index)

    group_matches = {}
    for index, group in groups.items():
        for label, name in catmap.items():
            if group == name:
                group_matches[index] = label
                break

    cat_matches = {}
    for index, group in catmap.items():
        for label, name in groups.items():
            if group == name:
                cat_matches[index] = label
                break

    _show_labels(groups, catmap)
    print("")
    _show_matches(groups, catmap, group_matches, cat_matches)
