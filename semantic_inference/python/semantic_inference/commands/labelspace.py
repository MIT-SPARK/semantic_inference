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
"""Show label to name mappings."""

import csv
import pathlib

import click
import rich.console
import rich.table
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)
yaml.default_flow_style = False


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
        contents = yaml.load(fin.read())

    info = {}
    for index, group in enumerate(contents["groups"]):
        for label in group["labels"]:
            info[label] = (index, group["name"])

    return info


def _make_label_table(groups, catmap):
    table = rich.table.Table(title="Label Mapping")
    table.add_column("Original Label")
    table.add_column("Original Name(s)", max_width=30)
    table.add_column("New Label")
    table.add_column("New Name")

    for i in range(max(catmap)):
        label = f"{i}"
        orig_names = ", ".join(catmap.get(i, "--").split(";"))

        group_info = groups.get(i)
        if group_info is None:
            table.add_row(label, orig_names, "--", "--")
        else:
            table.add_row(label, orig_names, str(group_info[0]), group_info[1])

    return table


@click.group(name="labelspace")
def cli():
    """Subcommands dealing with labelspace configuration."""
    pass


@cli.command()
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

    console = rich.console.Console()
    console.print(_make_label_table(groups, catmap))
