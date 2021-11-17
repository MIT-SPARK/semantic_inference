#!/usr/bin/env python3
"""Script to make color config."""
import seaborn as sns
import pathlib
import argparse
import random
import math
import yaml
import csv


ADE_LABEL_GROUPS = {
    "Unknown": [
        3,
        7,
        10,
        12,
        14,
        17,
        21,
        22,
        26,
        27,
        30,
        33,
        35,
        38,
        47,
        49,
        50,
        52,
        53,
        55,
        61,
        62,
        69,
        77,
        79,
        80,
        81,
        84,
        85,
        91,
        92,
        95,
        103,
        104,
        110,
        114,
        115,
        117,
        123,
        124,
        129,
        137,
        141,
    ],
    "Wall": [1, 2],
    "Floor": [4, 29, 102, 106],
    "Ceiling": [6, 87, 107],
    "Door": [15, 59],
    "Stairs": [54, 60, 97, 122],
    "Structure": [9, 19, 39, 41, 43, 64, 89, 94, 96, 105],
    "Shelf": [25, 63, 71, 74, 78, 100],
    "Plant": [5, 18, 67, 73, 126, 136],
    "Bed": [8, 58, 118],
    "Storage": [11, 36, 42, 45, 56, 112, 113, 116],
    "Table": [16, 34, 46, 57, 65],
    "Chair": [20, 31, 32, 76, 111],
    "Wall_Decoration": [23, 28, 44, 101, 145, 149, 150],
    "Couch": [24, 40, 70, 98],
    "Light": [37, 83, 86, 88, 135, 140],
    "Appliance": [48, 51, 66, 72, 108, 119, 125, 130, 134, 146, 147],
    "Thing": [
        68,
        75,
        90,
        109,
        120,
        128,
        131,
        133,
        138,
        139,
        142,
        144,
        99,  # kitchen stuff starts here
        121,
        143,
        148,
    ],
    "Deformable": [82, 93, 132],
    "Dynamic_NonHuman": [127],
    "Human": [13],
}


def main():
    """Export a config."""
    parser = argparse.ArgumentParser(description="utility for creating color configs.")
    parser.add_argument("output_dir", help="config output dir")
    parser.add_argument("--config_name", "-n", default="ade150_config")
    parser.add_argument(
        "-use_hsl", action="store_true", help="use hsl color palette (instead of husl)"
    )
    parser.add_argument("--seed", "-s", default=0, help="random seed")
    args = parser.parse_args()

    if args.use_hsl:
        colors = sns.color_palette("hsl", len(ADE_LABEL_GROUPS))
    else:
        colors = sns.color_palette("husl", len(ADE_LABEL_GROUPS))

    random.seed(args.seed)
    random.shuffle(colors)

    config = {}

    keys = []
    for index, group_name in enumerate(ADE_LABEL_GROUPS):
        keys.append(index)
        config_prefix = "class_info/{}".format(index)
        config["{}/color".format(config_prefix)] = [
            colors[index][0],
            colors[index][1],
            colors[index][2],
        ]
        # network and csv are off by 1
        config["{}/labels".format(config_prefix)] = [
            x - 1 for x in ADE_LABEL_GROUPS[group_name]
        ]

    output_path = pathlib.Path(args.output_dir).resolve().absolute()

    config["classes"] = keys
    config["default_color"] = [colors[0][0], colors[0][1], colors[0][2]]
    with open(str(output_path / "{}.yaml".format(args.config_name)), "w") as fout:
        fout.write(yaml.dump(config))

    with open(str(output_path / "{}.csv".format(args.config_name)), "w") as fout:
        fields = ["name", "red", "green", "blue", "alpha", "id"]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for index, group_name in enumerate(ADE_LABEL_GROUPS):
            writer.writerow(
                {
                    "name": group_name,
                    "red": int(math.floor(255.0 * colors[index][0])),
                    "green": int(math.floor(255.0 * colors[index][1])),
                    "blue": int(math.floor(255.0 * colors[index][2])),
                    "alpha": 255,
                    "id": index,
                }
            )


if __name__ == "__main__":
    main()
