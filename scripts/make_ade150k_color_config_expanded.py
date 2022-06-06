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
        50, #fire/fireplace
        79, #arcade/machine
        110, #pool
        114, #waterfall
    ],
    "Sky": [3], #sky
    "Tree": [5, 73], #tree, palm;palm;tree,
    "Water": [22, 27, 61, 129], #water, sea, river, lake
    "Ground": [10, 14, 30, 47, 95], #grass, earth/ground, field, sand, land/ground/soil
    "Sidewalk": [12, 141], #sidewalk, pier/dock
    "Road": [7, 53, 55, 92], #road, path, runway, dirt/track
    "Vehicle": [21, 81, 84, 91, 103, 117], #car, bus/passenger vehicle, truck, airplane, van
    "Building": [2, 26, 49, 52, 80, 85, 115], #building/edifice, house, skyscraper, grandstand/covered stand, hut/shack, tower, tent
    "Signal": [137], #traffic signal
    "Rock": [35], #rock/stone
    "Fence": [33], #fence
    "Boat": [77, 104], #boat, ship
    "Sign": [124], #trade name/brand name
    "Slope": [17, 69], #mountain/mount, hill
    "Bridge": [62], #bridge
    "Wall": [1],
    "Floor": [4, 29, 102, 106],
    "Ceiling": [6, 87, 107],
    "Door": [15, 59],
    "Stairs": [54, 60, 97, 122],
    "Structure": [19, 39, 41, 43, 64, 89, 94, 96, 105, 123], #19: curtain/drape, #123: storage tank
    "Window": [9],
    "Shelf": [25, 63, 71, 74, 78, 100],
    "Plant": [18, 67, 136], #18: plant;flora;;plant;life, #67: flower
    "Bed": [8, 58, 118],
    "Storage": [11, 36, 42, 45, 56, 112, 113, 116],
    "Table": [16, 34, 46, 57, 65],
    "Chair": [20, 31, 32, 76, 111],
    "Wall_Decoration": [23, 28, 44, 101, 145, 149, 150], #150: flag
    "Couch": [24, 40, 70, 98], #70: bench
    "Light": [37, 83, 86, 88, 135, 140], #88: street light, #140: fan
    "Appliance": [38, 48, 51, 66, 72, 108, 119, 125, 130, 134, 146, 147], #38: bathtub, #66: toilet
    "Thing": [
        68, #book
        75, #computer
        90, #tv
        109, #toy
        120, #ball
        128, #bike
        131, #screen
        133, #sculpture
        138, #tray
        139, #wastebin
        142, #crt;screen
        144, #monitor
        99, #kitchen stuff starts here, #99: bottle
        121, #food
        126, #pot/flowerpot
        136, #vase
        143, #plate
        148, #drinking glass
    ],
    "Deformable": [82, 93, 132], #82: towel, #93: clothes, #blanket
    "Dynamic_NonHuman": [127], #127: animal
    "Human": [13]
}

COLOR_PALLETE = {
    "black": [0,0,0],
    "white": [255,255,255],
    "blue": [0,0,225],
    "orange": [255,140,0],
    "green": [0,200,0],
    "red": [225,0,0],
    "purple": [120,0,120],
    "brown": [139,69,19],
    "grey": [128,128,128],
    "yellow": [225,225,0],
    "magenta": [225,0,225],
    "cyan": [0,255,255],
    "lightgrey": [220,220,220],
    "darkgrey": [100,100,100],
    "lightblue": [135,206,235],
    "darkblue": [0,0,128],
    "lightgreen": [0,255,0],
    "darkgreen": [0,108,0],
    "lightred": [255,120,120],
    "darkred": [128,0,0],
    "lightpurple": [186,85,211],
    "darkpurple": [75,0,130]
}

KNOWN_COLORS = {
    "Sky": COLOR_PALLETE["lightblue"],
    "Water": COLOR_PALLETE["blue"],
    "Tree": COLOR_PALLETE["darkgreen"],
    "Plant": COLOR_PALLETE["lightgreen"],
    "Vehicle": COLOR_PALLETE["magenta"],
    "Rock": COLOR_PALLETE["black"],
    "Ground": COLOR_PALLETE["brown"],
    "Slope": COLOR_PALLETE["yellow"],
    "Road": COLOR_PALLETE["darkgrey"],
    "Sidewalk": COLOR_PALLETE["lightgrey"],
    "Floor": COLOR_PALLETE["grey"],
    "Wall": COLOR_PALLETE["darkpurple"],
    "Building": COLOR_PALLETE["lightpurple"],
    "Couch": COLOR_PALLETE["red"],
    "Table": COLOR_PALLETE["magenta"],
    "Window": COLOR_PALLETE["lightred"]
}

def main():
    """Export a config."""
    parser = argparse.ArgumentParser(description="utility for creating color configs.")
    parser.add_argument("output_dir", help="config output dir")
    parser.add_argument("--config_name", "-n", default="ade150_expanded_config")
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

    known_colors = {}
    for label in KNOWN_COLORS:
        known_colors[label] = (KNOWN_COLORS[label][0] / 255.0, 
            KNOWN_COLORS[label][1] / 255.0, KNOWN_COLORS[label][2] / 255.0)
    # TODO(yun) might want to check for overlaps between random generate color and known colors

    keys = []
    for index, group_name in enumerate(ADE_LABEL_GROUPS):
        keys.append(index)
        config_prefix = "class_info/{}".format(index)
        if group_name in known_colors:
            config["{}/color".format(config_prefix)] = [
                known_colors[group_name][0],
                known_colors[group_name][1],
                known_colors[group_name][2],
            ]
        else :
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
            if group_name in known_colors:
                writer.writerow(
                    {
                        "name": group_name,
                        "red": int(math.floor(255.0 * known_colors[group_name][0])),
                        "green": int(math.floor(255.0 * known_colors[group_name][1])),
                        "blue": int(math.floor(255.0 * known_colors[group_name][2])),
                        "alpha": 255,
                        "id": index,
                    }
                )
            else:
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
