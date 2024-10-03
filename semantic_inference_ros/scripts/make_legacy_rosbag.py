#!/usr/bin/env python3
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
"""Script to make a new rosbag with semantics for a given color image topic."""
from cv_bridge import CvBridge
import numpy as np
import pathlib
import rosbag
import click
import yaml
import tqdm
import zmq
import cv2


def _get_config_dir():
    path_to_script = pathlib.Path(__file__).absolute()
    return path_to_script.parent.parent / "config"


def _to_color(rgb_float_values):
    return (255 * np.array([float(x) for x in rgb_float_values])).astype(np.uint8)


def _make_color_map(config_path, colors):
    with (config_path / "colors" / f"{colors}.yaml").open("r") as fin:
        contents = yaml.load(fin.read(), Loader=yaml.SafeLoader)

    class_colors = {}
    class_labels = {}
    all_labels = []
    default_color = (
        _to_color(contents["default_color"])
        if "default_color" in contents
        else np.zeros(3, dtype=np.uint8)
    )
    for key in contents:
        parts = key.split("/")
        if len(parts) < 3:
            continue

        if parts[2] == "color":
            color = _to_color(contents[key])
            class_colors[int(parts[1])] = color
            continue

        if parts[2] == "labels":
            for x in contents[key]:
                class_labels[int(x)] = int(parts[1])
                all_labels.append(int(x))
            continue

    all_labels = sorted(all_labels)
    N = max(all_labels)
    table = np.zeros((N + 1, 3), dtype=np.uint8)
    for i in range(0, N + 1):
        if i in class_labels:
            table[i, :] = class_colors[class_labels[i]]
        else:
            table[i, :] = default_color

    return table.T


def _apply_color(labels, colormap):
    semantic_img = np.take(colormap, labels, mode="raise", axis=1)
    return np.transpose(semantic_img, [1, 2, 0])


def _run_inference(socket, image):
    socket.send_pyobj(image)
    return socket.recv_pyobj()


@click.command()
@click.argument("path_to_bag", type=click.Path(exists=True))
@click.argument("topic")
@click.option("--labels-topic", default="/semantic_color/labels/image_raw")
@click.option("--semantics-topic", default="/semantic_color/semantics/image_raw")
@click.option("--colors", default="ade20k_mp3d")
@click.option("--compression", default="bz2")
@click.option("-p", "--port", default=5555, type=int)
@click.option("-u", "--url", default="127.0.0.1", type=str)
@click.option("-n", "--total", default=None, type=int)
@click.option("--is-compressed", is_flag=True, default=False)
@click.option("--write-colors", is_flag=True, default=False)
@click.option("-w", "--write-every-n", default=0, type=int)
@click.option("-e", "--infer-every-n", default=0, type=int)
@click.option("-a", "--overlay-alpha", default=0.5, type=float)
@click.option("-y", "--force-overwrite", is_flag=True, default=False)
@click.option("-o", "--output", default=None, type=str)
def main(
    path_to_bag,
    topic,
    labels_topic,
    semantics_topic,
    colors,
    compression,
    port,
    url,
    total,
    is_compressed,
    write_colors,
    write_every_n,
    infer_every_n,
    overlay_alpha,
    force_overwrite,
    output,
):
    """Run everything."""
    path_to_bag = pathlib.Path(path_to_bag).expanduser().absolute()
    config_path = _get_config_dir()
    colormap = _make_color_map(config_path, colors)
    bridge = CvBridge()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{url}:{port}")

    if output is None:
        new_path = path_to_bag.parent / f"{path_to_bag.stem}_semantics.bag"
    else:
        new_path = pathlib.Path(output).expanduser().absolute()

    if new_path.exists() and not force_overwrite:
        click.confirm(f"output path {new_path} exists! overwrite: ", abort=True)

    bag_out = rosbag.Bag(str(new_path), "w", compression=compression)
    num_written = 0
    num_read = 0
    with rosbag.Bag(str(path_to_bag), "r") as bag:
        N_msgs = bag.get_message_count(topic)
        for topic, msg, t in tqdm.tqdm(bag.read_messages(topics=[topic]), total=N_msgs):
            if is_compressed:
                img = bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
            else:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            num_read += 1
            if infer_every_n != 0 and (num_read - 1) % infer_every_n != 0:
                continue

            labels = _run_inference(socket, img)
            labels = labels.astype(np.uint16)
            labels_msg = bridge.cv2_to_imgmsg(labels, encoding="mono16")
            labels_msg.header = msg.header
            bag_out.write(labels_topic, labels_msg, t)

            semantics = _apply_color(labels, colormap)

            if write_colors:
                semantics_msg = bridge.cv2_to_imgmsg(semantics, encoding="rgb8")
                semantics_msg.header = msg.header
                bag_out.write(semantics_topic, semantics_msg, t)

            num_written += 1
            if total and num_written >= total:
                break

            if write_every_n != 0 and num_written % write_every_n == 0:
                rgb_img = img[:, :, ::-1]
                cv2.imwrite(f"rgb_{num_written:06d}.png", img)
                cv2.imwrite(f"semantics_{num_written:06d}.png", semantics[:, :, ::-1])
                a = overlay_alpha
                a_inv = 1 - overlay_alpha
                overlay = (a_inv * rgb_img + a * semantics).astype(np.uint8)
                cv2.imwrite(
                    f"semantics_overlay_{num_written:06d}.png", overlay[:, :, ::-1]
                )

    bag_out.close()


if __name__ == "__main__":
    main()
