import logging
import pathlib
from contextlib import contextmanager

import click
import distinctipy
import imageio.v3 as iio
import numpy as np
import spark_config as sc
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)

ENCODINGS = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 4),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}


def _parse_image(msg):
    if msg is None:
        return None

    if "CompressedImage" in msg.__msgtype__:
        return iio.imread(msg.data.tobytes())

    info = ENCODINGS.get(msg.encoding)
    if info is None:
        raise ValueError(f"Unhandled image message encoding: '{msg.encoding}'")

    img = np.frombuffer(msg.data, dtype=info[0]).reshape(
        (msg.height, msg.width, info[1])
    )
    return np.squeeze(img).copy()


def _message_iter(bag, typestore, topics):
    if len(topics) == 0:
        return None

    connections = [x for x in bag.connections if x.topic in topics]
    N_unique = len(set([x.topic for x in connections]))
    if N_unique != len(topics):
        all_topics = set([x.topic for x in bag.connections])
        missing = set(topics).difference(all_topics)
        logging.warning(f"Could not find {missing} in bag (available: {all_topics})")
        return None

    for connection, _, rawdata in bag.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        img = _parse_image(msg)
        if img is None:
            logging.warning(f"Could not parse image for {connection.topic}!")
            continue

        yield connection.topic, img


def _normalize_path(path):
    return pathlib.Path(path).expanduser().absolute()


@contextmanager
def bag_image_store(bag_path, topics):
    bag_path = _normalize_path(bag_path)
    bag = AnyReader([bag_path])
    bag.open()

    msg_types = {}
    for connection in bag.connections:
        msg_types.update(get_types_from_msg(connection.msgdef, connection.msgtype))

    typestore = get_typestore(Stores.EMPTY)
    typestore.register(msg_types)

    yield _message_iter(bag, typestore, topics)

    bag.close()


@click.group(name="run")
def cli():
    """Subcommands for running models."""
    pass


def _load_configs(configs, config_files):
    result = {}
    for config in configs:
        try:
            result.update(yaml.load(config))
        except Exception as e:
            print(f"Invalid YAML: {e}")
            continue

    for config in config_files:
        with open(config) as fin:
            try:
                result.update(yaml.load(fin))
            except Exception as e:
                print(f"Invalid YAML: {e}")
                continue

    return result


def _load_model(category, model_config):
    model_type = model_config.get("type")
    if model_type is None:
        raise ValueError("Missing model type from model config!")

    config = sc.ConfigFactory.create(category, model_type)
    config.update(model_config)
    return sc.ConfigFactory.get_constructor(category, model_type)(config)


@cli.command()
@click.argument("bag_path", type=click.Path(exists=True))
@click.option("--topic", "-t", multiple=True, type=str)
@click.option("--config", "-c", multiple=True, type=str)
@click.option("--config-file", "-f", multiple=True, type=click.Path(exists=True))
def bag(bag_path, topic, config, config_file):
    sc.discover_plugins("semantic_inference")
    model_config = _load_configs(config, config_file)
    model = _load_model("closed_set_model", model_config)

    colors = distinctipy.get_colors(
        150, exclude_colors=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    )
    colors = (255 * np.array(colors)).astype(np.uint8).T

    bag_path = _normalize_path(bag_path)
    with bag_image_store(bag_path, topic) as bag:
        for topic, img in bag:
            labels = model.segment(img)
            print(f"img @ {topic} -> shape: {img.shape}")
            labels = model.segment(img)
            output = np.take(colors, labels, mode="raise", axis=1)
            output = np.transpose(output, [1, 2, 0])
            iio.imwrite("/tmp/labels.png", output)
            break
