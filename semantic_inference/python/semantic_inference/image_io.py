import functools
import logging
import pathlib
from contextlib import contextmanager, nullcontext

import imageio.v3 as iio
import numpy as np
import rosbags.rosbag1
import rosbags.rosbag2
import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

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


def _message_iter(bag, typestore, topics, with_progress, is_ros1):
    if len(topics) == 0:
        return None

    connections = [x for x in bag.connections if x.topic in topics]
    N_unique = len(set([x.topic for x in connections]))
    if N_unique != len(topics):
        all_topics = set([x.topic for x in bag.connections])
        missing = set(topics).difference(all_topics)
        logging.warning(f"Could not find {missing} in bag (available: {all_topics})")
        return None

    msg_iter = bag.messages(connections=connections)
    if with_progress:
        N = sum([x.msgcount for x in connections])
        msg_iter = tqdm.tqdm(msg_iter, total=N)

    for connection, timestamp, rawdata in msg_iter:
        if is_ros1:
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        else:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        yield connection.topic, msg, timestamp


def _normalize_path(path):
    return pathlib.Path(path).expanduser().absolute()


def parse_image_msg(msg):
    if msg is None:
        return None

    if "CompressedImage" in msg.__msgtype__:
        return iio.imread(msg.data.tobytes())[:, :, ::-1]

    info = ENCODINGS.get(msg.encoding)
    if info is None:
        raise ValueError(f"Unhandled image message encoding: '{msg.encoding}'")

    img = np.frombuffer(msg.data, dtype=info[0]).reshape(
        (msg.height, msg.width, info[1])
    )

    if "rgb" in msg.encoding:
        img = img[:, :, ::-1]

    return np.squeeze(img).copy()


def _get_msg_encoding(img):
    step = img.shape[1] * img.dtype.itemsize
    encodings = {
        "uint8": "8UC1",
        "int8": "8SC1",
        "uint16": "16UC1",
        "int16": "16SC1",
        "int32": "32SC1",
        "uint32": "32SC1",
        "float32": "32FC1",
        "float64": "64FC1",
    }
    return encodings.get(img.dtype.name), step


class BagImageWriter:
    """Wrapper around a bag to manage writing output."""

    def __init__(self, bag_path, typestore, overwrite=True, compress=True):
        """Construct the bag writer."""
        self.typestore = typestore
        self._factory = self.typestore.types["sensor_msgs/msg/Image"]
        self._msg_type = self._factory.__msgtype__
        self._conn = {}

        bag_path = _normalize_path(bag_path)
        if overwrite and bag_path.exists():
            if bag_path.is_dir():
                bag_path.rmdir()
            else:
                bag_path.unlink()

        is_ros1 = bag_path.suffix == ".bag"
        if is_ros1:
            self.writer = rosbags.rosbag1.Writer(bag_path)
            if compress:
                self.writer.set_compression(
                    rosbags.rosbag1.Writer.CompressionFormat.BZ2
                )

            self.serializer = functools.partial(
                self.typestore.serialize_ros1, typename=self._msg_type
            )
        else:
            self.writer = rosbags.rosbag2.Writer(bag_path)
            if compress:
                self.writer.set_compression(
                    rosbags.rosbag2.Writer.CompressionMode.MESSAGE,
                    rosbags.rosbags2.Writer.CompressionFormat.ZSTD,
                )

            self.serializer = functools.partial(
                self.typestore.serialize_cdr, typename=self._msg_type
            )

    def __enter__(self):
        """Open the writer."""
        self.writer.open()
        return self

    def __exit__(self, *args):
        """Close the writer."""
        self.writer.close()

    def write(self, header, timestamp, rgb_topic, img, name="semantic"):
        """Write image to output bag."""
        topic_path = pathlib.Path(rgb_topic)
        camera_ns = topic_path.parent.parent
        if topic_path.stem == "compressed":
            camera_ns = camera_ns.parent

        topic = str(camera_ns / name / "image_raw")
        if topic not in self._conn:
            self._conn[topic] = self.writer.add_connection(
                topic, self._msg_type, typestore=self.typestore
            )

        conn = self._conn[topic]
        img = np.squeeze(img)
        if len(img.shape) > 2:
            logging.error(f"Skipping multi-channel image on topic {topic}")
            return

        encoding, step = _get_msg_encoding(img)
        if encoding is None:
            logging.error(f"Invalid dtype '{img.dtype}' for topic {topic}")
            return

        msg = self._factory(
            header=header,
            height=img.shape[0],
            width=img.shape[1],
            encoding=encoding,
            is_bigendian=False,
            step=step,
            data=img.flatten().view(np.uint8),
        )
        self.writer.write(conn, timestamp, self.serializer(msg))


@contextmanager
def bag_image_store(bag_path, topics, output_path=None, with_progress=True):
    bag_path = _normalize_path(bag_path)
    bag = AnyReader([bag_path])
    bag.open()

    msg_types = {}
    for connection in bag.connections:
        msg_types.update(get_types_from_msg(connection.msgdef, connection.msgtype))

    typestore = get_typestore(Stores.EMPTY)
    typestore.register(msg_types)

    if output_path is not None:
        bag_out = BagImageWriter(output_path, typestore)
    else:
        bag_out = nullcontext()

    is_ros1 = bag_path.suffix == ".bag"
    with bag_out as bag_out:
        yield _message_iter(bag, typestore, topics, with_progress, is_ros1), bag_out

    bag.close()
