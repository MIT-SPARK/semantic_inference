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
"""Module containing queue-based image processor."""

from semantic_inference import Config
from semantic_inference_ros.ros_conversions import Conversions
from dataclasses import dataclass

import sensor_msgs.msg

import rospy
import queue
import threading
import time


@dataclass
class ImageWorkerConfig(Config):
    """Configuration for image worker."""

    encoding: str = "rgb8"
    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class ImageWorker:
    """Class to simplify message processing."""

    def __init__(self, config, topic, callback, **kwargs):
        """Register worker with ros."""
        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = False
        self._last_stamp = None

        self._queue = queue.Queue(maxsize=config.queue_size)

        rospy.on_shutdown(self.stop)

        self._sub = rospy.Subscriber(
            topic, sensor_msgs.msg.Image, self.add_message, queue_size=1, **kwargs
        )
        self.start()

    def add_message(self, msg):
        """Add new message to queue."""
        if not self._queue.full():
            self._queue.put(msg, block=False, timeout=False)

    def start(self):
        """Start worker processing queue."""
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._do_work)
            self._thread.start()

    def stop(self):
        """Stop worker from processing queue."""
        if self._started:
            self._should_shutdown = True
            self._thread.join()

        self._started = False
        self._should_shutdown = False

    def spin(self):
        """Wait for ros to shutdown or worker to exit."""
        if not self._started:
            return

        while self._thread.is_alive() and not self._should_shutdown:
            time.sleep(1.0e-2)

        self.stop()

    def _do_work(self):
        while not self._should_shutdown:
            try:
                msg = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._last_stamp is not None:
                diff_s = (msg.header.stamp - self._last_stamp).to_sec()
                if diff_s < self._config.min_separation_s:
                    continue

            self._last_stamp = msg.header.stamp

            # try:
            img = Conversions.to_image(msg, encoding=self._config.encoding)
            self._callback(msg.header, img)
            # except Exception as e:
            # rospy.logerr(f"spin failed: {e}")
