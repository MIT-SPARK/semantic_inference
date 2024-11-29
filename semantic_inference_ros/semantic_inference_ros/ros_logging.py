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
"""Module containing ROS logging shim."""

from semantic_inference import Logger
import logging
import rospy


# adapted from https://gist.github.com/ablakey/4f57dca4ea75ed29c49ff00edf622b38
class RosForwarder(logging.Handler):
    """Class to forward logging to ros handler."""

    level_map = {
        logging.DEBUG: rospy.logdebug,
        logging.INFO: rospy.loginfo,
        logging.WARNING: rospy.logwarn,
        logging.ERROR: rospy.logerr,
        logging.CRITICAL: rospy.logfatal,
    }

    def emit(self, record):
        """Send message to ROS."""
        level = record.levelno if record.levelno in self.level_map else logging.CRITICAL
        self.level_map[level](f"{record.name}: {record.msg}")


def setup_ros_log_forwarding(level=logging.INFO):
    """Forward logging to ROS."""
    Logger.addHandler(RosForwarder())
    Logger.setLevel(logging.INFO)
