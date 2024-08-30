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
