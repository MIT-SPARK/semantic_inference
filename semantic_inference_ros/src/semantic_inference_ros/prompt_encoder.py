"""Utility for prompt embedding service."""

import rospy
from semantic_inference_msgs.srv import EncodeFeature, EncodeFeatureResponse

from semantic_inference_ros.ros_conversions import Conversions


class PromptEncoder:
    """Node implementation."""

    def __init__(self, model, name="~embed"):
        """Construct a feature encoder node."""
        self._model = model
        self._srv = rospy.Service(name, EncodeFeature, self._callback)

    def _callback(self, req):
        res = EncodeFeatureResponse()
        embedding = self._model.embed_text(req.prompt).cpu().numpy().squeeze()
        res.feature.header.stamp = rospy.Time.now()
        res.feature.feature = Conversions.to_feature(embedding)
        return res
