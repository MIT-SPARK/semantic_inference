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
"""Module containing ROS message conversions."""

import cv_bridge

import semantic_inference_msgs.msg


class Conversions:
    """Conversion namespace."""

    bridge = cv_bridge.CvBridge()

    @classmethod
    def to_image(cls, msg, encoding="passthrough", msg_type=None):
        """Convert sensor_msgs.Image to numpy array."""
        if msg_type is None or msg_type == "sensor_msgs/Image":
            return cls.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
        elif msg_type == "sensor_msgs/CompressedImage":
            return cls.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=encoding)
        else:
            raise ValueError(f"Message type '{msg_type} ({type(msg)})' not supported!")

    @staticmethod
    def to_feature(feature):
        """Create a feature vector message."""
        msg = semantic_inference_msgs.msg.FeatureVector()
        msg.data = feature.tolist()
        return msg

    @staticmethod
    def to_stamped_feature(header, feature):
        """
        Create a stamped feature vector message.

        Args:
            header (std_msgs.msg.Header): Original image header
            feature (np.ndarray): Image feature
        """
        msg = semantic_inference_msgs.msg.FeatureVectorStamped()
        msg.header = header
        msg.feature = Conversions.to_feature(feature)
        return msg

    @classmethod
    def to_feature_image(cls, header, results):
        """
        Create a FeatureImage from segmentation results.

        Args:
            header (std_msgs.msg.Header): Original image header
            results: (semantic_inference.SegmentationResults): Segmentation output
        """
        msg = semantic_inference_msgs.msg.FeatureImage()
        msg.header = header
        msg.image = cls.bridge.cv2_to_imgmsg(results.instances)
        msg.mask_ids = list(range(1, len(results.features) + 1))
        msg.features = [cls.to_feature(x) for x in results.features]
        return msg
