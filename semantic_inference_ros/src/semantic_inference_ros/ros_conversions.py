"""Module containing ROS message conversions."""

import cv_bridge

import semantic_inference_msgs.msg


class Conversions:
    """Conversion namespace."""

    bridge = cv_bridge.CvBridge()

    @classmethod
    def to_image(cls, msg, encoding="passthrough"):
        """Convert sensor_msgs.Image to numpy array."""
        return cls.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

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
