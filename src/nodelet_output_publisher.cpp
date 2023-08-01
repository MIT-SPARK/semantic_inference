#include "semantic_recolor/nodelet_output_publisher.h"

namespace semantic_recolor {

using image_transport::ImageTransport;

NodeletOutputPublisher::NodeletOutputPublisher(const ros::NodeHandle& nh,
                                               ImageTransport& transport,
                                               const OutputConfig& config)
    : config_(config) {
  color_config_ = SemanticColorConfig(ros::NodeHandle(nh, "colors"));

  if (config_.publish_labels) {
    semantic_image_pub_ = transport.advertise("semantic/image_raw", 1);
  }

  if (config_.publish_color) {
    color_image_pub_ = transport.advertise("semantic_color/image_raw", 1);
  }

  if (config_.publish_overlay) {
    overlay_image_pub_ = transport.advertise("semantic_overlay/image_raw", 1);
  }
}

void NodeletOutputPublisher::publish(const std_msgs::Header& header,
                                     const cv::Mat& original,
                                     const cv::Mat& classes) {
  if (!semantic_image_) {
    semantic_image_.reset(new cv_bridge::CvImage());
    // we can't support 32 signed labels, so we do 16-bit signed to distinguish from
    // depth
    semantic_image_->encoding = "16SC1";
    semantic_image_->image = cv::Mat(original.rows, original.cols, CV_16SC1);
  }

  semantic_image_->header = header;
  color_config_.relabelImage(classes, semantic_image_->image);
  // TODO(nathan) fill classes
  if (config_.publish_labels) {
    semantic_image_pub_.publish(semantic_image_->toImageMsg());
  }

  if (!config_.publish_color && !config_.publish_overlay) {
    return;
  }

  if (!color_image_) {
    color_image_.reset(new cv_bridge::CvImage());
    color_image_->encoding = "rgb8";
    color_image_->image = cv::Mat(original.rows, original.cols, CV_8UC3);
  }

  color_image_->header = header;
  color_config_.recolorImage(classes, color_image_->image);
  if (config_.publish_color) {
    color_image_pub_.publish(color_image_->toImageMsg());
  }

  if (!config_.publish_overlay) {
    return;
  }

  if (!overlay_image_) {
    overlay_image_.reset(new cv_bridge::CvImage());
    overlay_image_->encoding = "rgb8";
    overlay_image_->image = cv::Mat(original.rows, original.cols, CV_8UC3);
  }

  overlay_image_->header = header;

  double alpha = 0.4;
  cv::addWeighted(
      color_image_->image, alpha, original, (1.0 - alpha), 0.0, overlay_image_->image);

  overlay_image_pub_.publish(overlay_image_->toImageMsg());
}

}  // namespace semantic_recolor
