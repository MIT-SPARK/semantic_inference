#include "semantic_recolor/nodelet_output_publisher.h"

namespace semantic_recolor {

using image_transport::ImageTransport;

NodeletOutputPublisher::NodeletOutputPublisher(const ros::NodeHandle& nh,
                                               ImageTransport& transport)
    : create_overlay_(true) {
  nh.getParam("create_semantic_overlay", create_overlay_);
  color_config_ = SemanticColorConfig(ros::NodeHandle(nh, "colors"));

  semantic_image_pub_ = transport.advertise("semantic/image_raw", 1);

  if (create_overlay_) {
    overlay_image_pub_ = transport.advertise("semantic/overlay/image_raw", 1);
  }
}

void NodeletOutputPublisher::publish(const std_msgs::Header& header,
                                     const cv::Mat& original,
                                     const cv::Mat& classes) {
  if (!semantic_image_) {
    semantic_image_.reset(new cv_bridge::CvImage());
    semantic_image_->encoding = "rgb8";
    semantic_image_->image = cv::Mat(original.rows, original.cols, CV_8UC3);
  }

  semantic_image_->header = header;
  fillSemanticImage(color_config_, classes, semantic_image_->image);
  semantic_image_pub_.publish(semantic_image_->toImageMsg());

  if (!create_overlay_) {
    return;
  }

  if (!overlay_image_) {
    overlay_image_.reset(new cv_bridge::CvImage());
    overlay_image_->encoding = "rgb8";
    overlay_image_->image = cv::Mat(original.rows, original.cols, CV_8UC3);
  }

  overlay_image_->header = header;

  double alpha = 0.4;
  cv::addWeighted(semantic_image_->image,
                  alpha,
                  original,
                  (1.0 - alpha),
                  0.0,
                  overlay_image_->image);

  overlay_image_pub_.publish(overlay_image_->toImageMsg());
}

}  // namespace semantic_recolor
