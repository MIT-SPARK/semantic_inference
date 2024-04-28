#include "semantic_recolor/output_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/validation.h>

#include "semantic_recolor/image_utilities.h"
#include "semantic_recolor/logging.h"

namespace semantic_recolor {

using image_transport::ImageTransport;

OutputPublisher::OutputPublisher(const Config& config, ImageTransport& transport)
    : config(config::checkValid(config)), image_recolor_(config.recolor) {
  if (config.publish_labels) {
    label_pub_ = transport.advertise("semantic/image_raw", 1);
  }

  if (config.publish_color) {
    color_pub_ = transport.advertise("semantic_color/image_raw", 1);
  }

  if (config.publish_overlay) {
    overlay_pub_ = transport.advertise("semantic_overlay/image_raw", 1);
  }
}

void OutputPublisher::publish(const std_msgs::Header& header,
                              const cv::Mat& labels,
                              const cv::Mat& color) {
  if (!label_image_) {
    label_image_.reset(new cv_bridge::CvImage());
    // we can't support 32 signed labels, so we do 16-bit signed to distinguish from
    // depth
    label_image_->encoding = "16SC1";
    label_image_->image = cv::Mat(color.rows, color.cols, CV_16SC1);
  }

  label_image_->header = header;
  image_recolor_.relabelImage(labels, label_image_->image);
  if (config.publish_labels) {
    label_pub_.publish(label_image_->toImageMsg());
  }

  if (!config.publish_color && !config.publish_overlay) {
    return;
  }

  if (!color_image_) {
    color_image_.reset(new cv_bridge::CvImage());
    color_image_->encoding = "rgb8";
    color_image_->image = cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  color_image_->header = header;
  image_recolor_.recolorImage(label_image_->image, color_image_->image);
  if (config.publish_color) {
    color_pub_.publish(color_image_->toImageMsg());
  }

  if (!config.publish_overlay || color.empty()) {
    return;
  }

  if (!overlay_image_) {
    overlay_image_.reset(new cv_bridge::CvImage());
    overlay_image_->encoding = "rgb8";
    overlay_image_->image = cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  overlay_image_->header = header;

  cv::addWeighted(color_image_->image,
                  config.overlay_alpha,
                  color,
                  (1.0 - config.overlay_alpha),
                  0.0,
                  overlay_image_->image);

  overlay_pub_.publish(overlay_image_->toImageMsg());
}

void declare_config(OutputPublisher::Config& config) {
  using namespace config;
  name("OutputPublisher::Config");
  field(config.recolor, "recolor");
  field(config.publish_labels, "publish_labels");
  field(config.publish_color, "publish_color");
  field(config.publish_overlay, "publish_overlay");
  field(config.overlay_alpha, "overlay_alpha");
}

}  // namespace semantic_recolor
