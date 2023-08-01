#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "semantic_recolor/semantic_color_config.h"

namespace semantic_recolor {

struct OutputConfig {
  bool publish_labels = true;
  bool publish_color = true;
  bool publish_overlay = true;
};

class NodeletOutputPublisher {
 public:
  NodeletOutputPublisher(const ros::NodeHandle& nh,
                         image_transport::ImageTransport& transport,
                         const OutputConfig& config);

  void publish(const std_msgs::Header& header,
               const cv::Mat& original,
               const cv::Mat& classes);

 private:
  OutputConfig config_;
  SemanticColorConfig color_config_;

  image_transport::Publisher semantic_image_pub_;
  image_transport::Publisher color_image_pub_;
  image_transport::Publisher overlay_image_pub_;
  cv_bridge::CvImagePtr semantic_image_;
  cv_bridge::CvImagePtr color_image_;
  cv_bridge::CvImagePtr overlay_image_;
};

}  // namespace semantic_recolor
