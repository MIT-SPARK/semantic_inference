#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "semantic_recolor/image_recolor.h"

namespace semantic_recolor {

class OutputPublisher {
 public:
  struct Config {
    ImageRecolor::Config recolor;
    bool publish_labels = true;
    bool publish_color = true;
    bool publish_overlay = true;
    double overlay_alpha = 0.4;
  } const config;

  OutputPublisher(const Config& config, image_transport::ImageTransport& transport);

  void publish(const std_msgs::Header& header,
               const cv::Mat& labels,
               const cv::Mat& color = cv::Mat());

 private:
  ImageRecolor image_recolor_;

  image_transport::Publisher label_pub_;
  image_transport::Publisher color_pub_;
  image_transport::Publisher overlay_pub_;
  cv_bridge::CvImagePtr label_image_;
  cv_bridge::CvImagePtr color_image_;
  cv_bridge::CvImagePtr overlay_image_;
};

void declare_config(OutputPublisher::Config& config);

}  // namespace semantic_recolor
