#pragma once
#include "semantic_recolor/utilities.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>

namespace semantic_recolor {

class Nodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  SegmentationConfig config_;
  SemanticColorConfig color_config_;
  std::unique_ptr<TrtSegmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher semantic_image_pub_;
  cv_bridge::CvImagePtr semantic_image_;
};

}  // namespace semantic_recolor
