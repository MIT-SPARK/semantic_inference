#pragma once
#include "semantic_recolor/utilities.h"
#include "semantic_recolor/segmenter.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>

namespace semantic_recolor {

class SegmentationNodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  ModelConfig config_;
  SemanticColorConfig color_config_;
  std::unique_ptr<TrtSegmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher semantic_image_pub_;
  image_transport::Publisher overlay_image_pub_;
  cv_bridge::CvImagePtr semantic_image_;
  cv_bridge::CvImagePtr overlay_image_;
  bool create_overlay_;
};

}  // namespace semantic_recolor
