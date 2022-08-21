#pragma once
#include "semantic_recolor/nodelet_output_publisher.h"
#include "semantic_recolor/segmenter.h"
#include "semantic_recolor/utilities.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>

namespace semantic_recolor {

class SegmentationNodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  ModelConfig config_;
  std::unique_ptr<TrtSegmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber image_sub_;

  std::unique_ptr<NodeletOutputPublisher> output_pub_;
};

}  // namespace semantic_recolor
