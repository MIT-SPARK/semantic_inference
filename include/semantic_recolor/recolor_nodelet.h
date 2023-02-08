#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>

#include "semantic_recolor/nodelet_output_publisher.h"

namespace semantic_recolor {

class RecolorNodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

  virtual ~RecolorNodelet();

 private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber image_sub_;
  std::unique_ptr<NodeletOutputPublisher> output_pub_;
};

}  // namespace semantic_recolor
