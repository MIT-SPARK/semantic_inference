#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nodelet/nodelet.h>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/nodelet_output_publisher.h"
#include "semantic_recolor/rgbd_segmenter.h"

namespace semantic_recolor {

class RgbdSegmentationNodelet : public nodelet::Nodelet {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                      sensor_msgs::Image>;

  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                const sensor_msgs::ImageConstPtr& depth_msg);

  double depth_scale_;

  ModelConfig config_;
  std::unique_ptr<TrtRgbdSegmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  std::unique_ptr<NodeletOutputPublisher> output_pub_;
};

}  // namespace semantic_recolor
