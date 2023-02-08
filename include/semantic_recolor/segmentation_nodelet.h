#pragma once
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>

#include <atomic>
#include <mutex>
#include <thread>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/nodelet_output_publisher.h"
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

class SegmentationNodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

  virtual ~SegmentationNodelet();

 private:
  void spin();

  void runSegmentation(const sensor_msgs::ImageConstPtr& msg);

  void callback(const sensor_msgs::ImageConstPtr& msg);

  bool haveWork() const;

  ModelConfig config_;
  std::unique_ptr<TrtSegmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber image_sub_;

  int max_queue_size_;
  double image_separation_s_;
  std::list<sensor_msgs::ImageConstPtr> image_queue_;
  mutable std::mutex queue_mutex_;
  std::atomic<bool> should_shutdown_;
  std::unique_ptr<std::thread> spin_thread_;

  std::unique_ptr<NodeletOutputPublisher> output_pub_;
};

}  // namespace semantic_recolor
