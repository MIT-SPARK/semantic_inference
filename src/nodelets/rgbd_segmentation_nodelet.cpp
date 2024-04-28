#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/output_publisher.h"
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

using message_filters::Synchronizer;

class RgbdSegmentationNodelet : public nodelet::Nodelet {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                      sensor_msgs::Image>;

  struct Config {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    double depth_scale = 1.0;
  };

  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                const sensor_msgs::ImageConstPtr& depth_msg);

  Config config_;

  std::unique_ptr<Segmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  std::unique_ptr<OutputPublisher> output_pub_;
};

void declare_config(RgbdSegmentationNodelet::Config& config) {
  using namespace config;
  name("RgbdSegmentationNodelet::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.depth_scale, "depth_scale");
  check(config.depth_scale, GT, 0.0, "depth_scale");
}

void RgbdSegmentationNodelet::onInit() {
  auto nh = getPrivateNodeHandle();
  config_ = config::fromRos<Config>(nh);
  ROS_INFO_STREAM("config: " << config::toString(config_));
  config::checkValid(config_);

  segmenter_ = std::make_unique<Segmenter>(config_.segmenter);
  transport_ = std::make_unique<image_transport::ImageTransport>(nh);
  output_pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);

  image_sub_.subscribe(*transport_, "rgb/image_raw", 1);
  depth_sub_.subscribe(*transport_, "depth/image_rect", 1);
  sync.reset(new Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, depth_sub_));
  sync->registerCallback(boost::bind(&RgbdSegmentationNodelet::callback, this, _1, _2));
}

void RgbdSegmentationNodelet::callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                       const sensor_msgs::ImageConstPtr& depth_msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(rgb_msg, "rgb8");
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  cv_bridge::CvImageConstPtr depth_img_ptr;
  try {
    depth_img_ptr = cv_bridge::toCvShare(depth_msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  cv::Mat depth_img_float;
  if (depth_img_ptr->image.type() == CV_32FC1) {
    depth_img_float = depth_img_ptr->image;
  } else {
    depth_img_ptr->image.convertTo(depth_img_float, CV_32FC1, config_.depth_scale);
  }

  const auto result = segmenter_->infer(img_ptr->image, depth_img_float);
  if (!result) {
    ROS_ERROR("failed to run inference!");
    return;
  }

  output_pub_->publish(img_ptr->header, img_ptr->image, result.labels);
}

}  // namespace semantic_recolor

PLUGINLIB_EXPORT_CLASS(semantic_recolor::RgbdSegmentationNodelet, nodelet::Nodelet)
