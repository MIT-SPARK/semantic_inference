#include <config_utilities/parsing/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <optional>

#include "semantic_inference_ros/output_publisher.h"

namespace semantic_inference {

class RecolorNodelet : public nodelet::Nodelet {
 public:
  virtual void onInit() override;

  virtual ~RecolorNodelet();

 private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber sub_;
  std::unique_ptr<OutputPublisher> pub_;
};

void RecolorNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));

  const auto output_config = config::fromRos<OutputPublisher::Config>(nh);
  pub_ = std::make_unique<OutputPublisher>(output_config, *transport_);

  sub_ = transport_->subscribe("labels/image_raw", 1, &RecolorNodelet::callback, this);
}

RecolorNodelet::~RecolorNodelet() {}

void RecolorNodelet::callback(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  pub_->publish(img_ptr->header, img_ptr->image);
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::RecolorNodelet, nodelet::Nodelet)
