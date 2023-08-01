#include "semantic_recolor/recolor_nodelet.h"

#include <pluginlib/class_list_macros.h>

#include <optional>

#include "semantic_recolor/ros_utilities.h"

PLUGINLIB_EXPORT_CLASS(semantic_recolor::RecolorNodelet, nodelet::Nodelet)

namespace semantic_recolor {

void RecolorNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();

  ros::NodeHandle nh = getNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));

  OutputConfig config;
  config.publish_labels = true;
  pnh.getParam("publish_color", config.publish_color);
  config.publish_overlay = false; // no rgb to publish overlay
  output_pub_.reset(new NodeletOutputPublisher(pnh, *transport_, config));

  image_sub_ =
      transport_->subscribe("labels/image_raw", 1, &RecolorNodelet::callback, this);
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

  output_pub_->publish(img_ptr->header, img_ptr->image, img_ptr->image);
}

}  // namespace semantic_recolor
