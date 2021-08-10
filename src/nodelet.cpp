#include "semantic_recolor/nodelet.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(semantic_recolor::Nodelet, nodelet::Nodelet)

namespace semantic_recolor {
void Nodelet::onInit() {
  ros::NodeHandle& nh = getPrivateNodeHandle();
  config_ = readSegmenterConfig(nh);
  segmenter_.reset(new TrtSegmenter(config_));
  if (!segmenter_->init()) {
    ROS_FATAL("unable to init semantic segmentation model memory");
    throw std::runtime_error("bad segmenter init");
  }

  transport_.reset(new image_transport::ImageTransport(nh));
  image_sub_ = transport_->subscribe("rgb/image_raw", 1, &Nodelet::callback, this);
  semantic_image_pub_ = transport_->advertise("semantic/image_raw", 1);
}

void Nodelet::callback(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  if (!segmenter_->infer(img_ptr->image)) {
    ROS_ERROR("failed to run inference!");
    return;
  }

  if (!semantic_image_) {
    semantic_image_.reset(new cv_bridge::CvImage());
    semantic_image_->encoding = "rgb8";
    semantic_image_->image = cv::Mat(img_ptr->image.rows, img_ptr->image.cols, CV_8UC3);
  }

  semantic_image_->header = img_ptr->header;

  fillSemanticImage(color_config_, segmenter_->getClasses(), semantic_image_->image);
  semantic_image_pub_.publish(semantic_image_->toImageMsg());
}

}  // namespace semantic_recolor
