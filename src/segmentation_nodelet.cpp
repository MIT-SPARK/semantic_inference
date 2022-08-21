#include "semantic_recolor/segmentation_nodelet.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(semantic_recolor::SegmentationNodelet, nodelet::Nodelet)

namespace semantic_recolor {
void SegmentationNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();

  config_ = readModelConfig(pnh);
  segmenter_.reset(new TrtSegmenter(config_));
  if (!segmenter_->init()) {
    ROS_FATAL("unable to init semantic segmentation model memory");
    throw std::runtime_error("bad segmenter init");
  }

  ros::NodeHandle nh = getNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));
  output_pub_.reset(new NodeletOutputPublisher(pnh, *transport_));

  image_sub_ =
      transport_->subscribe("rgb/image_raw", 1, &SegmentationNodelet::callback, this);
}

void SegmentationNodelet::callback(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  ROS_DEBUG_STREAM("Encoding: " << img_ptr->encoding << " size: " << img_ptr->image.cols
                                << " x " << img_ptr->image.rows << " x "
                                << img_ptr->image.channels() << " is right type? "
                                << (img_ptr->image.type() == CV_8UC3 ? "yes" : "no"));
  if (!segmenter_->infer(img_ptr->image)) {
    ROS_ERROR("failed to run inference!");
    return;
  }

  output_pub_->publish(img_ptr->header, img_ptr->image, segmenter_->getClasses());
}

}  // namespace semantic_recolor
