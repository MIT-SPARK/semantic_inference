#include "semantic_recolor/rgbd_segmentation_nodelet.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(semantic_recolor::RgbdSegmentationNodelet, nodelet::Nodelet)

namespace semantic_recolor {

void RgbdSegmentationNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();
  config_ = readModelConfig(pnh);

  std::string depth_input_name;
  if (!pnh.getParam("depth_input_name", depth_input_name)) {
    ROS_FATAL("depth input name required!");
    throw std::runtime_error("missing depth input name");
  }

  depth_scale_ = 1.0;
  if (!pnh.getParam("depth_scale", depth_scale_)) {
    ROS_WARN("missing depth scale, default to 1.0");
  }

  segmenter_.reset(new TrtRgbdSegmenter(config_, depth_input_name));
  if (!segmenter_->init()) {
    ROS_FATAL("unable to init semantic segmentation model memory");
    throw std::runtime_error("bad segmenter init");
  }

  ros::NodeHandle nh = getNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));
  output_pub_.reset(new NodeletOutputPublisher(pnh, *transport_));

  image_sub_.subscribe(*transport_, "rgb/image_raw", 1);
  depth_sub_.subscribe(*transport_, "depth/image_rect", 1);
  sync.reset(new message_filters::Synchronizer<SyncPolicy>(
      SyncPolicy(10), image_sub_, depth_sub_));
  sync->registerCallback(boost::bind(&RgbdSegmentationNodelet::callback, this, _1, _2));
}

void RgbdSegmentationNodelet::callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                       const sensor_msgs::ImageConstPtr& depth_msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(rgb_msg);
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
    depth_img_ptr->image.convertTo(depth_img_float, CV_32FC1, depth_scale_);
  }

  if (!segmenter_->infer(img_ptr->image, depth_img_float)) {
    ROS_ERROR("failed to run inference!");
    return;
  }

  output_pub_->publish(img_ptr->header, img_ptr->image, segmenter_->getClasses());
}

}  // namespace semantic_recolor
