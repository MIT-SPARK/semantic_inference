#include "semantic_recolor/segmentation_nodelet.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(semantic_recolor::SegmentationNodelet, nodelet::Nodelet)

namespace semantic_recolor {
void SegmentationNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();
  config_ = readSegmenterConfig(pnh);
  color_config_ = SemanticColorConfig(ros::NodeHandle(pnh, "colors"));
  segmenter_.reset(new TrtSegmenter(config_));
  if (!segmenter_->init()) {
    ROS_FATAL("unable to init semantic segmentation model memory");
    throw std::runtime_error("bad segmenter init");
  }

  ros::NodeHandle nh = getNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));
  image_sub_ =
      transport_->subscribe("rgb/image_raw", 1, &SegmentationNodelet::callback, this);
  semantic_image_pub_ = transport_->advertise("semantic/image_raw", 1);

  // todo(jared): add parameter for this
  create_overlay_ = true;
  if (create_overlay_) {
    overlay_image_pub_ = transport_->advertise("semantic/overlay/image_raw", 1);
  }
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

  if (!semantic_image_) {
    semantic_image_.reset(new cv_bridge::CvImage());
    semantic_image_->encoding = "rgb8";
    // semantic_image_->image = cv::Mat(config_.height, config_.width, CV_8UC3);
    semantic_image_->image = cv::Mat(img_ptr->image.rows, img_ptr->image.cols, CV_8UC3);
  }

  semantic_image_->header = img_ptr->header;

  fillSemanticImage(color_config_, segmenter_->getClasses(), semantic_image_->image);
  semantic_image_pub_.publish(semantic_image_->toImageMsg());

  if (create_overlay_) {
    if (!overlay_image_) {
      overlay_image_.reset(new cv_bridge::CvImage());
      overlay_image_->encoding = "rgb8";
      overlay_image_->image =
          cv::Mat(img_ptr->image.rows, img_ptr->image.cols, CV_8UC3);
    }
    overlay_image_->header = img_ptr->header;
    createOverlayImage(img_ptr->image,
                       semantic_image_->image,
                       overlay_image_->image);
    overlay_image_pub_.publish(overlay_image_->toImageMsg());
  }
}

}  // namespace semantic_recolor
