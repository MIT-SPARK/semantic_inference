#include "semantic_recolor/segmentation_nodelet.h"

#include <pluginlib/class_list_macros.h>

#include <opencv2/core.hpp>
#include <optional>

#include "semantic_recolor/ros_utilities.h"

PLUGINLIB_EXPORT_CLASS(semantic_recolor::SegmentationNodelet, nodelet::Nodelet)

namespace semantic_recolor {

void SegmentationNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();

  max_queue_size_ = 30;
  pnh.getParam("max_queue_size", max_queue_size_);

  image_separation_s_ = 0.0;
  pnh.getParam("image_separation_s", image_separation_s_);

  std::string rotation_type = "none";
  pnh.getParam("image_rotation", rotation_type);
  if (rotation_type == "ROTATE_90_CLOCKWISE") {
    do_rotation_ = true;
    pre_rotation_ = cv::ROTATE_90_CLOCKWISE;
    post_rotation_ = cv::ROTATE_90_COUNTERCLOCKWISE;
  } else if (rotation_type == "ROTATE_90_COUNTERCLOCKWISE") {
    do_rotation_ = true;
    pre_rotation_ = cv::ROTATE_90_COUNTERCLOCKWISE;
    post_rotation_ = cv::ROTATE_90_CLOCKWISE;
  } else if (rotation_type == "ROTATE_180") {
    do_rotation_ = true;
    pre_rotation_ = cv::ROTATE_180;
    post_rotation_ = cv::ROTATE_180;
  } else {
    do_rotation_ = false;
  }

  config_ = readModelConfig(pnh);
  segmenter_.reset(new TrtSegmenter(config_));
  if (!segmenter_->init()) {
    ROS_FATAL("unable to init semantic segmentation model memory");
    throw std::runtime_error("bad segmenter init");
  }

  ros::NodeHandle nh = getPrivateNodeHandle();
  transport_.reset(new image_transport::ImageTransport(nh));
  OutputConfig config;
  config.publish_labels = true;  // always publish labels
  pnh.getParam("publish_color", config.publish_color);
  pnh.getParam("publish_overlay", config.publish_overlay);
  output_pub_.reset(new NodeletOutputPublisher(pnh, *transport_, config));

  image_sub_ =
      transport_->subscribe("rgb/image_raw", 1, &SegmentationNodelet::callback, this);

  should_shutdown_ = false;
  spin_thread_.reset(new std::thread(&SegmentationNodelet::spin, this));
}

SegmentationNodelet::~SegmentationNodelet() {
  should_shutdown_ = true;
  if (spin_thread_) {
    spin_thread_->join();
    spin_thread_.reset();
  }
}

void SegmentationNodelet::runSegmentation(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(msg, "rgb8");
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  ROS_DEBUG_STREAM("Encoding: " << img_ptr->encoding << " size: " << img_ptr->image.cols
                                << " x " << img_ptr->image.rows << " x "
                                << img_ptr->image.channels() << " is right type? "
                                << (img_ptr->image.type() == CV_8UC3 ? "yes" : "no"));

  cv::Mat rotated;
  if (do_rotation_) {
    cv::rotate(img_ptr->image, rotated, pre_rotation_);
  } else {
    rotated = img_ptr->image;
  }
  if (!segmenter_->infer(rotated)) {
    ROS_ERROR("failed to run inference!");
    return;
  }

  cv::Mat re_rotated;
  if (do_rotation_) {
    cv::rotate(segmenter_->getClasses(), re_rotated, post_rotation_);
  } else {
    re_rotated = segmenter_->getClasses();
  }
  output_pub_->publish(img_ptr->header, img_ptr->image, re_rotated);
}

bool SegmentationNodelet::haveWork() const {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  return !image_queue_.empty();
}

void SegmentationNodelet::spin() {
  const ros::Duration min_diff(image_separation_s_);

  ros::WallRate r(100);
  std::optional<ros::Time> last_time;
  while (ros::ok() && !should_shutdown_) {
    ros::spinOnce();
    if (!haveWork()) {
      r.sleep();
      continue;
    }

    sensor_msgs::ImageConstPtr msg;
    {  // start mutex scope
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (last_time) {
        const auto curr_diff = image_queue_.front()->header.stamp - *last_time;
        ROS_DEBUG_STREAM("current time diff: " << curr_diff.toSec() << "[s] (min: "
                                               << min_diff.toSec() << "[s])");
        if (curr_diff < min_diff) {
          image_queue_.pop_front();
          continue;
        }
      }

      last_time = image_queue_.front()->header.stamp;
      msg = image_queue_.front();
      image_queue_.pop_front();

    }  // end mutex scope

    runSegmentation(msg);
  }
}

void SegmentationNodelet::callback(const sensor_msgs::ImageConstPtr& msg) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  if (max_queue_size_ > 0 &&
      image_queue_.size() >= static_cast<size_t>(max_queue_size_)) {
    image_queue_.pop_front();
  }

  image_queue_.push_back(msg);
}

}  // namespace semantic_recolor
