#include "semantic_recolor/segmentation_nodelet.h"
#include "semantic_recolor/ros_utilities.h"

#include <pluginlib/class_list_macros.h>
#include <optional>

PLUGINLIB_EXPORT_CLASS(semantic_recolor::SegmentationNodelet, nodelet::Nodelet)

namespace semantic_recolor {

void SegmentationNodelet::onInit() {
  ros::NodeHandle& pnh = getPrivateNodeHandle();

  max_queue_size_ = 30;
  pnh.getParam("max_queue_size", max_queue_size_);

  image_separation_s_ = 0.0;
  pnh.getParam("image_separation_s", image_separation_s_);

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
        ROS_INFO_STREAM("current time diff: " << curr_diff.toSec() << "[s] (min: "
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
