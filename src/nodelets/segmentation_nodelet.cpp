#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <atomic>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <thread>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/output_publisher.h"
#include "semantic_recolor/ros_log_sink.h"
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

class SegmentationNodelet : public nodelet::Nodelet {
 public:
  struct Config {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    size_t max_queue_size = 30;
    double image_separation_s = 0.0;
    std::string rotation_type = "none";
  };

  virtual void onInit() override;

  virtual ~SegmentationNodelet();

 private:
  void spin();

  void runSegmentation(const sensor_msgs::ImageConstPtr& msg);

  void callback(const sensor_msgs::ImageConstPtr& msg);

  bool haveWork() const;

  Config config_;
  std::unique_ptr<Segmenter> segmenter_;

  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::Subscriber sub_;

  bool do_rotation_;
  cv::RotateFlags pre_rotation_;
  cv::RotateFlags post_rotation_;

  std::list<sensor_msgs::ImageConstPtr> image_queue_;
  mutable std::mutex queue_mutex_;
  std::atomic<bool> should_shutdown_;
  std::unique_ptr<std::thread> spin_thread_;

  std::unique_ptr<OutputPublisher> output_pub_;
};

void declare_config(SegmentationNodelet::Config& config) {
  using namespace config;
  name("SegmentationNodelet::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.max_queue_size, "max_queue_size");
  field(config.image_separation_s, "image_separation_s");
  field(config.rotation_type, "rotation_type");
  checkIsOneOf(
      config.rotation_type,
      {"none", "ROTATE_90_CLOCKWISE", "ROTATE_180", "ROTATE_90_COUNTERCLOCKWISE"},
      "rotation_type");
}

void SegmentationNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>());

  try {
    config_ = config::fromRos<SegmentationNodelet::Config>(nh);
    SLOG(INFO) << "\n" << config::toString(config_);
    config::checkValid(config_);

    segmenter_ = std::make_unique<Segmenter>(config_.segmenter);
    transport_ = std::make_unique<image_transport::ImageTransport>(nh);
    output_pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);

    if (config_.rotation_type == "ROTATE_90_CLOCKWISE") {
      do_rotation_ = true;
      pre_rotation_ = cv::ROTATE_90_CLOCKWISE;
      post_rotation_ = cv::ROTATE_90_COUNTERCLOCKWISE;
    } else if (config_.rotation_type == "ROTATE_90_COUNTERCLOCKWISE") {
      do_rotation_ = true;
      pre_rotation_ = cv::ROTATE_90_COUNTERCLOCKWISE;
      post_rotation_ = cv::ROTATE_90_CLOCKWISE;
    } else if (config_.rotation_type == "ROTATE_180") {
      do_rotation_ = true;
      pre_rotation_ = cv::ROTATE_180;
      post_rotation_ = cv::ROTATE_180;
    } else {
      do_rotation_ = false;
    }

    should_shutdown_ = false;
    spin_thread_.reset(new std::thread(&SegmentationNodelet::spin, this));
    sub_ = transport_->subscribe("image_raw", 1, &SegmentationNodelet::callback, this);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }
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
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  SLOG(DEBUG) << "Encoding: " << img_ptr->encoding << " size: " << img_ptr->image.cols
              << " x " << img_ptr->image.rows << " x " << img_ptr->image.channels()
              << " is right type? "
              << (img_ptr->image.type() == CV_8UC3 ? "yes" : "no");

  cv::Mat rotated;
  if (do_rotation_) {
    cv::rotate(img_ptr->image, rotated, pre_rotation_);
  } else {
    rotated = img_ptr->image;
  }

  const auto result = segmenter_->infer(rotated);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return;
  }

  cv::Mat rerotated;
  if (do_rotation_) {
    cv::rotate(result.labels, rerotated, post_rotation_);
  } else {
    rerotated = result.labels;
  }

  output_pub_->publish(img_ptr->header, rerotated, img_ptr->image);
}

bool SegmentationNodelet::haveWork() const {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  return !image_queue_.empty();
}

void SegmentationNodelet::spin() {

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
        const auto curr_diff_s =
            std::abs((image_queue_.front()->header.stamp - *last_time).toSec());
        SLOG(DEBUG) << "current time diff: " << curr_diff_s
                    << "[s] (min: " << config_.image_separation_s << "[s])";
        if (curr_diff_s < config_.image_separation_s) {
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
  if (config_.max_queue_size > 0 && image_queue_.size() >= config_.max_queue_size) {
    image_queue_.pop_front();
  }

  image_queue_.push_back(msg);
}

}  // namespace semantic_recolor

PLUGINLIB_EXPORT_CLASS(semantic_recolor::SegmentationNodelet, nodelet::Nodelet)
