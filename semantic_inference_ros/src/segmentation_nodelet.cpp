#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <semantic_inference/image_rotator.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <atomic>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <thread>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference {

class SegmentationNodelet : public nodelet::Nodelet {
 public:
  using ImageWorker = Worker<sensor_msgs::ImageConstPtr>;

  struct Config {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    WorkerConfig worker;
    ImageRotator::Config image_rotator;
  };

  virtual void onInit() override;

  virtual ~SegmentationNodelet();

 private:
  void runSegmentation(const sensor_msgs::ImageConstPtr& msg);

  Config config_;
  std::unique_ptr<Segmenter> segmenter_;
  ImageRotator image_rotator_;
  std::unique_ptr<ImageWorker> worker_;

  std::unique_ptr<image_transport::ImageTransport> transport_;
  std::unique_ptr<OutputPublisher> output_pub_;
  image_transport::Subscriber sub_;
};

void declare_config(SegmentationNodelet::Config& config) {
  using namespace config;
  name("SegmentationNodelet::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.worker, "worker");
  field(config.image_rotator, "image_rotator");
}

void SegmentationNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>());

  config_ = config::fromRos<SegmentationNodelet::Config>(nh);
  SLOG(INFO) << "\n" << config::toString(config_);
  config::checkValid(config_);

  try {
    segmenter_ = std::make_unique<Segmenter>(config_.segmenter);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }

  image_rotator_ = ImageRotator(config_.image_rotator);

  transport_ = std::make_unique<image_transport::ImageTransport>(nh);
  output_pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);
  worker_ = std::make_unique<ImageWorker>(
      config_.worker,
      [this](const auto& msg) { runSegmentation(msg); },
      [](const auto& msg) { return msg->header.stamp; });

  sub_ = transport_->subscribe(
      "color/image_raw", 1, &ImageWorker::addMessage, worker_.get());
}

SegmentationNodelet::~SegmentationNodelet() {
  if (worker_) {
    worker_->stop();
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

  const auto rotated = image_rotator_.rotate(img_ptr->image);
  const auto result = segmenter_->infer(rotated);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return;
  }

  const auto derotated = image_rotator_.rotate(result.labels);
  output_pub_->publish(img_ptr->header, derotated, img_ptr->image);
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::SegmentationNodelet, nodelet::Nodelet)
