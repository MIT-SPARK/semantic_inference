/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * -------------------------------------------------------------------------- */

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/settings.h>
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
    WorkerConfig worker;
    ImageRotator::Config image_rotator;
    bool show_config = true;
    bool show_output_config = false;
  };

  virtual void onInit() override;

  virtual ~SegmentationNodelet();

 private:
  void runSegmentation(const sensor_msgs::ImageConstPtr& msg);

  Config config_;
  OutputPublisher::Config output_;

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
  field(config.worker, "worker");
  field(config.image_rotator, "image_rotator");
  field(config.show_config, "show_config");
  field(config.show_output_config, "show_output_config");
}

void SegmentationNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>());
  logging::setConfigUtilitiesLogger();

  config_ = config::fromRos<SegmentationNodelet::Config>(nh);
  // NOTE(nathan) parsed separately to avoid spamming console with labelspace remapping
  output_ = config::fromRos<OutputPublisher::Config>(nh, "output");

  if (config_.show_config) {
    SLOG(INFO) << "\n" << config::toString(config_);
    if (config_.show_output_config) {
      SLOG(INFO) << "\n" << config::toString(output_);
    }
  }

  config::checkValid(config_);

  try {
    segmenter_ = std::make_unique<Segmenter>(config_.segmenter);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }

  image_rotator_ = ImageRotator(config_.image_rotator);

  transport_ = std::make_unique<image_transport::ImageTransport>(nh);
  output_pub_ = std::make_unique<OutputPublisher>(output_, *transport_);
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

  const auto derotated = image_rotator_.derotate(result.labels);
  output_pub_->publish(img_ptr->header, derotated, img_ptr->image);
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::SegmentationNodelet, nodelet::Nodelet)
