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
#include <config_utilities/parsing/commandline.h>
#include <ianvs/image_subscription.h>
#include <semantic_inference/image_rotator.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference {

using sensor_msgs::msg::Image;

class SegmentationNode : public rclcpp::Node {
 public:
  using ImageWorker = Worker<Image::ConstSharedPtr>;

  struct Config {
    Segmenter::Config segmenter;
    WorkerConfig worker;
    ImageRotator::Config image_rotator;
    bool show_output_config = false;
  } const config;

  explicit SegmentationNode(const rclcpp::NodeOptions& options);
  virtual ~SegmentationNode();

  const OutputPublisher::Config output_config;

 private:
  void runSegmentation(const Image::ConstSharedPtr& msg);

  std::unique_ptr<Segmenter> segmenter_;
  std::unique_ptr<ImageWorker> worker_;

  OutputPublisher output_pub_;
  ImageRotator image_rotator_;
  ianvs::ImageSubscription sub_;
};

void declare_config(SegmentationNode::Config& config) {
  using namespace config;
  name("SegmentationNode::Config");
  field(config.segmenter, "segmenter");
  field(config.worker, "worker");
  field(config.image_rotator, "image_rotator");
  field(config.show_output_config, "show_output_config");
}

SegmentationNode::SegmentationNode(const rclcpp::NodeOptions& options)
    : Node("segmentation_node", options),
      config(config::fromCLI<Config>(options.arguments())),
      output_config(config::fromCLI<Config>(options.arguments(), "output")),
      output_pub_(output_config, *this),
      image_rotator_(config.image_rotator),
      sub_(*this) {
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>(get_logger()));
  logging::setConfigUtilitiesLogger();
  SLOG(INFO) << "\n" << config::toString(config);
  if (config_.show_output_config) {
    SLOG(INFO) << "\n" << config::toString(output_);
  }

  config::checkValid(config);
  config::checkValid(output_config);

  try {
    segmenter_ = std::make_unique<Segmenter>(config.segmenter);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }

  worker_ = std::make_unique<ImageWorker>(
      config.worker,
      [this](const auto& msg) { runSegmentation(msg); },
      [](const auto& msg) { return msg->header.stamp; });

  sub_.registerCallback(&ImageWorker::addMessage, worker_.get());
  sub_.subscribe("color/image_raw");
}

SegmentationNode::~SegmentationNode() {
  if (worker_) {
    worker_->stop();
  }
}

void SegmentationNode::runSegmentation(const Image::ConstSharedPtr& msg) {
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
  output_pub_.publish(img_ptr->header, derotated, img_ptr->image);
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::SegmentationNode)
