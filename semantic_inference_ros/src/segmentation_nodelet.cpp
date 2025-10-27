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
#include <chrono>
#include <mutex>
#include <optional>
#include <thread>

#include <cv_bridge/cv_bridge.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <std_msgs/msg/string.hpp>

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
    bool show_config = true;
    bool show_output_config = false;
    struct Status {
      std::string nickname = "semantic_inference";
      size_t rate_window_size = 10;
      double period_s = 0.5;
      double max_heartbeat_s = 10.0;
    } status;
  } const config;

  explicit SegmentationNode(const rclcpp::NodeOptions& options);
  virtual ~SegmentationNode();

  const OutputPublisher::Config output_config;

 private:
  void runSegmentation(const Image::ConstSharedPtr& msg);

  void recordStatus(double elapsed_s, const std::string& error = "");

  void publishStatus();

  std::unique_ptr<Segmenter> segmenter_;
  std::unique_ptr<ImageWorker> worker_;

  std::mutex status_mutex_;
  std::optional<rclcpp::Time> last_call_;
  std::string last_status_;
  std::list<double> elapsed_samples_s_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

  OutputPublisher output_pub_;
  ImageRotator image_rotator_;
  ianvs::ImageSubscription sub_;
};

void declare_config(SegmentationNode::Config::Status& config) {
  using namespace config;
  name("SegmentationNode::Config");
  field(config.nickname, "nickname");
  field(config.rate_window_size, "rate_window_size");
  field(config.period_s, "period_s");
  field(config.max_heartbeat_s, "max_heartbeat_s");

  checkCondition(!config.nickname.empty(), "nickname is empty");
  check(config.rate_window_size, GT, 0, "rate_window_size");
  check(config.period_s, GT, 0.0, "period_s");
  check(config.max_heartbeat_s, GT, 0.0, "max_heartbeat_s");
}

void declare_config(SegmentationNode::Config& config) {
  using namespace config;
  name("SegmentationNode::Config");
  field(config.segmenter, "segmenter");
  field(config.worker, "worker");
  field(config.image_rotator, "image_rotator");
  field(config.show_config, "show_config");
  field(config.show_output_config, "show_output_config");
  field(config.status, "status");
}

SegmentationNode::SegmentationNode(const rclcpp::NodeOptions& options)
    : Node("segmentation_node", options),
      config(config::fromCLI<Config>(options.arguments())),
      output_config(
          config::fromCLI<OutputPublisher::Config>(options.arguments(), "output")),
      output_pub_(output_config, *this),
      image_rotator_(config.image_rotator),
      sub_(*this) {
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>(get_logger()));
  logging::setConfigUtilitiesLogger();
  if (config.show_config) {
    SLOG(INFO) << "\n" << config::toString(config);
  }

  if (config.show_output_config) {
    SLOG(INFO) << "\n" << config::toString(output_config);
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

  const auto period_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::duration<double>(config.status.period_s));
  status_pub_ = create_publisher<std_msgs::msg::String>("~/status", rclcpp::QoS(1));
  timer_ =
      create_wall_timer(period_ms, std::bind(&SegmentationNode::publishStatus, this));
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
    recordStatus(0.0, "Conversion error: " + std::string(e.what()));
    return;
  }

  SLOG(DEBUG) << "Encoding: " << img_ptr->encoding << " size: " << img_ptr->image.cols
              << " x " << img_ptr->image.rows << " x " << img_ptr->image.channels()
              << " is right type? "
              << (img_ptr->image.type() == CV_8UC3 ? "yes" : "no");

  const auto start = std::chrono::steady_clock::now();
  const auto rotated = image_rotator_.rotate(img_ptr->image);
  const auto result = segmenter_->infer(rotated);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    recordStatus(0.0, "Failed to run inference");
    return;
  }

  const auto derotated = image_rotator_.derotate(result.labels);
  output_pub_.publish(img_ptr->header, derotated, img_ptr->image);
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  recordStatus(elapsed.count());
}

void SegmentationNode::recordStatus(double elapsed_s, const std::string& error) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  last_call_ = now();
  last_status_ = error;
  if (!error.empty()) {
    return;
  }

  elapsed_samples_s_.push_back(elapsed_s);
  if (elapsed_samples_s_.size() > config.status.rate_window_size) {
    elapsed_samples_s_.pop_front();
  }
}

void SegmentationNode::publishStatus() {
  std::lock_guard<std::mutex> lock(status_mutex_);
  std::chrono::nanoseconds curr_time_ns(now().nanoseconds());

  nlohmann::json record;
  record["nickname"] = config.status.nickname;
  record["node_name"] = get_fully_qualified_name();
  if (!last_call_) {
    record["status"] = "WARNING";
    record["note"] = "Waiting for input";
  } else {
    const auto diff = now() - *last_call_;
    if (diff.seconds() > config.status.max_heartbeat_s) {
      record["status"] = "ERROR";
      std::stringstream ss;
      ss << "No input processed in " << diff.seconds() << " s";
      record["note"] = ss.str();
    } else if (!last_status_.empty()) {
      record["status"] = "ERROR";
      record["note"] = last_status_;
    } else {
      double average_elapsed_s = 0.0;
      for (const auto sample : elapsed_samples_s_) {
        average_elapsed_s += sample;
      }
      if (elapsed_samples_s_.size()) {
        average_elapsed_s /= elapsed_samples_s_.size();
      }

      record["status"] = "NOMINAL";
      std::stringstream ss;
      ss << "Average elapsed time: " << average_elapsed_s << " s";
      record["note"] = ss.str();
    }
  }

  std::stringstream ss;
  ss << record;

  auto msg = std::make_unique<std_msgs::msg::String>();
  msg->data = ss.str();
  status_pub_->publish(std::move(msg));
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::SegmentationNode)
