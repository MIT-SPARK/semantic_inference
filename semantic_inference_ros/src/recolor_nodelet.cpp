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

#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference {

using message_filters::Synchronizer;
using sensor_msgs::msg::Image;

struct ColorLabelPacket {
  Image::ConstSharedPtr color;
  Image::ConstSharedPtr labels;
};

class RecolorNode : public rclcpp::Node {
 public:
  using ImageWorker = Worker<ColorLabelPacket>;
  using SyncPolicy = message_filters::sync_policies::ExactTime<Image, Image>;

  struct Config {
    OutputPublisher::Config output;
    WorkerConfig worker;
  };

  explicit RecolorNode(const rclcpp::NodeOptions& options);
  virtual ~RecolorNode();

 private:
  void callback(const Image::ConstSharedPtr& color,
                const Image::ConstSharedPtr& labels);

  void publish(const ColorLabelPacket& packet) const;

  Config config_;
  std::unique_ptr<ImageWorker> worker_;

  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::SubscriberFilter color_sub_;
  image_transport::SubscriberFilter label_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::unique_ptr<OutputPublisher> pub_;
};

void declare_config(RecolorNode::Config& config) {
  config::name("RecolorNode::Config");
  config::field(config.output, "output");
  config::field(config.worker, "worker");
}

RecolorNode::RecolorNode(const rclcpp::NodeOptions& options)
    : Node("recolor_node", options) {
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>(get_logger()));
  logging::setConfigUtilitiesLogger();

  transport_.reset(new image_transport::ImageTransport(shared_from_this()));

  // config_ = config::fromRos<RecolorNode::Config>(nh);
  SLOG(INFO) << "\n" << config::toString(config_);
  config::checkValid(config_);

  pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);

  worker_ = std::make_unique<ImageWorker>(
      config_.worker,
      [this](const auto& msg) { publish(msg); },
      [](const auto& msg) { return msg.color->header.stamp; });

  color_sub_.subscribe(this, "color/image_raw", "raw");
  label_sub_.subscribe(this, "labels/image_raw", "raw");
  sync_.reset(new Synchronizer<SyncPolicy>(SyncPolicy(10), color_sub_, label_sub_));
  sync_->registerCallback(std::bind(
      &RecolorNode::callback, this, std::placeholders::_1, std::placeholders::_2));
}

void RecolorNode::callback(const Image::ConstSharedPtr& color,
                           const Image::ConstSharedPtr& labels) {
  worker_->addMessage({color, labels});
}

RecolorNode::~RecolorNode() {
  if (worker_) {
    worker_->stop();
  }
}

void RecolorNode::publish(const ColorLabelPacket& msg) const {
  cv_bridge::CvImageConstPtr color_ptr;
  try {
    color_ptr = cv_bridge::toCvShare(msg.color, "rgb8");
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  cv_bridge::CvImageConstPtr label_ptr;
  try {
    label_ptr = cv_bridge::toCvShare(msg.labels);
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  pub_->publish(color_ptr->header, label_ptr->image, color_ptr->image);
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::RecolorNode)
