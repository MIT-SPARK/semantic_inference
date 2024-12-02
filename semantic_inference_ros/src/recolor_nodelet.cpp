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

#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference {

using message_filters::Synchronizer;

struct ColorLabelPacket {
  sensor_msgs::ImageConstPtr color;
  sensor_msgs::ImageConstPtr labels;
};

class RecolorNodelet : public nodelet::Nodelet {
 public:
  using ImageWorker = Worker<ColorLabelPacket>;
  using SyncPolicy =
      message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image>;

  struct Config {
    OutputPublisher::Config output;
    WorkerConfig worker;
  };

  virtual void onInit() override;

  virtual ~RecolorNodelet();

 private:
  void publish(const ColorLabelPacket& packet) const;

  void callback(const sensor_msgs::ImageConstPtr& color,
                const sensor_msgs::ImageConstPtr& labels);

  Config config_;
  std::unique_ptr<ImageWorker> worker_;

  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::SubscriberFilter color_sub_;
  image_transport::SubscriberFilter label_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::unique_ptr<OutputPublisher> pub_;
};

void declare_config(RecolorNodelet::Config& config) {
  using namespace config;
  name("RecolorNodelet::Config");
  field(config.output, "output");
  field(config.worker, "worker");
}

void RecolorNodelet::onInit() {
  ros::NodeHandle nh = getPrivateNodeHandle();
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>());
  logging::setConfigUtilitiesLogger();

  transport_.reset(new image_transport::ImageTransport(nh));

  config_ = config::fromRos<RecolorNodelet::Config>(nh);
  SLOG(INFO) << "\n" << config::toString(config_);
  config::checkValid(config_);

  pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);

  worker_ = std::make_unique<ImageWorker>(
      config_.worker,
      [this](const auto& msg) { publish(msg); },
      [](const auto& msg) { return msg.color->header.stamp; });

  color_sub_.subscribe(*transport_, "color/image_raw", 1);
  label_sub_.subscribe(*transport_, "labels/image_raw", 1);
  sync_.reset(new Synchronizer<SyncPolicy>(SyncPolicy(10), color_sub_, label_sub_));
  sync_->registerCallback(boost::bind(&RecolorNodelet::callback, this, _1, _2));
}

RecolorNodelet::~RecolorNodelet() {
  if (worker_) {
    worker_->stop();
  }
}

void RecolorNodelet::publish(const ColorLabelPacket& msg) const {
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

void RecolorNodelet::callback(const sensor_msgs::ImageConstPtr& color,
                              const sensor_msgs::ImageConstPtr& labels) {
  worker_->addMessage({color, labels});
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::RecolorNodelet, nodelet::Nodelet)
