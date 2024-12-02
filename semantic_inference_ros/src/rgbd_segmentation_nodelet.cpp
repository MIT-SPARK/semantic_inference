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
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"

namespace semantic_inference {

using message_filters::Synchronizer;

class RgbdSegmentationNodelet : public nodelet::Nodelet {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                      sensor_msgs::Image>;

  struct Config {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    double depth_scale = 1.0;
  };

  virtual void onInit() override;

 private:
  void callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                const sensor_msgs::ImageConstPtr& depth_msg);

  Config config_;

  std::unique_ptr<Segmenter> segmenter_;
  std::unique_ptr<image_transport::ImageTransport> transport_;
  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  std::unique_ptr<OutputPublisher> output_pub_;
};

void declare_config(RgbdSegmentationNodelet::Config& config) {
  using namespace config;
  name("RgbdSegmentationNodelet::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.depth_scale, "depth_scale");
  check(config.depth_scale, GT, 0.0, "depth_scale");
}

void RgbdSegmentationNodelet::onInit() {
  auto nh = getPrivateNodeHandle();
  logging::Logger::addSink("ros", std::make_shared<RosLogSink>());
  logging::setConfigUtilitiesLogger();

  config_ = config::fromRos<Config>(nh);
  SLOG(INFO) << "config: " << config::toString(config_);
  config::checkValid(config_);

  segmenter_ = std::make_unique<Segmenter>(config_.segmenter);
  transport_ = std::make_unique<image_transport::ImageTransport>(nh);
  output_pub_ = std::make_unique<OutputPublisher>(config_.output, *transport_);

  image_sub_.subscribe(*transport_, "color/image_raw", 1);
  depth_sub_.subscribe(*transport_, "depth/image_rect", 1);
  sync.reset(new Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, depth_sub_));
  sync->registerCallback(boost::bind(&RgbdSegmentationNodelet::callback, this, _1, _2));
}

void RgbdSegmentationNodelet::callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                                       const sensor_msgs::ImageConstPtr& depth_msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(rgb_msg, "rgb8");
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  cv_bridge::CvImageConstPtr depth_img_ptr;
  try {
    depth_img_ptr = cv_bridge::toCvShare(depth_msg);
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  cv::Mat depth_img_float;
  if (depth_img_ptr->image.type() == CV_32FC1) {
    depth_img_float = depth_img_ptr->image;
  } else {
    depth_img_ptr->image.convertTo(depth_img_float, CV_32FC1, config_.depth_scale);
  }

  const auto result = segmenter_->infer(img_ptr->image, depth_img_float);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return;
  }

  output_pub_->publish(img_ptr->header, img_ptr->image, result.labels);
}

}  // namespace semantic_inference

PLUGINLIB_EXPORT_CLASS(semantic_inference::RgbdSegmentationNodelet, nodelet::Nodelet)
