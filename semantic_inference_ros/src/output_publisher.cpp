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

#include "semantic_inference_ros/output_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/validation.h>
#include <semantic_inference/image_utilities.h>
#include <semantic_inference/logging.h>

namespace semantic_inference {

using image_transport::ImageTransport;

OutputPublisher::OutputPublisher(const Config& config, ImageTransport& transport)
    : config(config::checkValid(config)), image_recolor_(config.recolor) {
  if (config.publish_labels) {
    label_pub_ = transport.advertise("semantic/image_raw", 1);
  }

  if (config.publish_color) {
    color_pub_ = transport.advertise("semantic_color/image_raw", 1);
  }

  if (config.publish_overlay) {
    overlay_pub_ = transport.advertise("semantic_overlay/image_raw", 1);
  }
}

void OutputPublisher::publish(const std_msgs::Header& header,
                              const cv::Mat& labels,
                              const cv::Mat& color) {
  if (labels.empty() || color.empty()) {
    SLOG(ERROR) << "Invalid inputs: color=" << std::boolalpha << !color.empty()
                << ", labels=" << !labels.empty();
    return;
  }

  if (!label_image_) {
    label_image_.reset(new cv_bridge::CvImage());
    // we can't support 32 signed labels, so we do 16-bit signed to distinguish from
    // depth
    label_image_->encoding = "16SC1";
    label_image_->image = cv::Mat(color.rows, color.cols, CV_16SC1);
  }

  label_image_->header = header;
  image_recolor_.relabelImage(labels, label_image_->image);
  if (config.publish_labels) {
    label_pub_.publish(label_image_->toImageMsg());
  }

  if (!config.publish_color && !config.publish_overlay) {
    return;
  }

  if (!color_image_) {
    color_image_.reset(new cv_bridge::CvImage());
    color_image_->encoding = "rgb8";
    color_image_->image = cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  color_image_->header = header;
  image_recolor_.recolorImage(label_image_->image, color_image_->image);
  if (config.publish_color) {
    color_pub_.publish(color_image_->toImageMsg());
  }

  if (!config.publish_overlay) {
    return;
  }

  if (!overlay_image_) {
    overlay_image_.reset(new cv_bridge::CvImage());
    overlay_image_->encoding = "rgb8";
    overlay_image_->image = cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  overlay_image_->header = header;

  cv::addWeighted(color_image_->image,
                  config.overlay_alpha,
                  color,
                  (1.0 - config.overlay_alpha),
                  0.0,
                  overlay_image_->image);

  overlay_pub_.publish(overlay_image_->toImageMsg());
}

void declare_config(OutputPublisher::Config& config) {
  using namespace config;
  name("OutputPublisher::Config");
  field(config.recolor, "recolor");
  field(config.publish_labels, "publish_labels");
  field(config.publish_color, "publish_color");
  field(config.publish_overlay, "publish_overlay");
  field(config.overlay_alpha, "overlay_alpha");
}

}  // namespace semantic_inference
