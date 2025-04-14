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
#include <config_utilities/validation.h>
#include <ianvs/image_publisher.h>
#include <semantic_inference/image_utilities.h>
#include <semantic_inference/logging.h>

#include <cv_bridge/cv_bridge.hpp>

namespace semantic_inference {

struct OutputPublisher::Impl {
  Impl(const OutputPublisher::Config& config, OutputPublisher::Interface node)
      : config(config) {
    // TODO(nathan) think about param for alpha
    if (config.publish_labels) {
      label_pub = ianvs::ImagePublisher(node, "semantic/image_raw", 1);
    }

    if (config.publish_color) {
      color_pub = ianvs::ImagePublisher(node, "semantic_color/image_raw", 1);
    }

    if (config.publish_overlay) {
      overlay_pub = ianvs::ImagePublisher(node, "semantic_overlay/image_raw", 1);
    }
  }

  void publish(const ImageRecolor& recolor,
               const std_msgs::msg::Header& header,
               const cv::Mat& labels,
               const cv::Mat& color) {
    // TODO(nathan need to check subscription count before doing work
    if (label_image.image.empty()) {
      // we can't support 32SC1, so we do 16SC1 signed to distinguish from depth
      label_image.encoding = "16SC1";
      label_image.image = cv::Mat(color.rows, color.cols, CV_16SC1);
    }

    label_image.header = header;
    recolor.relabelImage(labels, label_image.image);
    label_pub.publish(label_image.toImageMsg());

    const auto publish_overlay = !config.publish_overlay || color.empty();
    if (!config.publish_color && publish_overlay) {
      return;
    }

    if (color_image.image.empty()) {
      color_image.encoding = "rgb8";
      color_image.image = cv::Mat(color.rows, color.cols, CV_8UC3);
    }

    color_image.header = header;
    recolor.recolorImage(label_image.image, color_image.image);
    if (config.publish_color) {
      color_pub.publish(color_image.toImageMsg());
    }

    if (!config.publish_overlay) {
      return;
    }

    if (overlay_image.image.empty()) {
      overlay_image.encoding = "rgb8";
      overlay_image.image = cv::Mat(color.rows, color.cols, CV_8UC3);
    }

    overlay_image.header = header;
    cv::addWeighted(color_image.image,
                    config.overlay_alpha,
                    color,
                    (1.0 - config.overlay_alpha),
                    0.0,
                    overlay_image.image);
    overlay_pub.publish(overlay_image.toImageMsg());
  }

  const OutputPublisher::Config config;
  ianvs::ImagePublisher label_pub;
  ianvs::ImagePublisher color_pub;
  ianvs::ImagePublisher overlay_pub;
  cv_bridge::CvImage label_image;
  cv_bridge::CvImage color_image;
  cv_bridge::CvImage overlay_image;
};

OutputPublisher::OutputPublisher(const Config& config, Interface node)
    : config(config::checkValid(config)),
      image_recolor_(config.recolor),
      impl_(std::make_unique<Impl>(config, node)) {}

OutputPublisher::~OutputPublisher() = default;

void OutputPublisher::publish(const std_msgs::msg::Header& header,
                              const cv::Mat& labels,
                              const cv::Mat& color) {
  if (labels.empty()) {
    SLOG(ERROR) << "Labels required!";
    return;
  }

  impl_->publish(image_recolor_, header, labels, color);
}

void declare_config(OutputPublisher::Config& config) {
  config::name("OutputPublisher::Config");
  config::field(config.recolor, "recolor");
  config::field(config.publish_labels, "publish_labels");
  config::field(config.publish_color, "publish_color");
  config::field(config.publish_overlay, "publish_overlay");
  config::field(config.overlay_alpha, "overlay_alpha");
}

}  // namespace semantic_inference
