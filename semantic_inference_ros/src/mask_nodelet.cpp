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
#include <ianvs/image_publisher.h>
#include <ianvs/image_subscription.h>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/node.hpp>

namespace semantic_inference {

using sensor_msgs::msg::Image;

class MaskNode : public rclcpp::Node {
 public:
  explicit MaskNode(const rclcpp::NodeOptions& options);
  void callback(const Image::ConstSharedPtr& msg);

 private:
  cv_bridge::CvImagePtr result_image_;
  ianvs::ImagePublisher pub_;
  ianvs::ImageSubscription sub_;
  cv::Mat mask_;
};

MaskNode::MaskNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("mask_node", options), sub_(*this) {
  declare_parameter<std::string>("mask_path", "");

  std::string mask_path = "";
  if (!get_parameter("mask_path", mask_path) || mask_path.empty()) {
    RCLCPP_FATAL(get_logger(), "Mask path is required!");
    throw std::runtime_error("mask path not specified");
  }

  RCLCPP_INFO_STREAM(get_logger(), "Reading mask from '" << mask_path << "'");
  mask_ = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
  if (mask_.empty()) {
    RCLCPP_FATAL(get_logger(), "Invalid mask; mat is empty");
    throw std::runtime_error("invalid mask!");
  }

  pub_ = ianvs::ImagePublisher(*this, "masked/image_raw", 1);

  sub_.registerCallback(&MaskNode::callback, this);
  sub_.subscribe("input/image_raw");
}

void MaskNode::callback(const Image::ConstSharedPtr& msg) {
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(msg);
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR_STREAM(get_logger(), "Image conversion error: " << e.what());
    return;
  }

  if (!result_image_) {
    result_image_.reset(new cv_bridge::CvImage());
    result_image_->encoding = img_ptr->encoding;
    result_image_->image =
        cv::Mat(img_ptr->image.rows, img_ptr->image.cols, img_ptr->image.type());
  }

  result_image_->image.setTo(0);
  result_image_->header = img_ptr->header;
  cv::bitwise_or(img_ptr->image, img_ptr->image, result_image_->image, mask_);

  pub_.publish(result_image_->toImageMsg());
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::MaskNode)
