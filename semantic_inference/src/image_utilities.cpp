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

#include "semantic_inference/image_utilities.h"

#include <config_utilities/config.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "semantic_inference/logging.h"

namespace semantic_inference {

float ColorConverter::convert(uint8_t input_val, size_t channel) const {
  float to_return = config.map_to_unit_range ? (input_val / 255.0f) : input_val;
  to_return = config.normalize
                  ? (to_return - config.mean[channel]) / config.stddev[channel]
                  : to_return;
  return to_return;
}

void ColorConverter::fillImage(const cv::Mat& input, cv::Mat& output) const {
  if (output.size.dims() != 3) {
    SLOG(ERROR) << "Invalid output matrix!";
    return;
  }

  const bool is_chw_order = output.size[0] == 3;
  const int rows = is_chw_order ? output.size[1] : output.size[0];
  const int cols = is_chw_order ? output.size[2] : output.size[1];

  cv::Mat img;
  if (input.cols == output.cols && input.rows == output.rows) {
    img = input;
  } else {
    cv::resize(input, img, cv::Size(cols, rows));
  }

  std::array<int, 3> input_addr;
  if (config.rgb_order) {
    input_addr = {2, 1, 0};
  } else {
    input_addr = {0, 1, 2};
  }

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      const uint8_t* pixel = img.ptr<uint8_t>(row, col);
      if (is_chw_order) {
        output.at<float>(0, row, col) = convert(pixel[input_addr[0]], 0);
        output.at<float>(1, row, col) = convert(pixel[input_addr[1]], 1);
        output.at<float>(2, row, col) = convert(pixel[input_addr[2]], 2);
      } else {
        output.at<float>(row, col, 0) = convert(pixel[input_addr[0]], 0);
        output.at<float>(row, col, 1) = convert(pixel[input_addr[1]], 1);
        output.at<float>(row, col, 2) = convert(pixel[input_addr[2]], 2);
      }
    }
  }
}

void declare_config(ColorConverter::Config& config) {
  using namespace config;
  name("ColorConverter::Config");
  field(config.mean, "mean");
  field(config.stddev, "stddev");
  field(config.map_to_unit_range, "map_to_unit_range");
  field(config.normalize, "normalize");
  field(config.rgb_order, "rgb_order");
}

float DepthConverter::convert(float input_val) const {
  if (!config.normalize) {
    return input_val;
  }

  const float new_value = (input_val - config.mean) / config.stddev;
  if (new_value < 0.0f) {
    return 0.0f;
  }

  return new_value;
}

void DepthConverter::fillImage(const cv::Mat& input, cv::Mat& output) const {
  if (output.size.dims() != 2) {
    SLOG(ERROR) << "Invalid output matrix!";
    return;
  }

  const bool size_ok = input.cols == output.cols && input.rows == output.rows;
  cv::Mat img;
  if (size_ok) {
    img = input;
  } else {
    cv::resize(input, img, cv::Size(output.cols, output.rows), 0, 0, cv::INTER_NEAREST);
  }

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      output.at<float>(row, col) = convert(img.at<float>(row, col));
    }
  }
}

void declare_config(DepthConverter::Config& config) {
  using namespace config;
  name("DepthConverter::Config");
  field(config.mean, "mean");
  field(config.stddev, "stddev");
  field(config.normalize, "normalize");
}

cv::Mat DepthLabelMask::maskLabels(const cv::Mat& labels, const cv::Mat& depth) const {
  cv::Mat resized_depth;
  cv::resize(depth,
             resized_depth,
             cv::Size(labels.cols, labels.rows),
             0,
             0,
             cv::INTER_NEAREST);

  cv::Mat mask;
  cv::inRange(resized_depth, config.min_depth, config.max_depth, mask);

  cv::Mat masked_labels;
  cv::bitwise_or(labels, labels, masked_labels, mask);
  return masked_labels;
}

void declare_config(DepthLabelMask::Config& config) {
  using namespace config;
  name("DepthLabelMask::Config");
  field(config.min_depth, "min_depth");
  field(config.max_depth, "max_depth");
}

std::string getLabelPercentages(const cv::Mat& labels) {
  std::map<int32_t, size_t> counts;
  std::vector<int32_t> unique_classes;
  for (int r = 0; r < labels.rows; ++r) {
    for (int c = 0; c < labels.cols; ++c) {
      int32_t class_id = 0;
      if (labels.type() == CV_32SC1) {
        class_id = labels.at<int32_t>(r, c);
      } else if (labels.type() == CV_16SC1) {
        class_id = labels.at<int16_t>(r, c);
      } else {
        return "invalid type: " + std::to_string(labels.type());
      }

      if (!counts.count(class_id)) {
        counts[class_id] = 0;
        unique_classes.push_back(class_id);
      }

      counts[class_id]++;
    }
  }

  double total = static_cast<double>(labels.rows * labels.cols);
  std::sort(unique_classes.begin(),
            unique_classes.end(),
            [&](const int32_t& lhs, const int32_t& rhs) {
              return counts[lhs] > counts[rhs];
            });

  std::stringstream ss;
  ss << " Class pixel percentages:" << std::endl;
  for (const int32_t id : unique_classes) {
    ss << "  - " << id << ": " << static_cast<double>(counts[id]) / total * 100.0 << "%"
       << std::endl;
  }

  return ss.str();
}

}  // namespace semantic_inference
