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

#pragma once
#include <opencv2/core/mat.hpp>

namespace semantic_inference {

struct ColorConverter {
  struct Config {
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stddev{0.229f, 0.224f, 0.225f};
    bool map_to_unit_range = true;
    bool normalize = true;
    bool rgb_order = true;
  } const config;

  explicit ColorConverter(const Config& config) : config(config) {}
  ColorConverter() : ColorConverter(Config()) {}
  float convert(uint8_t input_val, size_t channel) const;
  void fillImage(const cv::Mat& input, cv::Mat& output) const;
};

void declare_config(ColorConverter::Config& config);

struct DepthConverter {
  struct Config {
    float mean = 0.213;
    float stddev = 0.285;
    bool normalize = false;
  } const config;

  explicit DepthConverter(const Config& config) : config(config) {}
  DepthConverter() : DepthConverter(Config()) {}
  float convert(float input_val) const;
  void fillImage(const cv::Mat& input, cv::Mat& output) const;
};

void declare_config(DepthConverter::Config& config);

struct DepthLabelMask {
  struct Config {
    float min_depth = 0.1f;
    float max_depth = 10.0f;
  } const config;

  DepthLabelMask(const Config& config) : config(config) {}
  DepthLabelMask() : DepthLabelMask(Config()) {}
  cv::Mat maskLabels(const cv::Mat& labels, const cv::Mat& depth) const;
};

void declare_config(DepthLabelMask::Config& config);

std::string getLabelPercentages(const cv::Mat& labels);

}  // namespace semantic_inference
