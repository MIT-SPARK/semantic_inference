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
