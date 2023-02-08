#pragma once
#include <array>
#include <vector>

#include "semantic_recolor/trt_utilities.h"

namespace semantic_recolor {

struct ModelConfig {
  using ImageAddress = std::array<int, 3>;

  std::string model_file;
  std::string engine_file;
  int width;
  int height;
  std::string input_name;
  std::string output_name;
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stddev{0.229f, 0.224f, 0.225f};
  bool map_to_unit_range = true;
  bool normalize = true;
  bool use_network_order = true;
  bool network_uses_rgb_order = true;
  bool use_ros_logging = true;
  Severity log_severity = Severity::kINFO;
  bool show_stats = false;
  bool set_builder_flags = false;

  void fillInputAddress(ImageAddress& addr) const;

  float getValue(uint8_t input_val, size_t channel) const;

  nvinfer1::Dims4 getInputDims(int channels) const;

  std::vector<int> getInputMatDims(int channels) const;
};

struct DepthConfig {
  std::string depth_input_name;

  float depth_mean = 0.213;
  float depth_stddev = 0.285;
  bool normalize_depth = false;
  bool mask_predictions = true;
  float min_depth = 0.1f;
  float max_depth = 10.0f;

  float getValue(float input_val) const;
};

}  // namespace semantic_recolor
