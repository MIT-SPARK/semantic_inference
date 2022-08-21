#pragma once
#include "semantic_recolor/trt_utilities.h"

#include <vector>
#include <array>

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

  void fillInputAddress(ImageAddress &addr) const;

  float getValue(float input_val, size_t channel) const;

  nvinfer1::Dims4 getInputDims(int channels) const;

  std::vector<int> getInputMatDims(int channels) const;
};

}  // namespace semantic_recolor
