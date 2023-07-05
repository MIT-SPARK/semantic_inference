#include "semantic_recolor/model_config.h"

namespace semantic_recolor {

void ModelConfig::fillInputAddress(ImageAddress& addr) const {
  if (network_uses_rgb_order) {
    addr = {2, 1, 0};
  } else {
    addr = {0, 1, 2};
  }
}

float ModelConfig::getValue(uint8_t input_val, size_t channel) const {
  float to_return = map_to_unit_range ? (input_val / 255.0f) : input_val;
  to_return = normalize ? (to_return - mean[channel]) / stddev[channel] : to_return;
  return to_return;
}

std::vector<int> ModelConfig::getInputMatDims(int channels) const {
  if (use_network_order) {
    return {channels, height, width};
  } else {
    return {height, width, channels};
  }
}

float DepthConfig::getValue(float input_val) const {
  if (!normalize_depth) {
    return input_val;
  }

  const float new_value = (input_val - depth_mean) / depth_stddev;
  if (new_value < 0.0f) {
    return 0.0f;
  }

  return new_value;
}

}  // namespace semantic_recolor
