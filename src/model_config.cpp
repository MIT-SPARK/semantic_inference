#include "semantic_recolor/model_config.h"

namespace semantic_recolor {

void ModelConfig::fillInputAddress(ImageAddress &addr) const {
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

nvinfer1::Dims4 ModelConfig::getInputDims(int channels) const {
  if (use_network_order) {
    return {1, channels, height, width};
  } else {
    return {1, height, width, channels};
  }
}

std::vector<int> ModelConfig::getInputMatDims(int channels) const {
  if (use_network_order) {
    return {channels, height, width};
  } else {
    return {height, width, channels};
  }
}

}  // namespace semantic_recolor
