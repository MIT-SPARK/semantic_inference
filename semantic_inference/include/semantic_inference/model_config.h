#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "semantic_inference/image_utilities.h"

namespace semantic_inference {

struct ModelConfig {
  std::filesystem::path model_file;
  std::filesystem::path engine_file;
  std::string log_severity = "INFO";
  bool force_rebuild = false;

  ColorConverter::Config color;
  DepthConverter::Config depth;
};

void declare_config(ModelConfig& config);

}  // namespace semantic_inference