#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

struct ModelConfig {
  std::filesystem::path model_file;
  std::filesystem::path engine_file;
  std::string log_severity = "INFO";

  ColorConverter::Config color;
  DepthConverter::Config depth;
};

void declare_config(ModelConfig& config);

}  // namespace semantic_recolor
