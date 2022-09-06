#pragma once
#include "semantic_recolor/model_config.h"
#include "semantic_recolor/semantic_color_config.h"

#include <yaml-cpp/yaml.h>

namespace semantic_recolor {

ModelConfig readModelConfigFromYaml(const YAML::Node& node);

DepthConfig readDepthModelConfigFromYaml(const YAML::Node& node);

SemanticColorConfig readSemanticColorConfigFromYaml(const YAML::Node& node);

void printModelConfig(const ModelConfig& config);

void printDepthModelConfig(const DepthConfig& config);

}  // namespace semantic_recolor
