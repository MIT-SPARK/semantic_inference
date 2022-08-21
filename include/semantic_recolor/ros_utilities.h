#pragma once
#include "semantic_recolor/model_config.h"

#include <ros/ros.h>

namespace semantic_recolor {

ModelConfig readModelConfig(const ros::NodeHandle& nh);

void showModelConfig(const ModelConfig& config);

}  // namespace semantic_recolor
