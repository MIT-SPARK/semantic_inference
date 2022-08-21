#pragma once
#include "semantic_recolor/model_config.h"

#include <ros/ros.h>

namespace semantic_recolor {

ModelConfig readModelConfig(const ros::NodeHandle &nh);

}  // namespace semantic_recolor
