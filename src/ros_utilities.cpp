#include "semantic_recolor/ros_utilities.h"

namespace semantic_recolor {

#define READ_REQUIRED(nh, config, name)                      \
  if (!nh.getParam(#name, config.name)) {                    \
    ROS_FATAL("Missing " #name " when parsing ModelConfig"); \
    throw std::runtime_error("missing param " #name "!");    \
  }                                                          \
  static_assert(true, "")

ModelConfig readModelConfig(const ros::NodeHandle &nh) {
  ModelConfig config;

  READ_REQUIRED(nh, config, model_file);
  READ_REQUIRED(nh, config, width);
  READ_REQUIRED(nh, config, height);
  READ_REQUIRED(nh, config, input_name);
  READ_REQUIRED(nh, config, output_name);

  nh.getParam("engine_file", config.engine_file);
  nh.getParam("mean", config.mean);
  nh.getParam("stddev", config.stddev);
  nh.getParam("map_to_unit_range", config.map_to_unit_range);
  nh.getParam("normalize", config.normalize);
  nh.getParam("use_network_order", config.use_network_order);
  nh.getParam("network_uses_rgb_order", config.network_uses_rgb_order);

  return config;
}

#undef READ_REQUIRED

}  // namespace semantic_recolor
