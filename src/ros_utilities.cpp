#include "semantic_recolor/ros_utilities.h"

namespace semantic_recolor {

#define READ_REQUIRED(nh, config, name)                      \
  if (!nh.getParam(#name, config.name)) {                    \
    ROS_FATAL("Missing " #name " when parsing ModelConfig"); \
    throw std::runtime_error("missing param " #name "!");    \
  }                                                          \
  static_assert(true, "")

ModelConfig readModelConfig(const ros::NodeHandle& nh) {
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
  nh.getParam("set_builder_flags", config.set_builder_flags);

  return config;
}

DepthConfig readDepthModelConfig(const ros::NodeHandle& nh) {
  DepthConfig config;

  if (!nh.getParam("depth_input_name", config.depth_input_name)) {
    ROS_FATAL("Depth input name required!");
    throw std::runtime_error("depth_input_name required");
  }

  nh.getParam("depth_mean", config.depth_mean);
  nh.getParam("depth_stddev", config.depth_stddev);
  nh.getParam("normalize_depth", config.normalize_depth);
  nh.getParam("mask_predictions", config.mask_predictions);
  nh.getParam("min_depth", config.min_depth);
  nh.getParam("max_depth", config.max_depth);
  return config;
}

#undef READ_REQUIRED

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  auto iter = v.begin();
  while (iter != v.end()) {
    out << *iter;
    ++iter;
    if (iter != v.end()) {
      out << ", ";
    }
  }
  out << "]";
  return out;
}

#define SHOW_PARAM(config, name)                               \
  {                                                            \
    std::stringstream ss;                                      \
    ss << std::boolalpha << (" - " #name ": ") << config.name; \
    ROS_INFO_STREAM(ss.str());                                 \
  }                                                            \
  static_assert(true, "")

void showModelConfig(const ModelConfig& config) {
  ROS_INFO_STREAM("ModelConfig:");
  SHOW_PARAM(config, model_file);
  SHOW_PARAM(config, engine_file);
  SHOW_PARAM(config, width);
  SHOW_PARAM(config, height);
  SHOW_PARAM(config, input_name);
  SHOW_PARAM(config, output_name);
  SHOW_PARAM(config, mean);
  SHOW_PARAM(config, stddev);
  SHOW_PARAM(config, map_to_unit_range);
  SHOW_PARAM(config, normalize);
  SHOW_PARAM(config, use_network_order);
  SHOW_PARAM(config, network_uses_rgb_order);
  ROS_INFO_STREAM("rgb dimensions: " << config.getInputMatDims(3));
  ROS_INFO_STREAM("depth dimensions: " << config.getInputMatDims(1));
}

void showDepthModelConfig(const DepthConfig& config) {
  ROS_INFO_STREAM("ModelConfig:");
  SHOW_PARAM(config, depth_input_name);
  SHOW_PARAM(config, depth_mean);
  SHOW_PARAM(config, depth_stddev);
  SHOW_PARAM(config, normalize_depth);
  SHOW_PARAM(config, mask_predictions);
  SHOW_PARAM(config, min_depth);
  SHOW_PARAM(config, max_depth);
}

#undef SHOW_PARAM

}  // namespace semantic_recolor
