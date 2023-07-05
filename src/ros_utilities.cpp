#include "semantic_recolor/ros_utilities.h"

namespace semantic_recolor {

#define READ_PARAM(nh, config, name, required)             \
  if (!nh.getParam(#name, config.name) && required) {      \
    ROS_FATAL("Missing " #name " when parsing parameter"); \
    throw std::runtime_error("missing param " #name "!");  \
  }                                                        \
  static_assert(true, "")

ModelConfig readModelConfig(const ros::NodeHandle& nh) {
  ModelConfig config;
  READ_PARAM(nh, config, width, true);
  READ_PARAM(nh, config, height, true);
  READ_PARAM(nh, config, input_name, true);
  READ_PARAM(nh, config, output_name, true);
  READ_PARAM(nh, config, mean, false);
  READ_PARAM(nh, config, stddev, false);
  READ_PARAM(nh, config, map_to_unit_range, false);
  READ_PARAM(nh, config, normalize, false);
  READ_PARAM(nh, config, use_network_order, false);
  READ_PARAM(nh, config, network_uses_rgb_order, false);
  return config;
}

DepthConfig readDepthModelConfig(const ros::NodeHandle& nh) {
  DepthConfig config;
  READ_PARAM(nh, config, depth_input_name, true);
  READ_PARAM(nh, config, depth_mean, false);
  READ_PARAM(nh, config, depth_stddev, false);
  READ_PARAM(nh, config, normalize_depth, false);
  READ_PARAM(nh, config, mask_predictions, false);
  READ_PARAM(nh, config, min_depth, false);
  READ_PARAM(nh, config, max_depth, false);
  return config;
}

#undef READ_PARAM

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
  ROS_INFO_STREAM("rgb dimensions: " << config.getInputDims(3));
  ROS_INFO_STREAM("depth dimensions: " << config.getInputDims(1));
}

void showDepthModelConfig(const DepthConfig& config) {
  ROS_INFO_STREAM("DepthConfig:");
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
