#include "semantic_recolor/yaml_utilities.h"

namespace semantic_recolor {

#define READ_PARAM(node, config, name, required)                           \
  if (!node[#name] && required) {                                          \
    std::cerr << "Missing " #name " when parsing config" << std::endl;     \
    throw std::runtime_error("missing param " #name "!");                  \
  }                                                                        \
  if (node[#name]) {                                                       \
    try {                                                                  \
      config.name = node[#name].as<decltype(config.name)>();               \
    } catch (const std::exception& e) {                                    \
      std::cerr << "failed to parse " #name ": " << e.what() << std::endl; \
    }                                                                      \
  }                                                                        \
  static_assert(true, "")

ModelConfig readModelConfigFromYaml(const YAML::Node& node) {
  ModelConfig config;
  READ_PARAM(node, config, width, true);
  READ_PARAM(node, config, height, true);
  READ_PARAM(node, config, input_name, true);
  READ_PARAM(node, config, output_name, true);
  READ_PARAM(node, config, mean, false);
  READ_PARAM(node, config, stddev, false);
  READ_PARAM(node, config, map_to_unit_range, false);
  READ_PARAM(node, config, normalize, false);
  READ_PARAM(node, config, use_network_order, false);
  READ_PARAM(node, config, network_uses_rgb_order, false);
  return config;
}

DepthConfig readDepthModelConfigFromYaml(const YAML::Node& node) {
  DepthConfig config;
  READ_PARAM(node, config, depth_input_name, false);
  READ_PARAM(node, config, depth_mean, false);
  READ_PARAM(node, config, depth_stddev, false);
  READ_PARAM(node, config, normalize_depth, false);
  READ_PARAM(node, config, mask_predictions, false);
  READ_PARAM(node, config, min_depth, false);
  READ_PARAM(node, config, max_depth, false);
  return config;
}

SemanticColorConfig readSemanticColorConfigFromYaml(const YAML::Node& node) {
  SemanticColorConfig config;

  if (!node["classes"]) {
    std::cerr << "failed to find classes parameter" << std::endl;
    throw std::runtime_error("missing semantic color config param");
  }

  auto class_ids = node["classes"].as<std::vector<int>>();

  std::map<int, ColorLabelPair> classes;
  for (const auto& class_id : class_ids) {
    const std::string param_name = "class_info/" + std::to_string(class_id);
    const std::string color_name = param_name + "/color";
    const std::string label_name = param_name + "/labels";

    ColorLabelPair class_info;

    if (!node[color_name]) {
      std::cerr << "failed to find color @ " << color_name << std::endl;
      throw std::runtime_error("missing semantic color config param");
    }

    class_info.color = node[color_name].as<std::vector<double>>();

    if (!node[label_name]) {
      std::cerr << "failed to find labels @ " << label_name << std::endl;
      throw std::runtime_error("missing semantic color config param");
    }

    class_info.labels = node[label_name].as<std::vector<int>>();
    classes[class_id] = class_info;
  }

  std::vector<double> default_color(3, 0.0);
  if (node["default_color"]) {
    default_color = node["default_color"].as<std::vector<double>>();
  }

  config.initialize(classes, default_color);
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
    std::cout << ss.str() << std::endl;                        \
  }                                                            \
  static_assert(true, "")

void printModelConfig(const ModelConfig& config) {
  std::cout << "ModelConfig:" << std::endl;
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
  std::cout << "rgb dimensions: " << config.getInputDims(3) << std::endl;
  std::cout << "depth dimensions: " << config.getInputDims(1) << std::endl;
}

void printDepthModelConfig(const DepthConfig& config) {
  std::cout << "DepthConfig:" << std::endl;
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
