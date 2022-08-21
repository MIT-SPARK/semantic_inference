#include "semantic_recolor/utilities.h"

#include <opencv2/imgproc.hpp>

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

SemanticColorConfig::SemanticColorConfig() : initialized_(false) {}

std::vector<uint8_t> convertToRGB8(const std::vector<double> &color) {
  return {static_cast<uint8_t>(std::floor(color[0] * 255)),
          static_cast<uint8_t>(std::floor(color[1] * 255)),
          static_cast<uint8_t>(std::floor(color[2] * 255))};
}

SemanticColorConfig::SemanticColorConfig(const ros::NodeHandle &nh) {
  std::vector<int> classes;
  if (!nh.getParam("classes", classes)) {
    ROS_FATAL("failed to find classes parameter");
    throw std::runtime_error("missing semantic color config param");
  }

  for (const auto &class_id : classes) {
    const std::string param_name = "class_info/" + std::to_string(class_id);

    std::vector<double> color;
    if (!nh.getParam(param_name + "/color", color)) {
      ROS_FATAL_STREAM("failed to find color for " << param_name);
      throw std::runtime_error("missing semantic color config param");
    }

    if (color.size() != 3) {
      ROS_FATAL_STREAM("invalid color: num elements " << color.size() << " != 3");
      throw std::runtime_error("invalid semantic color config param");
    }

    std::vector<int> labels;
    if (!nh.getParam(param_name + "/labels", labels)) {
      ROS_FATAL_STREAM("failed to find labels for " << param_name);
      throw std::runtime_error("missing semantic color config param");
    }

    std::vector<uint8_t> actual_color = convertToRGB8(color);
    for (const auto &label : labels) {
      color_map_[label] = actual_color;
    }
  }

  std::vector<double> default_color;
  if (!nh.getParam("default_color", default_color)) {
    default_color_ = std::vector<uint8_t>(3, 0);
    return;
  }

  if (default_color.size() != 3) {
    ROS_FATAL_STREAM("invalid color: num elements " << default_color.size() << " != 3");
    throw std::runtime_error("invalid semantic color config param");
  }

  default_color_ = convertToRGB8(default_color);
  initialized_ = true;
}

void SemanticColorConfig::fillColor(int32_t class_id,
                                    uint8_t *pixel,
                                    size_t pixel_size) {
  if (!initialized_) {
    ROS_FATAL("SemanticColorConfig not initialized");
    throw std::runtime_error("uninitialized color config");
  }

  if (color_map_.count(class_id)) {
    const auto &color = color_map_.at(class_id);
    std::memcpy(pixel, color.data(), pixel_size);
  } else {
    if (!seen_unknown_labels_.count(class_id)) {
      ROS_ERROR_STREAM("Encountered unhandled class id: " << class_id);
      seen_unknown_labels_.insert(class_id);
    }
    std::memcpy(pixel, default_color_.data(), pixel_size);
  }
}

void fillSemanticImage(SemanticColorConfig &config,
                       const cv::Mat &classes,
                       cv::Mat &output) {
  cv::Mat resized_classes;
  classes.convertTo(resized_classes, CV_8UC1);
  if (classes.rows != output.rows || classes.cols != output.cols) {
    // interpolating class labels doesn't make sense
    cv::resize(resized_classes,
               resized_classes,
               cv::Size(output.cols, output.rows),
               0.0f,
               0.0f,
               cv::INTER_NEAREST);
  }

  for (int r = 0; r < resized_classes.rows; ++r) {
    for (int c = 0; c < resized_classes.cols; ++c) {
      uint8_t *pixel = output.ptr<uint8_t>(r, c);
      const auto class_id = resized_classes.at<uint8_t>(r, c);
      config.fillColor(class_id, pixel);
    }
  }
}

void fillNetworkImage(const ModelConfig &cfg, const cv::Mat &input, cv::Mat &output) {
  cv::Mat img;
  if (input.cols == cfg.width && input.rows == cfg.height) {
    img = input;
  } else {
    ROS_DEBUG_STREAM("Resizing from " << input.cols << " x " << input.rows << " x "
                                      << input.channels() << " to " << cfg.width
                                      << " x " << cfg.height << " x 3");
    cv::resize(input, img, cv::Size(cfg.width, cfg.height));
  }

  ModelConfig::ImageAddress input_addr;
  cfg.fillInputAddress(input_addr);

  // TODO(nathan) check
  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      const uint8_t *pixel = img.ptr<uint8_t>(row, col);
      if (cfg.use_network_order) {
        output.at<float>(0, row, col) = cfg.getValue(pixel[input_addr[0]], 0);
        output.at<float>(1, row, col) = cfg.getValue(pixel[input_addr[1]], 1);
        output.at<float>(2, row, col) = cfg.getValue(pixel[input_addr[2]], 2);
      } else {
        output.at<float>(row, col, 0) = cfg.getValue(pixel[input_addr[0]], 0);
        output.at<float>(row, col, 1) = cfg.getValue(pixel[input_addr[1]], 1);
        output.at<float>(row, col, 2) = cfg.getValue(pixel[input_addr[2]], 2);
      }
    }
  }
}

void fillNetworkDepthImage(const ModelConfig &cfg,
                           const cv::Mat &input,
                           cv::Mat &output) {
  const bool size_ok = input.cols == cfg.width && input.rows == cfg.height;
  if (size_ok && !cfg.use_network_order) {
    output = input;
    return;
  }

  if (!size_ok && !cfg.use_network_order) {
    cv::resize(input, output, cv::Size(cfg.width, cfg.height), 0, 0, cv::INTER_NEAREST);
    return;
  }

  cv::Mat img;
  if (size_ok) {
    img = input;
  } else {
    ROS_DEBUG_STREAM("Resizing from " << input.cols << " x " << input.rows << " to "
                                      << cfg.width << " x " << cfg.height);
    cv::resize(input, img, cv::Size(cfg.width, cfg.height), 0, 0, cv::INTER_NEAREST);
  }

  // TODO(nathan) check
  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      output.at<float>(0, row, col) = input.at<float>(row, col);
    }
  }
}

}  // namespace semantic_recolor
