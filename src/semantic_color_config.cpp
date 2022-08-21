#include <semantic_recolor/semantic_color_config.h>

namespace semantic_recolor {

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
                                    size_t pixel_size) const {
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

void SemanticColorConfig::fillImage(const cv::Mat &classes, cv::Mat &output) const {
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
      fillColor(class_id, pixel);
    }
  }
}

}  // namespace semantic_recolor
