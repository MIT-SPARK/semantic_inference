#include <ros/ros.h>
#include <semantic_recolor/semantic_color_config.h>

namespace semantic_recolor {

SemanticColorConfig::SemanticColorConfig() : initialized_(false) {}

std::vector<uint8_t> convertToRGB8(const std::vector<double>& color) {
  return {static_cast<uint8_t>(std::floor(color[0] * 255)),
          static_cast<uint8_t>(std::floor(color[1] * 255)),
          static_cast<uint8_t>(std::floor(color[2] * 255))};
}

SemanticColorConfig::SemanticColorConfig(const ros::NodeHandle& nh) {
  std::vector<int> class_ids;
  if (!nh.getParam("classes", class_ids)) {
    ROS_FATAL("failed to find classes parameter");
    throw std::runtime_error("missing semantic color config param");
  }

  std::map<int, ColorLabelPair> classes;
  for (const auto& class_id : class_ids) {
    const std::string param_name = "class_info/" + std::to_string(class_id);

    ColorLabelPair class_info;
    if (!nh.getParam(param_name + "/color", class_info.color)) {
      ROS_FATAL_STREAM("failed to find color for " << param_name);
      throw std::runtime_error("missing semantic color config param");
    }

    std::vector<int> labels;
    if (!nh.getParam(param_name + "/labels", class_info.labels)) {
      ROS_FATAL_STREAM("failed to find labels for " << param_name);
      throw std::runtime_error("missing semantic color config param");
    }

    classes[class_id] = class_info;
  }

  std::vector<double> default_color;
  if (!nh.getParam("default_color", default_color)) {
    default_color = std::vector<double>(3, 0.0);
  }

  initialize(classes, default_color);
}

void SemanticColorConfig::initialize(const std::map<int, ColorLabelPair>& classes,
                                     const std::vector<double>& default_color) {
  for (const auto& id_info_pair : classes) {
    const auto& class_info = id_info_pair.second;
    if (class_info.color.size() != 3) {
      ROS_FATAL_STREAM("invalid color: num elements " << class_info.color.size()
                                                      << " != 3");
      throw std::runtime_error("invalid semantic color config param");
    }

    std::vector<uint8_t> actual_color = convertToRGB8(class_info.color);
    for (const auto& label : class_info.labels) {
      color_map_[label] = actual_color;
    }
  }

  if (default_color.size() != 3) {
    ROS_FATAL_STREAM("invalid color: num elements " << default_color.size() << " != 3");
    throw std::runtime_error("invalid semantic color config param");
  }

  default_color_ = convertToRGB8(default_color);
  initialized_ = true;
}

void SemanticColorConfig::fillColor(int32_t class_id,
                                    uint8_t* pixel,
                                    size_t pixel_size) const {
  if (!initialized_) {
    ROS_FATAL("SemanticColorConfig not initialized");
    throw std::runtime_error("uninitialized color config");
  }

  if (color_map_.count(class_id)) {
    const auto& color = color_map_.at(class_id);
    std::memcpy(pixel, color.data(), pixel_size);
  } else {
    if (!seen_unknown_labels_.count(class_id)) {
      ROS_ERROR_STREAM("Encountered unhandled class id: " << class_id);
      seen_unknown_labels_.insert(class_id);
    }
    std::memcpy(pixel, default_color_.data(), pixel_size);
  }
}

void SemanticColorConfig::fillImage(const cv::Mat& classes, cv::Mat& output) const {
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
      uint8_t* pixel = output.ptr<uint8_t>(r, c);
      const auto class_id = resized_classes.at<uint8_t>(r, c);
      fillColor(class_id, pixel);
    }
  }
}

void SemanticColorConfig::show(std::ostream& out) const {
  out << "SemanticColorConfig:" << std::endl;
  for (const auto id_color_pair : color_map_) {
    out << "  - " << id_color_pair.first
        << " -> [r=" << static_cast<int>(id_color_pair.second[0])
        << ", g=" << static_cast<int>(id_color_pair.second[1])
        << ", b=" << static_cast<int>(id_color_pair.second[2]) << "]" << std::endl;
  }
}

}  // namespace semantic_recolor
