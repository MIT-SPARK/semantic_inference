#include <ros/ros.h>
#include <semantic_recolor/semantic_color_config.h>

namespace semantic_recolor {

SemanticColorConfig::SemanticColorConfig() : initialized_(false) {}

std::array<uint8_t, 3> convertToRGB8(const std::vector<double>& color) {
  if (color.size() != 3) {
    ROS_FATAL_STREAM("invalid color: size " << color.size() << " != 3");
    throw std::runtime_error("invalid semantic color config param");
  }

  return {static_cast<uint8_t>(std::floor(color.at(0) * 255)),
          static_cast<uint8_t>(std::floor(color.at(1) * 255)),
          static_cast<uint8_t>(std::floor(color.at(2) * 255))};
}

SemanticColorConfig::SemanticColorConfig(const ros::NodeHandle& nh)
    : initialized_(false), default_id_(-1) {
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
  for (auto&& [class_id, class_info] : classes) {
    if (class_id < 0 || class_id > std::numeric_limits<int16_t>::max()) {
      ROS_FATAL_STREAM("found invalid class id: " << class_id);
      throw std::runtime_error("invalid class id");
    }

    const auto actual_color = convertToRGB8(class_info.color);
    for (const auto& label : class_info.labels) {
      if (label < 0 || label > std::numeric_limits<int16_t>::max()) {
        ROS_FATAL_STREAM("found invalid label: " << label);
        throw std::runtime_error("invalid label");
      }

      color_map_[label] = actual_color;
      label_remapping_[label] = class_id;
    }
  }

  default_color_ = convertToRGB8(default_color);
  initialized_ = true;
}

void SemanticColorConfig::fillColor(int16_t class_id,
                                    uint8_t* pixel,
                                    size_t pixel_size) const {
  const auto iter = color_map_.find(class_id);
  if (iter != color_map_.end()) {
    std::memcpy(pixel, iter->second.data(), pixel_size);
    return;
  }

  if (!seen_unknown_labels_.count(class_id)) {
    ROS_ERROR_STREAM("Encountered unhandled class id: " << class_id);
    seen_unknown_labels_.insert(class_id);
  }

  std::memcpy(pixel, default_color_.data(), pixel_size);
}

int16_t SemanticColorConfig::getRemappedLabel(int16_t class_id) const {
  const auto iter = label_remapping_.find(class_id);
  if (iter != label_remapping_.end()) {
    return iter->second;
  }

  if (!seen_unknown_labels_.count(class_id)) {
    ROS_ERROR_STREAM("Encountered unhandled class id: " << class_id);
    seen_unknown_labels_.insert(class_id);
  }

  return default_id_;
}

void SemanticColorConfig::relabelImage(const cv::Mat& classes, cv::Mat& output) const {
  if (!initialized_) {
    ROS_FATAL("SemanticColorConfig not initialized");
    throw std::runtime_error("uninitialized color config");
  }

  if (output.type() != CV_16S) {
    return;
  }

  // opencv doesn't allow resizing of 32S images...
  cv::Mat resized_classes;
  classes.convertTo(resized_classes, CV_16S);
  if (classes.rows != output.rows || classes.cols != output.cols) {
    // interpolating class labels doesn't make sense so use NEAREST
    cv::resize(resized_classes,
               resized_classes,
               cv::Size(output.cols, output.rows),
               0.0f,
               0.0f,
               cv::INTER_NEAREST);
  }

  for (int r = 0; r < resized_classes.rows; ++r) {
    for (int c = 0; c < resized_classes.cols; ++c) {
      int16_t* pixel = output.ptr<int16_t>(r, c);
      const auto class_id = resized_classes.at<int16_t>(r, c);
      *pixel = getRemappedLabel(class_id);
    }
  }
}

void SemanticColorConfig::recolorImage(const cv::Mat& classes, cv::Mat& output) const {
  if (!initialized_) {
    ROS_FATAL("SemanticColorConfig not initialized");
    throw std::runtime_error("uninitialized color config");
  }

  if (output.type() != CV_8UC3) {
    return;
  }

  // opencv doesn't allow resizing of 32S images...
  cv::Mat resized_classes;
  classes.convertTo(resized_classes, CV_16S);
  if (classes.rows != output.rows || classes.cols != output.cols) {
    // interpolating class labels doesn't make sense so use NEAREST
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
      const auto class_id = resized_classes.at<int16_t>(r, c);
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
