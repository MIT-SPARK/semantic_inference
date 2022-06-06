#include "semantic_recolor/utilities.h"

#include <opencv2/imgproc.hpp>

namespace semantic_recolor {

SegmentationConfig readSegmenterConfig(const ros::NodeHandle &nh) {
  SegmentationConfig config;

  if (!nh.getParam("model_file", config.model_file)) {
    ROS_FATAL("Missing model_file");
    throw std::runtime_error("missing param!");
  }

  nh.getParam("engine_file", config.engine_file);
  nh.getParam("width", config.width);
  nh.getParam("height", config.height);
  nh.getParam("input_name", config.input_name);
  nh.getParam("output_name", config.output_name);
  nh.getParam("mean", config.mean);
  nh.getParam("stddev", config.stddev);

  return config;
}

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

void createOverlayImage(SemanticColorConfig &config,
                        const cv::Mat &classes,
                        const cv::Mat &semantic,
                        const cv::Mat &original,
                        cv::Mat &output) {
  double alpha = 0.4;
  cv::addWeighted( semantic, alpha, original, (1.0-alpha), 0.0, output);
  return;
}

void outputDemoImage(const DemoConfig &config, const cv::Mat &classes) {
  cv::Mat new_image_hls(classes.rows, classes.cols, CV_32FC3);
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      float *pixel = new_image_hls.ptr<float>(r, c);

      const int remainder = classes.at<int32_t>(r, c) % config.max_classes;
      const double ratio =
          static_cast<double>(remainder) / static_cast<double>(config.max_classes);
      pixel[0] = ratio * 360.0;
      pixel[1] = config.luminance;
      pixel[2] = config.saturation;
    }
  }

  cv::Mat new_image;
  cv::cvtColor(new_image_hls, new_image, cv::COLOR_HLS2BGR);
  new_image *= 255.0;
  cv::imwrite(config.output_file, new_image);
}

void showStatistics(const cv::Mat &classes) {
  std::map<int32_t, size_t> counts;
  std::vector<int32_t> unique_classes;
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      int32_t class_id = classes.at<int32_t>(r, c);
      if (!counts.count(class_id)) {
        counts[class_id] = 0;
        unique_classes.push_back(class_id);
      }

      counts[class_id]++;
    }
  }

  double total = static_cast<double>(classes.rows * classes.cols);
  std::sort(unique_classes.begin(),
            unique_classes.end(),
            [&](const int32_t &lhs, const int32_t &rhs) {
              return counts[lhs] > counts[rhs];
            });

  std::stringstream ss;
  ss << " Class pixel percentages:" << std::endl;
  for (const int32_t id : unique_classes) {
    ss << "  - " << id << ": " << static_cast<double>(counts[id]) / total * 100.0 << "%"
       << std::endl;
  }

  ROS_INFO_STREAM(ss.str());
}

void fillNetworkImage(const SegmentationConfig &cfg,
                      const cv::Mat &input,
                      cv::Mat &output) {
  cv::Mat img;
  if (input.cols == cfg.width && input.rows == cfg.height) {
    img = input;
  } else {
    ROS_DEBUG_STREAM("Resizing from " << input.cols << " x " << input.rows << " x "
                                      << input.channels() << " to " << cfg.width
                                      << " x " << cfg.height << " x 3");
    cv::resize(input, img, cv::Size(cfg.width, cfg.height));
  }

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      const uint8_t *pixel = img.ptr<uint8_t>(row, col);
      output.at<float>(0, row, col) =
          (static_cast<float>(pixel[2]) / 255.0f - cfg.mean[0]) / cfg.stddev[0];
      output.at<float>(1, row, col) =
          (static_cast<float>(pixel[1]) / 255.0f - cfg.mean[1]) / cfg.stddev[1];
      output.at<float>(2, row, col) =
          (static_cast<float>(pixel[0]) / 255.0f - cfg.mean[2]) / cfg.stddev[2];
    }
  }
}

}  // namespace semantic_recolor
