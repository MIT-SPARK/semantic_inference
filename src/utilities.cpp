#include "semantic_recolor_nodelet/utilities.h"

#include <opencv2/imgproc.hpp>

namespace semantic_recolor {

TestConfig readTestConfig(const ros::NodeHandle &nh) {
  TestConfig config;

  if (!nh.getParam("model_path", config.model_path)) {
    ROS_FATAL("Missing model_path");
    throw std::runtime_error("missing param!");
  }

  nh.getParam("image_width", config.width);
  nh.getParam("image_height", config.height);
  nh.getParam("saturation", config.saturation);
  nh.getParam("luminance", config.luminance);

  if (!nh.getParam("input_file", config.input_file)) {
    ROS_FATAL("Missing input_file");
    throw std::runtime_error("missing param!");
  }

  if (!nh.getParam("output_file", config.output_file)) {
    ROS_FATAL("Missing output_file");
    throw std::runtime_error("missing param!");
  }

  return config;
}

void outputDebugImg(const TestConfig &config, const cv::Mat &classes) {
  double max_class = 0.0;
  double min_class = 0.0;
  cv::minMaxLoc(classes, &min_class, &max_class);
  ROS_INFO_STREAM("Min class: " << min_class << " Max class: " << max_class);

  if (max_class - min_class == 0.0) {
    ROS_WARN_STREAM("Min and max class are the same: " << max_class);
    return;
  }

  const double class_diff = max_class - min_class;

  cv::Mat new_image_hls(classes.rows, classes.cols, CV_32FC3);
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      float *pixel = new_image_hls.ptr<float>(r, c);
      double ratio =
          static_cast<double>(classes.at<int32_t>(r, c) - min_class) / class_diff;
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

cv::Mat getImage(const TestConfig &config) {
  cv::Mat img = cv::imread(config.input_file);
  if (img.empty()) {
    ROS_FATAL_STREAM("Image not found: " << config.input_file);
    throw std::runtime_error("bad input file");
  }

  cv::Mat infer_img;
  cv::resize(img, infer_img, cv::Size(config.width, config.height));

  // TODO(nathan) move to config
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stddev{0.229f, 0.224f, 0.225f};

  std::vector<int> dims{3, config.height, config.width};
  cv::Mat nn_img(dims, CV_32FC1);

  for (int row = 0; row < infer_img.rows; ++row) {
    for (int col = 0; col < infer_img.cols; ++col) {
      const uint8_t *pixel = infer_img.ptr<uint8_t>(row, col);
      nn_img.at<float>(0, row, col) = (static_cast<float>(pixel[2]) / 255.0 - mean[0]) / stddev[0];
      nn_img.at<float>(1, row, col) = (static_cast<float>(pixel[1]) / 255.0 - mean[1]) / stddev[1];
      nn_img.at<float>(2, row, col) = (static_cast<float>(pixel[0]) / 255.0 - mean[2]) / stddev[2];
    }
  }

  //double min_value = 0.0;
  //double max_value = 0.0;
  //cv::minMaxLoc(nn_img, &min_value, &max_value);
  //ROS_INFO_STREAM("Image values: min=" << min_value << ", max=" << max_value);

  return nn_img;
}

}  // namespace semantic_recolor
