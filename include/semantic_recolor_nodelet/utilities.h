#pragma once
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

namespace semantic_recolor {

inline size_t getFileSize(std::istream &to_check) {
  to_check.seekg(0, std::istream::end);
  size_t size = to_check.tellg();
  to_check.seekg(0, std::ifstream::beg);
  return size;
}

template <typename T>
std::string displayVector(const std::vector<T> &vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    ss << vec[i] << ", ";
  }
  if (vec.size()) {
    ss << vec.back();
  }
  ss << "]";
  return ss.str();
}

struct TestConfig {
  std::string model_path;
  int width = 640;
  int height = 360;
  double saturation = 0.85;
  double luminance = 0.75;
  std::string input_file;
  std::string output_file;
};

TestConfig readTestConfig(const ros::NodeHandle &nh);

void outputDebugImg(const TestConfig &config, const cv::Mat &classes);

void showStatistics(const cv::Mat &classes);

cv::Mat getImage(const TestConfig &config);

}  // namespace semantic_recolor
