#pragma once
#include "semantic_recolor/segmenter.h"

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace semantic_recolor {

struct DemoConfig {
  DemoConfig(const ros::NodeHandle &nh) {
    if (!nh.getParam("input_file", input_file)) {
      ROS_FATAL("Missing input_file");
      throw std::runtime_error("missing param!");
    }

    if (!nh.getParam("output_file", output_file)) {
      ROS_FATAL("Missing output_file");
      throw std::runtime_error("missing param!");
    }

    nh.getParam("saturation", saturation);
    nh.getParam("luminance", luminance);
    nh.getParam("max_classes", max_classes);
    nh.getParam("num_timing_inferences", num_timing_inferences);
  }

  std::string input_file;
  std::string output_file;
  double saturation = 0.85;
  double luminance = 0.75;
  int max_classes = 150;
  int num_timing_inferences = 10;
};

inline size_t getFileSize(std::istream &to_check) {
  to_check.seekg(0, std::istream::end);
  size_t size = to_check.tellg();
  to_check.seekg(0, std::ifstream::beg);
  return size;
}

SegmentationConfig readSegmenterConfig(const ros::NodeHandle &nh);

void outputDemoImage(const DemoConfig &config, const cv::Mat &classes);

void showStatistics(const cv::Mat &classes);

void fillNetworkImage(const SegmentationConfig &config,
                      const cv::Mat &input,
                      cv::Mat &output);

}  // namespace semantic_recolor
