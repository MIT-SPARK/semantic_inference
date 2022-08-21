#pragma once
#include "semantic_recolor/model_config.h"

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace semantic_recolor {

class SemanticColorConfig {
 public:
  SemanticColorConfig();

  explicit SemanticColorConfig(const ros::NodeHandle &nh);

  void fillColor(int32_t class_id, uint8_t *pixel, size_t pixel_size = 3);

 private:
  bool initialized_;
  std::map<int32_t, std::vector<uint8_t>> color_map_;
  std::vector<uint8_t> default_color_;
  std::set<int32_t> seen_unknown_labels_;
};

ModelConfig readModelConfig(const ros::NodeHandle &nh);

void fillSemanticImage(SemanticColorConfig &config,
                       const cv::Mat &classes,
                       cv::Mat &output);

void fillNetworkImage(const ModelConfig &config, const cv::Mat &input, cv::Mat &output);

void fillNetworkDepthImage(const ModelConfig &cfg,
                           const cv::Mat &input,
                           cv::Mat &output);

}  // namespace semantic_recolor
