#pragma once
#include <opencv2/opencv.hpp>

#include "semantic_recolor/model_config.h"

namespace semantic_recolor {

void fillNetworkImage(const ModelConfig& config, const cv::Mat& input, cv::Mat& output);

void fillNetworkDepthImage(const ModelConfig& cfg,
                           const DepthConfig& depth_config,
                           const cv::Mat& input,
                           cv::Mat& output);

}  // namespace semantic_recolor
