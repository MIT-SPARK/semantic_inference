#pragma once
#include "semantic_recolor/model_config.h"

#include <opencv2/opencv.hpp>

namespace semantic_recolor {

void fillNetworkImage(const ModelConfig &config, const cv::Mat &input, cv::Mat &output);

void fillNetworkDepthImage(const ModelConfig &cfg,
                           const cv::Mat &input,
                           cv::Mat &output);

}  // namespace semantic_recolor
