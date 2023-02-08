#include "semantic_recolor/image_utilities.h"

#include <opencv2/imgproc.hpp>

namespace semantic_recolor {

void fillNetworkImage(const ModelConfig& cfg, const cv::Mat& input, cv::Mat& output) {
  cv::Mat img;
  if (input.cols == cfg.width && input.rows == cfg.height) {
    img = input;
  } else {
    cv::resize(input, img, cv::Size(cfg.width, cfg.height));
  }

  ModelConfig::ImageAddress input_addr;
  cfg.fillInputAddress(input_addr);

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      const uint8_t* pixel = img.ptr<uint8_t>(row, col);
      if (cfg.use_network_order) {
        output.at<float>(0, row, col) = cfg.getValue(pixel[input_addr[0]], 0);
        output.at<float>(1, row, col) = cfg.getValue(pixel[input_addr[1]], 1);
        output.at<float>(2, row, col) = cfg.getValue(pixel[input_addr[2]], 2);
      } else {
        output.at<float>(row, col, 0) = cfg.getValue(pixel[input_addr[0]], 0);
        output.at<float>(row, col, 1) = cfg.getValue(pixel[input_addr[1]], 1);
        output.at<float>(row, col, 2) = cfg.getValue(pixel[input_addr[2]], 2);
      }
    }
  }
}

void fillNetworkDepthImage(const ModelConfig& cfg,
                           const DepthConfig& depth_config,
                           const cv::Mat& input,
                           cv::Mat& output) {
  const bool size_ok = input.cols == cfg.width && input.rows == cfg.height;
  if (size_ok && !cfg.use_network_order) {
    output = input;
    return;
  }

  if (!size_ok && !cfg.use_network_order) {
    cv::resize(input, output, cv::Size(cfg.width, cfg.height), 0, 0, cv::INTER_NEAREST);
    return;
  }

  cv::Mat img;
  if (size_ok) {
    img = input;
  } else {
    cv::resize(input, img, cv::Size(cfg.width, cfg.height), 0, 0, cv::INTER_NEAREST);
  }

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      output.at<float>(0, row, col) = depth_config.getValue(img.at<float>(row, col));
    }
  }
}

}  // namespace semantic_recolor
