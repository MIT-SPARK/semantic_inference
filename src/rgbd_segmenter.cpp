#include "semantic_recolor/rgbd_segmenter.h"

#include <NvOnnxParser.h>
#include <ros/ros.h>

#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

TrtRgbdSegmenter::TrtRgbdSegmenter(const ModelConfig& config,
                                   const DepthConfig& depth_config)
    : TrtSegmenter(config), depth_config_(depth_config) {}

TrtRgbdSegmenter::~TrtRgbdSegmenter() {}

bool TrtRgbdSegmenter::createDepthBuffer() {
  // TODO(nathan) think about querying if name exists
  const auto tensor_name = depth_config_.depth_input_name.c_str();
  const auto dtype = engine_->getTensorDataType(tensor_name);

  LOG_TO_LOGGER(kINFO, "Input binding type: " << dtype);
  if (dtype != nvinfer1::DataType::kFLOAT) {
    LOG_TO_LOGGER(kERROR,
                  "Input type doesn't match expected: " << dtype << " != "
                                                        << nvinfer1::DataType::kFLOAT);
    return false;
  }

  depth_input_buffer_.reset(config_.getInputDims(1));
  nn_depth_img_ = cv::Mat(config_.getInputMatDims(1), CV_32FC1);

  context_->setInputTensorAddress(tensor_name, depth_input_buffer_.memory.get());
  return true;
}

bool TrtRgbdSegmenter::init() {
  if (!createDepthBuffer()) {
    return false;
  }

  return TrtSegmenter::init();
}

void TrtRgbdSegmenter::showStats(const cv::Mat& img,
                                 int channel,
                                 const std::string& name) const {
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  float mean = 0.0f;

  const float total = config_.height * config_.width;
  for (int r = 0; r < config_.height; ++r) {
    for (int c = 0; c < config_.width; ++c) {
      const float val = img.at<float>(channel, r, c);
      mean += val / total;

      if (val < min) {
        min = val;
      }

      if (val > max) {
        max = val;
      }
    }
  }

  float stddev = 0.0f;
  for (int r = 0; r < config_.height; ++r) {
    for (int c = 0; c < config_.width; ++c) {
      const float val = img.at<float>(channel, r, c);
      stddev += (mean - val) * (mean - val) / total;
    }
  }
  stddev = std::sqrt(stddev);

  std::cout << std::setprecision(6) << name << ": min=" << min << ", max=" << max
            << ", dist=" << mean << " +/- " << stddev << std::endl;
}

bool TrtRgbdSegmenter::infer(const cv::Mat& img, const cv::Mat& depth_img) {
  fillNetworkDepthImage(config_, depth_config_, depth_img, nn_depth_img_);
  auto error = cudaMemcpyAsync(depth_input_buffer_.memory.get(),
                               nn_depth_img_.data,
                               depth_input_buffer_.size,
                               cudaMemcpyHostToDevice,
                               stream_);
  if (error != cudaSuccess) {
    LOG_TO_LOGGER(kERROR,
                  "copying depth image to gpu failed: " << cudaGetErrorString(error));
    return false;
  }

  auto result = TrtSegmenter::infer(img);
  if (config_.show_stats) {
    showStats(nn_img_, 0, "red  ");
    showStats(nn_img_, 1, "green");
    showStats(nn_img_, 2, "blue ");
    showStats(nn_depth_img_, 0, "depth");
  }

  if (depth_config_.mask_predictions) {
    // TODO(nathan) rethink
    cv::Mat resized_depth;
    cv::resize(depth_img,
               resized_depth,
               cv::Size(config_.width, config_.height),
               0,
               0,
               cv::INTER_NEAREST);

    cv::Mat mask;
    cv::inRange(resized_depth, depth_config_.min_depth, depth_config_.max_depth, mask);

    cv::Mat masked_classes;
    cv::bitwise_or(classes_, classes_, masked_classes, mask);
    classes_ = masked_classes;
  }

  return result;
}

}  // namespace semantic_recolor
