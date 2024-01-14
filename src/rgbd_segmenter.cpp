#include "semantic_recolor/rgbd_segmenter.h"

#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

RgbdSegmenter::RgbdSegmenter(const ModelConfig& config, const DepthConfig& depth_config)
    : SemanticSegmenter(config), depth_config_(depth_config) {}

RgbdSegmenter::~RgbdSegmenter() {}

bool RgbdSegmenter::createDepthBuffer() {
  auto input_idx = engine_->getBindingIndex(depth_config_.depth_input_name.c_str());
  if (input_idx == -1) {
    LOG_TO_LOGGER(kERROR,
                  "Failed to get index for input: " << depth_config_.depth_input_name);
    return false;
  }

  LOG_TO_LOGGER(kINFO,
                "Input binding index: " << engine_->getBindingDataType(input_idx));

  if (engine_->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
    LOG_TO_LOGGER(
        kWARNING,
        "Input type doesn't match expected: " << engine_->getBindingDataType(input_idx)
                                              << " != " << nvinfer1::DataType::kFLOAT);
  }

  context_->setBindingDimensions(input_idx, config_.getInputDims(1));
  depth_input_buffer_.reset(config_.getInputDims(1));

  nn_depth_img_ = cv::Mat(config_.getInputMatDims(1), CV_32FC1);
  return true;
}

bool RgbdSegmenter::init() {
  if (!createDepthBuffer()) {
    return false;
  }

  return TrtSegmenter::init();
}

std::vector<void*> RgbdSegmenter::getBindings() const {
  return {input_buffer_.memory.get(),
          depth_input_buffer_.memory.get(),
          output_buffer_.memory.get()};
}

void RgbdSegmenter::showStats(const cv::Mat& img,
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

bool RgbdSegmenter::infer(const cv::Mat& img, const cv::Mat& depth_img) {
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
