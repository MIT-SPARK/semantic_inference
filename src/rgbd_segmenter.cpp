#include "semantic_recolor/rgbd_segmenter.h"
#include "semantic_recolor/utilities.h"

#include <NvOnnxParser.h>

#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace semantic_recolor {

TrtRgbdSegmenter::TrtRgbdSegmenter(const ModelConfig &config,
                                   const std::string &depth_input_name)
    : TrtSegmenter(config), depth_input_name_(depth_input_name) {}

TrtRgbdSegmenter::~TrtRgbdSegmenter() {}

bool TrtRgbdSegmenter::createDepthBuffer() {
  auto input_idx = engine_->getBindingIndex(depth_input_name_.c_str());
  if (input_idx == -1) {
    ROS_FATAL_STREAM("Failed to get index for input: " << depth_input_name_);
    return false;
  }

  ROS_INFO_STREAM("Input binding index: " << engine_->getBindingDataType(input_idx));

  if (engine_->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
    ROS_WARN_STREAM("Input type doesn't match expected: "
                    << engine_->getBindingDataType(input_idx)
                    << " != " << nvinfer1::DataType::kFLOAT);
  }

  context_->setBindingDimensions(input_idx, config_.getInputDims(1));
  input_buffer_.reset(config_.getInputDims(1));

  nn_depth_img_ = cv::Mat(config_.getInputMatDims(1), CV_32FC1);
  return true;
}

bool TrtRgbdSegmenter::init() {
  if (!createDepthBuffer()) {
    return false;
  }

  return TrtSegmenter::init();
}

std::vector<void *> TrtRgbdSegmenter::getBindings() const {
  return {input_buffer_.memory.get(),
          depth_input_buffer_.memory.get(),
          output_buffer_.memory.get()};
}

bool TrtRgbdSegmenter::infer(const cv::Mat &img, const cv::Mat &depth_img) {
  fillNetworkDepthImage(config_, depth_img, nn_depth_img_);
  auto error = cudaMemcpyAsync(depth_input_buffer_.memory.get(),
                               nn_depth_img_.data,
                               depth_input_buffer_.size,
                               cudaMemcpyHostToDevice,
                               stream_);
  if (error != cudaSuccess) {
    ROS_FATAL_STREAM(
        "copying depth image to gpu failed: " << cudaGetErrorString(error));
    return false;
  }

  return TrtSegmenter::infer(img);
}

}  // namespace semantic_recolor
