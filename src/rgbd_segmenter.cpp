#include "semantic_recolor/rgbd_segmenter.h"
#include "semantic_recolor/utilities.h"

#include <NvOnnxParser.h>

#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace semantic_recolor {

TrtRgbdSegmenter::TrtRgbdSegmenter(const SegmentationConfig &config,
                                   const DepthConfig &depth_config)
    : TrtSegmenter(config), depth_config_(depth_config) {}

TrtRgbdSegmenter::~TrtRgbdSegmenter() {}

bool TrtRgbdSegmenter::init() {
  auto input_idx = engine_->getBindingIndex(depth_config_.input_name.c_str());
  if (input_idx == -1) {
    ROS_FATAL_STREAM("Failed to get index for input: " << depth_config_.input_name);
    return false;
  }
  ROS_INFO_STREAM("Input binding index: " << engine_->getBindingDataType(input_idx));

  if (engine_->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
    ROS_WARN_STREAM("Input type doesn't match expected: "
                    << engine_->getBindingDataType(input_idx)
                    << " != " << nvinfer1::DataType::kFLOAT);
  }

  nvinfer1::Dims4 input_dims{1, 1, config_.height, config_.width};
  context_->setBindingDimensions(input_idx, input_dims);
  depth_input_buffer_.reset(input_dims);

  return TrtSegmenter::init();
}

std::vector<void *> TrtRgbdSegmenter::getBindings() const {
  return {input_buffer_.memory.get(),
          depth_input_buffer_.memory.get(),
          output_buffer_.memory.get()};
}

bool TrtRgbdSegmenter::infer(const cv::Mat &img, const cv::Mat &depth_img) {
  // TODO(nathan) handle depth image copy
  fillNetworkImage(config_, img, nn_img_);
  auto error = cudaMemcpyAsync(depth_input_buffer_.memory.get(),
                               nn_img_.data,
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
