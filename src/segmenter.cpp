#include "semantic_recolor/segmenter.h"
#include "semantic_recolor/utilities.h"

#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <memory>
#include <numeric>

namespace semantic_recolor {

TrtSegmenter::TrtSegmenter(const ModelConfig &config)
    : config_(config), logger_(Severity::kINFO), initialized_(false) {
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_ = deserializeEngine(*runtime_, config_.engine_file);
  if (!engine_) {
    ROS_WARN("TRT engine not found! Rebuilding.");
    engine_ = buildEngineFromOnnx(
        *runtime_, logger_, config_.model_file, config_.engine_file);
    ROS_INFO_STREAM("Finished building engine");
  }

  if (!engine_) {
    ROS_FATAL_STREAM("Building engine from onnx failed!");
    throw std::runtime_error("failed to load or build engine");
  }

  ROS_INFO_STREAM("Loaded TRT engine");

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    ROS_FATAL_STREAM("Failed to create execution context");
    throw std::runtime_error("failed to set up trt context");
  }

  ROS_INFO_STREAM("TRT execution context started");
}

TrtSegmenter::~TrtSegmenter() {
  if (initialized_) {
    cudaStreamDestroy(stream_);
  }
}

bool TrtSegmenter::createInputBuffer() {
  auto input_idx = engine_->getBindingIndex(config_.input_name.c_str());
  if (input_idx == -1) {
    ROS_FATAL_STREAM("Failed to get index for input: " << config_.input_name);
    return false;
  }

  ROS_INFO_STREAM("Input binding index: " << engine_->getBindingDataType(input_idx));

  if (engine_->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
    ROS_WARN_STREAM("Input type doesn't match expected: "
                    << engine_->getBindingDataType(input_idx)
                    << " != " << nvinfer1::DataType::kFLOAT);
  }

  context_->setBindingDimensions(input_idx, config_.getInputDims(3));
  input_buffer_.reset(config_.getInputDims(3));

  nn_img_ = cv::Mat(config_.getInputMatDims(3), CV_32FC1);
  return true;
}

bool TrtSegmenter::createOutputBuffer() {
  auto output_idx = engine_->getBindingIndex(config_.output_name.c_str());
  if (output_idx == -1) {
    ROS_FATAL_STREAM("Failed to get index for output: " << config_.output_name);
    return false;
  }

  ROS_INFO_STREAM("Output binding index: " << engine_->getBindingDataType(output_idx));

  // The output datatype controls precision,
  // https://github.com/NVIDIA/TensorRT/issues/717
  if (engine_->getBindingDataType(output_idx) != nvinfer1::DataType::kINT32) {
    ROS_WARN_STREAM("Output type doesn't match expected: "
                    << engine_->getBindingDataType(output_idx)
                    << " != " << nvinfer1::DataType::kINT32);
  }

  auto output_dims = context_->getBindingDimensions(output_idx);
  output_buffer_.reset(output_dims);

  classes_ = cv::Mat(config_.height, config_.width, CV_32S);
  return true;
}

bool TrtSegmenter::init() {
  if (!createInputBuffer()) {
    return false;
  }

  if (!createOutputBuffer()) {
    return false;
  }

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    ROS_FATAL_STREAM("Creating cuda stream failed!");
    return false;
  }

  initialized_ = true;
  ROS_INFO("Segmenter initialized!");
  return true;
}

std::vector<void *> TrtSegmenter::getBindings() const {
  return {input_buffer_.memory.get(), output_buffer_.memory.get()};
}

bool TrtSegmenter::infer(const cv::Mat &img) {
  fillNetworkImage(config_, img, nn_img_);

  // TODO(nathan) we probably should double check that sizes line up
  auto error = cudaMemcpyAsync(input_buffer_.memory.get(),
                               nn_img_.data,
                               input_buffer_.size,
                               cudaMemcpyHostToDevice,
                               stream_);
  if (error != cudaSuccess) {
    ROS_FATAL_STREAM("copying image to gpu failed: " << cudaGetErrorString(error));
    return false;
  }

  auto bindings = getBindings();
  bool status = context_->enqueueV2(bindings.data(), stream_, nullptr);
  if (!status) {
    ROS_FATAL_STREAM("initializing inference failed!");
    return false;
  }

  error = cudaMemcpyAsync(classes_.data,
                          output_buffer_.memory.get(),
                          classes_.step[0] * classes_.rows,
                          cudaMemcpyDeviceToHost,
                          stream_);
  if (error != cudaSuccess) {
    ROS_FATAL_STREAM("Copying output failed: " << cudaGetErrorString(error));
    return false;
  }

  cudaStreamSynchronize(stream_);
  return true;
}

const cv::Mat &TrtSegmenter::getClasses() const { return classes_; }

}  // namespace semantic_recolor
