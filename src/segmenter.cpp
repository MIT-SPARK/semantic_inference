#include "semantic_recolor/segmenter.h"
#include "semantic_recolor/utilities.h"

#include <NvOnnxParser.h>

#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <memory>
#include <numeric>

namespace semantic_recolor {

using TrtNetworkDef = nvinfer1::INetworkDefinition;
using nvinfer1::NetworkDefinitionCreationFlag;

std::unique_ptr<TrtEngine> deserializeEngine(TrtRuntime &runtime,
                                             const std::string &engine_path) {
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (engine_file.fail()) {
    ROS_INFO_STREAM("Engine file: " << engine_path << " not found!");
    return nullptr;
  }

  const size_t engine_size = getFileSize(engine_file);
  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  std::unique_ptr<TrtEngine> engine(
      runtime.deserializeCudaEngine(engine_data.data(), engine_size));
  if (!engine) {
    ROS_FATAL_STREAM("Engine creation failed");
    return nullptr;
  }

  return engine;
}

std::unique_ptr<TrtEngine> buildEngineFromOnnx(TrtRuntime &runtime,
                                               Logger &logger,
                                               const std::string &model_path,
                                               const std::string &engine_path) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  const auto network_flags =
      1u << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<TrtNetworkDef> network(builder->createNetworkV2(network_flags));

  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));
  parser->parseFromFile(model_path.c_str(), static_cast<int>(Severity::kINFO));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    ROS_ERROR_STREAM("Parser Error #" << i << ": " << parser->getError(i)->desc());
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

  std::unique_ptr<nvinfer1::IHostMemory> memory(
      builder->buildSerializedNetwork(*network, *config));
  if (!memory) {
    ROS_FATAL_STREAM("Failed to build network");
    return nullptr;
  }

  std::ofstream fout(engine_path, std::ios::binary);
  fout.write(reinterpret_cast<char *>(memory->data()), memory->size());

  std::unique_ptr<TrtEngine> engine(
      runtime.deserializeCudaEngine(memory->data(), memory->size()));
  if (!engine) {
    ROS_FATAL_STREAM("Engine creation failed");
    return nullptr;
  }
  return engine;
}

TrtSegmenter::TrtSegmenter(const SegmentationConfig &config)
    : config_(config), logger_(Severity::kINFO), initialized_(false) {
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_ = deserializeEngine(*runtime_, config_.engine_file);
  if (!engine_) {
    ROS_WARN("TRT engine not found! Rebuilding.");
    engine_ = buildEngineFromOnnx(
        *runtime_, logger_, config_.model_file, config_.engine_file);
  }

  if (!engine_) {
    ROS_FATAL_STREAM("Building engine from onnx failed!");
    throw std::runtime_error("failed to load or build engine");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    ROS_FATAL_STREAM("Failed to create execution context");
    throw std::runtime_error("failed to set up trt context");
  }
}

TrtSegmenter::~TrtSegmenter() {
  if (initialized_) {
    cudaStreamDestroy(stream_);
  }
}

bool TrtSegmenter::init() {
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
    // return false;
  }

  nvinfer1::Dims4 input_dims{1, 3, config_.height, config_.width};
  context_->setBindingDimensions(input_idx, input_dims);
  input_buffer_.reset(input_dims);

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
    // return false;
  }

  auto output_dims = context_->getBindingDimensions(output_idx);
  output_buffer_.reset(output_dims);

  std::vector<int> nn_dims{3, config_.height, config_.width};
  nn_img_ = cv::Mat(nn_dims, CV_32FC1);
  classes_ = cv::Mat(config_.height, config_.width, CV_32S);

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
