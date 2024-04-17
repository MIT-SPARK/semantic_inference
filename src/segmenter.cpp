#include "semantic_recolor/segmenter.h"

#include <memory>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

TrtSegmenter::TrtSegmenter(const ModelConfig& config)
    : config_(config),
      logger_(config.log_severity, config.use_ros_logging),
      initialized_(false) {
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_ = deserializeEngine(*runtime_, config_.engine_file);
  if (!engine_) {
    LOG_TO_LOGGER(kWARNING, "TRT engine not found! Rebuilding.");
    engine_ = buildEngineFromOnnx(*runtime_,
                                  logger_,
                                  config_.model_file,
                                  config_.engine_file,
                                  config_.set_builder_flags);
    LOG_TO_LOGGER(kINFO, "Finished building engine");
  }

  if (!engine_) {
    LOG_TO_LOGGER(kERROR, "Building engine from onnx failed!");
    throw std::runtime_error("failed to load or build engine");
  }

  LOG_TO_LOGGER(kINFO, "Loaded TRT engine");

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    LOG_TO_LOGGER(kERROR, "Failed to create execution context");
    throw std::runtime_error("failed to set up trt context");
  }

  LOG_TO_LOGGER(kINFO, "TRT execution context started");
}

TrtSegmenter::~TrtSegmenter() {
  if (initialized_) {
    cudaStreamDestroy(stream_);
  }
}

bool TrtSegmenter::createInputBuffer() {
  // TODO(nathan) think about querying if name exists
  const auto tensor_name = config_.input_name.c_str();
  const auto dtype = engine_->getTensorDataType(tensor_name);

  LOG_TO_LOGGER(kINFO, "Input binding type: " << dtype);
  if (dtype != nvinfer1::DataType::kFLOAT) {
    LOG_TO_LOGGER(kERROR,
                  "Input type doesn't match expected: " << dtype << " != "
                                                        << nvinfer1::DataType::kFLOAT);
    return false;
  }

  input_buffer_.reset(config_.getInputDims(3));
  nn_img_ = cv::Mat(config_.getInputMatDims(3), CV_32FC1);

  context_->setInputTensorAddress(tensor_name, input_buffer_.memory.get());
  return true;
}

bool TrtSegmenter::createOutputBuffer() {
  // TODO(nathan) think about querying if name exists
  const auto tensor_name = config_.output_name.c_str();
  const auto dtype = engine_->getTensorDataType(tensor_name);

  LOG_TO_LOGGER(kINFO, "Output binding type: " << dtype);

  // The output datatype controls precision,
  // https://github.com/NVIDIA/TensorRT/issues/717
  if (dtype != nvinfer1::DataType::kINT32) {
    LOG_TO_LOGGER(kWARNING,
                  "Output type doesn't match expected: " << dtype << " != "
                                                         << nvinfer1::DataType::kINT32);
  }

  auto output_dims = context_->getTensorShape(tensor_name);
  output_buffer_.reset(output_dims);
  classes_ = cv::Mat(config_.height, config_.width, CV_32S);

  context_->setTensorAddress(tensor_name, output_buffer_.memory.get());
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
    LOG_TO_LOGGER(kERROR, "Creating cuda stream failed!");
    return false;
  }

  initialized_ = true;
  LOG_TO_LOGGER(kINFO, "Segmenter initialized!");
  return true;
}

bool TrtSegmenter::infer(const cv::Mat& img) {
  fillNetworkImage(config_, img, nn_img_);

  // TODO(nathan) we probably should double check that sizes line up
  auto error = cudaMemcpyAsync(input_buffer_.memory.get(),
                               nn_img_.data,
                               input_buffer_.size,
                               cudaMemcpyHostToDevice,
                               stream_);
  if (error != cudaSuccess) {
    LOG_TO_LOGGER(kERROR, "copying image to gpu failed: " << cudaGetErrorString(error));
    return false;
  }

  cudaStreamSynchronize(stream_);
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    LOG_TO_LOGGER(kERROR, "initializing inference failed!");
    return false;
  }

  error = cudaMemcpyAsync(classes_.data,
                          output_buffer_.memory.get(),
                          classes_.step[0] * classes_.rows,
                          cudaMemcpyDeviceToHost,
                          stream_);
  if (error != cudaSuccess) {
    LOG_TO_LOGGER(kERROR, "Copying output failed: " << cudaGetErrorString(error));
    return false;
  }

  cudaStreamSynchronize(stream_);
  return true;
}

const cv::Mat& TrtSegmenter::getClasses() const { return classes_; }

}  // namespace semantic_recolor
