#pragma once
#include <map>
#include <memory>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/segmenter.h"
#include "trt_utilities.h"

namespace semantic_recolor {

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;

  bool isCHWOrder() const;
  bool isDynamic() const;
  Shape shape() const;
  nvinfer1::Dims replaceDynamic(const cv::Mat& mat) const;
};

std::ostream& operator<<(std::ostream& out, const TensorInfo& info);

class ModelInfo {
 public:
  ModelInfo();

  explicit ModelInfo(const nvinfer1::ICudaEngine& engine);

  operator bool() const { return color_ && labels_; }
  const TensorInfo& color() const { return color_.value(); }
  const std::optional<TensorInfo>& depth() const { return depth_; }
  const TensorInfo& labels() const { return labels_.value(); }

 private:
  bool setIfUnset(const TensorInfo& info, std::optional<TensorInfo>& field);

  std::optional<TensorInfo> color_;
  std::optional<TensorInfo> depth_;
  std::optional<TensorInfo> labels_;
};

std::ostream& operator<<(std::ostream& out, const ModelInfo& info);

class Model {
 public:
  explicit Model(const ModelConfig& config);

  ~Model();

  void initOutput(const cv::Mat& color);

  bool setInputs(const cv::Mat& color, const cv::Mat& depth = cv::Mat());

  SegmentationResult infer() const;

  const ModelConfig model;

 private:
  bool initialized_ = false;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;

  ModelInfo info_;
  ColorConverter color_conversion_;
  DepthConverter depth_conversion_;

  ImageMemoryPair color_;
  ImageMemoryPair depth_;
  std::unique_ptr<int32_t, CudaMemoryManager::Delete> label_memory_;
};

}  // namespace semantic_recolor
