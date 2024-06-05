#pragma once
#include <NvInfer.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <string>

namespace semantic_inference {

using Severity = nvinfer1::ILogger::Severity;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine>;
using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime>;

struct CudaMemoryManager {
  static void* alloc(size_t size);

  struct Delete {
    void operator()(void* object);
  };
};

struct Shape {
  int width = -1;
  int height = -1;
  std::optional<int> channels;
  bool chw_order = false;

  Shape updateFrom(const cv::Mat& input);

  nvinfer1::Dims dims() const;

  size_t numel() const;

  bool operator==(const Shape& other) const;
};

struct ImageMemoryPair {
  std::unique_ptr<float, CudaMemoryManager::Delete> device_image;
  cv::Mat host_image;
  Shape shape;

  inline operator bool() const {
    return !host_image.empty() && device_image != nullptr;
  }

  bool updateShape(const Shape& shape);

  size_t size() const;
};

std::string toString(const nvinfer1::Dims& dims);

std::string toString(nvinfer1::DataType dtype);

std::string toString(nvinfer1::TensorIOMode mode);

RuntimePtr getRuntime(const std::string& verbosity);

EnginePtr deserializeEngine(nvinfer1::IRuntime& runtime,
                            const std::filesystem::path& engine_path);

EnginePtr buildEngineFromOnnx(nvinfer1::IRuntime& runtime,
                              const std::filesystem::path& model_path,
                              const std::filesystem::path& engine_path,
                              const std::string& verbosity = "INFO");

}  // namespace semantic_inference
