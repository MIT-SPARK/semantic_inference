#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <numeric>
#include <string>

namespace semantic_recolor {

using Severity = nvinfer1::ILogger::Severity;
using TrtRuntime = nvinfer1::IRuntime;
using TrtEngine = nvinfer1::ICudaEngine;
using TrtContext = nvinfer1::IExecutionContext;

struct SegmentationConfig {
  std::string model_file;
  std::string engine_file;
  int width;
  int height;
  std::string input_name{"input.1"};
  std::string output_name{"4464"};
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stddev{0.229f, 0.224f, 0.225f};
};

template <typename T>
struct CudaMemoryHolder {
  CudaMemoryHolder() {}

  explicit CudaMemoryHolder(const nvinfer1::Dims &dims) { reset(dims); }

  void reset(const nvinfer1::Dims &desired_dims) {
    memory.reset(nullptr);  // let cuda decide if it wants to realloc the memory

    dims = desired_dims;
    size =
        std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) *
        sizeof(T);

    void *raw_memory = nullptr;
    auto error = cudaMalloc(&raw_memory, size);
    if (error != cudaSuccess) {
      return;
    }

    memory.reset(reinterpret_cast<T *>(raw_memory));
  }

  struct Deleter {
    void operator()(T *object) {
      if (object != nullptr) {
        cudaFree(object);
      }
    }
  };

  std::unique_ptr<T, Deleter> memory;
  size_t size;
  nvinfer1::Dims dims;
};

class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kINFO) : min_severity_(severity) {}

  void log(Severity severity, const char *msg) noexcept override;

 private:
  Severity min_severity_;
};

class TrtSegmenter {
 public:
  explicit TrtSegmenter(const SegmentationConfig &config);
  ~TrtSegmenter();

  bool init();
  bool infer(const cv::Mat &img);
  const cv::Mat &getClasses() const;

 private:
  SegmentationConfig config_;
  Logger logger_;
  std::unique_ptr<TrtRuntime> runtime_;
  std::unique_ptr<TrtEngine> engine_;
  std::unique_ptr<TrtContext> context_;
  cudaStream_t stream_;

  CudaMemoryHolder<float> input_buffer_;
  CudaMemoryHolder<int32_t> output_buffer_;

  cv::Mat nn_img_;
  cv::Mat classes_;

  bool initialized_;
};

}  // namespace semantic_recolor
