#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>

namespace semantic_recolor {

using Severity = nvinfer1::ILogger::Severity;
using TrtRuntime = nvinfer1::IRuntime;
using TrtEngine = nvinfer1::ICudaEngine;
using TrtContext = nvinfer1::IExecutionContext;

std::ostream &operator<<(std::ostream &out, const nvinfer1::Dims &dims);

std::ostream &operator<<(std::ostream &out, const nvinfer1::DataType &dtype);

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
  explicit Logger(Severity severity = Severity::kINFO, bool use_ros = true)
      : min_severity_(severity), use_ros_(use_ros) {}

  void log(Severity severity, const char *msg) noexcept override;

 private:
  Severity min_severity_;
  bool use_ros_;
};

std::unique_ptr<TrtEngine> deserializeEngine(TrtRuntime &runtime,
                                             const std::string &engine_path);

std::unique_ptr<TrtEngine> buildEngineFromOnnx(TrtRuntime &runtime,
                                               Logger &logger,
                                               const std::string &model_path,
                                               const std::string &engine_path,
                                               bool set_builder_flags);

}  // namespace semantic_recolor
