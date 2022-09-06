#pragma once
#include "semantic_recolor/model_config.h"
#include "semantic_recolor/trt_utilities.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <numeric>
#include <string>

namespace semantic_recolor {

#define LOG_TO_LOGGER(level, message)             \
  {                                               \
    std::stringstream ss;                         \
    ss << message;                                \
    const std::string to_log = ss.str();          \
    logger_.log(Severity::level, to_log.c_str()); \
  }                                               \
  static_assert(true, "")

class TrtSegmenter {
 public:
  explicit TrtSegmenter(const ModelConfig &config);

  virtual ~TrtSegmenter();

  virtual bool init();

  bool infer(const cv::Mat &img);

  const cv::Mat &getClasses() const;

 protected:
  bool createInputBuffer();

  bool createOutputBuffer();

  virtual std::vector<void *> getBindings() const;

  ModelConfig config_;
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
