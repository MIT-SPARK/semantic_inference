#pragma once
#include "semantic_recolor/trt_utilities.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <numeric>
#include <string>

namespace semantic_recolor {

struct SegmentationConfig {
  using ImageAddress = std::array<int, 3>;
  using OutputImageAddress = std::array<ImageAddress, 3>;

  std::string model_file;
  std::string engine_file;
  int width;
  int height;
  std::string input_name;
  std::string output_name;
  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stddev{0.229f, 0.224f, 0.225f};
  bool map_to_unit_range = true;
  bool normalize = true;
  bool use_network_order = true;
  bool network_uses_rgb_order = true;

  void fillInputAddress(ImageAddress &addr) const;

  void fillOutputAddress(OutputImageAddress &addr) const;

  void updateOutputAddress(OutputImageAddress &addr, int row, int col) const;

  float getValue(float input_val, size_t channel) const;
};

class TrtSegmenter {
 public:
  explicit TrtSegmenter(const SegmentationConfig &config);

  virtual ~TrtSegmenter();

  virtual bool init();

  bool infer(const cv::Mat &img);

  const cv::Mat &getClasses() const;

 protected:
  virtual std::vector<void *> getBindings() const;

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
