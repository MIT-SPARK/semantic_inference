#pragma once
#include <opencv2/core/mat.hpp>

#include "semantic_inference/model_config.h"

namespace semantic_inference {

struct SegmentationResult {
  bool valid = false;
  cv::Mat labels;

  inline operator bool() const { return valid; }
};

class Segmenter {
 public:
  struct Config {
    ModelConfig model;
    bool mask_predictions_with_depth = true;
    DepthLabelMask::Config depth_mask;
  } const config;

  explicit Segmenter(const Config& config);

  virtual ~Segmenter();

  SegmentationResult infer(const cv::Mat& img, const cv::Mat& depth = cv::Mat());

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DepthLabelMask mask_;
};

void declare_config(Segmenter::Config& config);

}  // namespace semantic_inference
