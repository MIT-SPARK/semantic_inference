#pragma once
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>

#include "semantic_recolor/model_config.h"
#include "semantic_recolor/segmenter_backend.h"

namespace semantic_recolor {

class SemanticSegmenter {
 public:
  explicit SemanticSegmenter(const ModelConfig& config);

  virtual ~SemanticSegmenter();

  virtual bool init();

  bool infer(const cv::Mat& img, cv::Mat* classes = nullptr);

  const cv::Mat& getClasses() const;

 protected:
  ModelConfig config_;

  cv::Mat nn_img_;
  cv::Mat classes_;

  bool initialized_;
  std::unique_ptr<SegmenterBackend> backend_;
};

}  // namespace semantic_recolor
