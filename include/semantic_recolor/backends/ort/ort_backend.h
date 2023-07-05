#pragma once
#include <memory>

#include "semantic_recolor/segmenter_backend.h"

namespace semantic_recolor {

class OrtBackendImpl;

class OrtBackend : public SegmenterBackend {
 public:
  OrtBackend();

  ~OrtBackend();

  void init(const ModelConfig& config, const std::string& model_path) override;

  void run(const cv::Mat& input, cv::Mat& output) const override;

 private:
  std::unique_ptr<OrtBackendImpl> impl_;
};

}  // namespace semantic_recolor
