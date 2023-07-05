#pragma once
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

class RgbdSegmenter : public SemanticSegmenter {
 public:
  RgbdSegmenter(const ModelConfig& config, const DepthConfig& depth_config);

  virtual ~RgbdSegmenter();

  bool init() override;

  bool infer(const cv::Mat& img, const cv::Mat& depth_img);

 protected:
  void showStats(const cv::Mat& img, int channel, const std::string& name) const;

  bool createDepthBuffer();

  std::vector<void*> getBindings() const override;

  DepthConfig depth_config_;
  cv::Mat nn_depth_img_;
};

}  // namespace semantic_recolor
