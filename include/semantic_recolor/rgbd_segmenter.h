#pragma once
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

struct DepthConfig {
  std::string input_name{"input.1"};
  double conversion_factor = 1.0;
};

class TrtRgbdSegmenter : public TrtSegmenter {
 public:
  TrtRgbdSegmenter(const SegmentationConfig &config, const DepthConfig &depth_config);

  virtual ~TrtRgbdSegmenter();

  bool init() override;

  bool infer(const cv::Mat &img, const cv::Mat &depth_img);

 protected:
  std::vector<void *> getBindings() const override;

  DepthConfig depth_config_;
  CudaMemoryHolder<float> depth_input_buffer_;
};

}  // namespace semantic_recolor
