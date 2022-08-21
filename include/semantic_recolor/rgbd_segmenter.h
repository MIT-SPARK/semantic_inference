#pragma once
#include "semantic_recolor/segmenter.h"

namespace semantic_recolor {

class TrtRgbdSegmenter : public TrtSegmenter {
 public:
  TrtRgbdSegmenter(const ModelConfig &config, const std::string &depth_input_name);

  virtual ~TrtRgbdSegmenter();

  bool init() override;

  bool infer(const cv::Mat &img, const cv::Mat &depth_img);

 protected:
  bool createDepthBuffer();

  std::vector<void *> getBindings() const override;

  std::string depth_input_name_;
  CudaMemoryHolder<float> depth_input_buffer_;
  cv::Mat nn_depth_img_;
};

}  // namespace semantic_recolor
