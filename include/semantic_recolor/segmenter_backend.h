#include <opencv2/opencv.hpp>
#include <string>

#include "semantic_recolor/model_config.h"

namespace semantic_recolor {

class SegmenterBackend {
 public:
  virtual void init(const ModelConfig& config, const std::string& model_path) = 0;

  virtual void run(const cv::Mat& input, cv::Mat& output) const = 0;
};

}  // namespace semantic_recolor
