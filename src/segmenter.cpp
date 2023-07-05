#include "semantic_recolor/segmenter.h"

#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

SemanticSegmenter::SemanticSegmenter(const ModelConfig& config)
    : config_(config), initialized_(false) {}

SemanticSegmenter::~SemanticSegmenter() {}

bool SemanticSegmenter::init() {
  initialized_ = true;
  // LOG_TO_LOGGER(kINFO, "Segmenter initialized!");
  return true;
}

bool SemanticSegmenter::infer(const cv::Mat& img, cv::Mat* classes) {
  if (!initialized_) {
    return false;
  }

  fillNetworkImage(config_, img, nn_img_);
  backend_->run(nn_img_, classes_);

  if (classes) {
    *classes = classes_;
  }
  return true;
}

const cv::Mat& SemanticSegmenter::getClasses() const { return classes_; }

}  // namespace semantic_recolor
