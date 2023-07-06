#include "semantic_recolor/segmenter.h"

#include "semantic_recolor/backends/ort/ort_backend.h"
#include "semantic_recolor/image_utilities.h"

namespace semantic_recolor {

SemanticSegmenter::SemanticSegmenter(const ModelConfig& config)
    : config_(config), initialized_(false) {}

SemanticSegmenter::~SemanticSegmenter() {}

bool SemanticSegmenter::init(const std::string& model_path) {
  backend_.reset(new OrtBackend());
  if (!backend_->init(config_, model_path)) {
    return false;
  }

  nn_img_ = cv::Mat(config_.getInputDims(3), CV_32FC1);
  classes_ = cv::Mat::zeros(config_.height, config_.width, CV_32S);

  initialized_ = true;
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
