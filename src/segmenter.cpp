#include "semantic_recolor/segmenter.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

#include <memory>

#include "model.h"
#include "semantic_recolor/image_utilities.h"
#include "semantic_recolor/logging.h"
#include "semantic_recolor/model_config.h"
#include "trt_utilities.h"

namespace semantic_recolor {

struct Segmenter::Impl {
  explicit Impl(const ModelConfig& config) : model(config) {}
  Model model;
};

Segmenter::Segmenter(const Config& config)
    : config(config::checkValid(config)),
      impl_(new Impl(config.model)),
      mask_(config.depth_mask) {}

Segmenter::~Segmenter() = default;

SegmentationResult Segmenter::infer(const cv::Mat& color, const cv::Mat& depth) {
  if (!impl_->model.setInputs(color, depth)) {
    SLOG(ERROR) << "Failed to set input(s) for model!";
    return {};
  }

  auto result = impl_->model.infer();
  if (!result || depth.empty() || !config.mask_predictions_with_depth) {
    return result;
  }

  return {true, mask_.maskLabels(result.labels, depth)};
}

void declare_config(Segmenter::Config& config) {
  using namespace config;
  name("Segmenter::Config");
  field(config.model, "model");
  field(config.mask_predictions_with_depth, "mask_predictions_with_depth");
  field(config.depth_mask, "depth_mask");
}

}  // namespace semantic_recolor
