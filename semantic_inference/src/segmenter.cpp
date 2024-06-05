#include "semantic_inference/segmenter.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

#include <memory>

#include "semantic_inference/image_utilities.h"
#include "semantic_inference/logging.h"
#include "semantic_inference/model_config.h"
#include "semantic_inference_config.h"
#if defined(ENABLE_TENSORRT) && ENABLE_TENSORRT
#include "model.h"
#include "trt_utilities.h"
#endif

namespace semantic_inference {

#if defined(ENABLE_TENSORRT) && ENABLE_TENSORRT
struct Segmenter::Impl {
  explicit Impl(const ModelConfig& config) : model(config) {}
  Model model;
  SegmentationResult infer(const cv::Mat& color, const cv::Mat& depth) {
    if (!model.setInputs(color, depth)) {
      SLOG(ERROR) << "Failed to set input(s) for model!";
      return {};
    }

    return model.infer();
  }
};
#else
struct Segmenter::Impl {
  explicit Impl(const ModelConfig&) {
    SLOG(FATAL) << "Segmentation not supported without tensorrt!"
                << " See readme for installation instructions";
    throw std::runtime_error("tensorrt not installed");
  }

  SegmentationResult infer(const cv::Mat&, const cv::Mat&) { return {}; }
};
#endif

Segmenter::Segmenter(const Config& config)
    : config(config::checkValid(config)),
      impl_(new Impl(config.model)),
      mask_(config.depth_mask) {}

Segmenter::~Segmenter() = default;

SegmentationResult Segmenter::infer(const cv::Mat& color, const cv::Mat& depth) {
  auto result = impl_->infer(color, depth);
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

}  // namespace semantic_inference
