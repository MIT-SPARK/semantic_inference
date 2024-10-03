/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * -------------------------------------------------------------------------- */

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
