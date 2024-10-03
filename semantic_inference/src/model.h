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

#pragma once
#include <map>
#include <memory>

#include "semantic_inference/model_config.h"
#include "semantic_inference/segmenter.h"
#include "trt_utilities.h"

namespace semantic_inference {

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dims;
  nvinfer1::DataType dtype;

  bool isCHWOrder() const;
  bool isDynamic() const;
  Shape shape() const;
  nvinfer1::Dims replaceDynamic(const cv::Mat& mat) const;
};

std::ostream& operator<<(std::ostream& out, const TensorInfo& info);

class ModelInfo {
 public:
  ModelInfo();

  explicit ModelInfo(const nvinfer1::ICudaEngine& engine);

  operator bool() const { return color_ && labels_; }
  const TensorInfo& color() const { return color_.value(); }
  const std::optional<TensorInfo>& depth() const { return depth_; }
  const TensorInfo& labels() const { return labels_.value(); }

 private:
  bool setIfUnset(const TensorInfo& info, std::optional<TensorInfo>& field);

  std::optional<TensorInfo> color_;
  std::optional<TensorInfo> depth_;
  std::optional<TensorInfo> labels_;
};

std::ostream& operator<<(std::ostream& out, const ModelInfo& info);

class Model {
 public:
  explicit Model(const ModelConfig& config);

  ~Model();

  void initOutput(const cv::Mat& color);

  bool setInputs(const cv::Mat& color, const cv::Mat& depth = cv::Mat());

  SegmentationResult infer() const;

  const ModelConfig model;

 private:
  bool initialized_ = false;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_;

  ModelInfo info_;
  ColorConverter color_conversion_;
  DepthConverter depth_conversion_;

  ImageMemoryPair color_;
  ImageMemoryPair depth_;
  std::unique_ptr<int32_t, CudaMemoryManager::Delete> label_memory_;
};

}  // namespace semantic_inference
