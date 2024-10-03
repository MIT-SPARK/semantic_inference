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

#include "semantic_inference/image_rotator.h"

#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>

namespace semantic_inference {
namespace {

RotationInfo getRotationInfo(RotationType type) {
  switch (type) {
    case RotationType::ROTATE_90_CLOCKWISE:
      return {true, cv::ROTATE_90_CLOCKWISE, cv::ROTATE_90_COUNTERCLOCKWISE};
    case RotationType::ROTATE_180:
      return {true, cv::ROTATE_180, cv::ROTATE_180};
    case RotationType::ROTATE_90_COUNTERCLOCKWISE:
      return {true, cv::ROTATE_90_COUNTERCLOCKWISE, cv::ROTATE_90_CLOCKWISE};
    case RotationType::NONE:
    default:
      return {};
  }
}

}  // namespace

void declare_config(ImageRotator::Config& config) {
  using namespace config;
  name("ImageRotator::Config");
  enum_field(
      config.rotation,
      "rotation",
      {{RotationType::NONE, "none"},
       {RotationType::ROTATE_90_CLOCKWISE, "ROTATE_90_CLOCKWISE"},
       {RotationType::ROTATE_180, "ROTATE_180"},
       {RotationType::ROTATE_90_COUNTERCLOCKWISE, "ROTATE_90_COUNTERCLOCKWISE"}});
}

ImageRotator::ImageRotator() : ImageRotator(Config{}) {}

ImageRotator::ImageRotator(const Config& config)
    : config(config::checkValid(config)), info_(getRotationInfo(config.rotation)) {}

ImageRotator::ImageRotator(const ImageRotator& other)
    : config(other.config), info_(other.info_) {}

ImageRotator& ImageRotator::operator=(const ImageRotator& other) {
  const_cast<Config&>(config) = other.config;
  info_ = other.info_;
  return *this;
}

ImageRotator::operator bool() const { return info_.needs_rotation; }

cv::Mat ImageRotator::rotate(const cv::Mat& original) const {
  cv::Mat rotated;
  if (!info_.needs_rotation) {
    return rotated = original;
  } else {
    cv::rotate(original, rotated, info_.pre_rotation);
  }

  return rotated;
}

cv::Mat ImageRotator::derotate(const cv::Mat& rotated) const {
  cv::Mat original;
  if (!info_.needs_rotation) {
    return original = rotated;
  } else {
    cv::rotate(rotated, original, info_.post_rotation);
  }

  return original;
}

}  // namespace semantic_inference
