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

#include "semantic_inference/image_resizer.h"
#include "semantic_inference/logging.h"

#include <config_utilities/config.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <opencv2/opencv.hpp>


namespace semantic_inference {

void declare_config(ImageResizer::Config& config) {
  using namespace config;
  name("ImageResizer::Config");
  config::field(config.width, "width", "pixels");
  config::field(config.height, "height", "pixels");
}

ImageResizer::ImageResizer() : ImageResizer(Config{}) {}

ImageResizer::ImageResizer(const Config& config)
    : config(config::checkValid(config)) {}

ImageResizer::ImageResizer(const ImageResizer& other)
    : config(other.config) {}

ImageResizer& ImageResizer::operator=(const ImageResizer& other) {
  const_cast<Config&>(config) = other.config;
  return *this;
}

cv::Mat ImageResizer::resizeForModelInput(const cv::Mat& input) const
{
    // if either width or height is negative, then no op. Note defaults are both -1
    if (config.width < 0 || config.height < 0)
        return input;

    cv::Mat scaled;
    cv::resize(input, scaled, cv::Size(config.width, config.height), 0.0, 0.0, cv::INTER_AREA);
    return scaled;
}

cv::Mat ImageResizer::restoreToOriginal(const cv::Mat& input, const cv::Size& target) const
{
    if (config.width < 0 || config.height < 0)
        return input;

    cv::Mat scaled;
    cv::resize(input, scaled, target, 0, 0, cv::INTER_NEAREST);
    return scaled;
}


}  // namespace semantic_inference
