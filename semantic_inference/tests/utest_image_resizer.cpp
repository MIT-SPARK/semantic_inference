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

#include <gtest/gtest.h>
#include <semantic_inference/image_resizer.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace semantic_inference {

TEST(ImageResizer, ResizeForModelInput) {
  cv::Mat input(cv::Size(500, 500), CV_8UC3);
  cv::randu(input, 0, 255);

  {  // Expect output is unchanged if height is -1
    ImageResizer::Config config;
    config.width = 200;
    config.height = -1;
    const ImageResizer resizer(config);

    cv::Mat output = resizer.resizeForModelInput(input);
    EXPECT_EQ(input.size(), output.size());
    EXPECT_EQ(input.type(), output.type());

    cv::Mat diff;
    cv::compare(input, output, diff, cv::CmpTypes::CMP_NE);
    EXPECT_TRUE(cv::countNonZero(diff.reshape(1)) == 0);
  }

  {  // Expect output is unchanged if width is -1
    ImageResizer::Config config;
    config.width = -1;
    config.height = 200;
    const ImageResizer resizer(config);

    cv::Mat output = resizer.resizeForModelInput(input);
    EXPECT_EQ(input.size(), output.size());
    EXPECT_EQ(input.type(), output.type());

    cv::Mat diff;
    cv::compare(input, output, diff, cv::CmpTypes::CMP_NE);
    EXPECT_TRUE(cv::countNonZero(diff.reshape(1)) == 0);
  }

  {  // Expect output is unchanged with default configuration
    ImageResizer::Config config;
    const ImageResizer resizer(config);

    cv::Mat output = resizer.resizeForModelInput(input);
    EXPECT_EQ(input.size(), output.size());
    EXPECT_EQ(input.type(), output.type());

    cv::Mat diff;
    cv::compare(input, output, diff, cv::CmpTypes::CMP_NE);
    EXPECT_TRUE(cv::countNonZero(diff.reshape(1)) == 0);
  }

  {  // Expect output size is changed to configuration
    ImageResizer::Config config;
    config.width = 300;
    config.height = 200;
    const ImageResizer resizer(config);

    cv::Mat output = resizer.resizeForModelInput(input);
    EXPECT_EQ(output.size(), cv::Size(config.width, config.height));
  }
}

TEST(ImageResizer, RestoreToOriginal) {
  cv::Mat input(cv::Size(300, 200), CV_16SC1);
  cv::randu(input, 0, 32);

  ImageResizer::Config config;
  config.width = 300;
  config.height = 200;
  const ImageResizer resizer(config);

  int original_width = 600;
  int original_height = 400;

  cv::Mat output =
      resizer.restoreToOriginal(input, cv::Size(original_width, original_height));

  EXPECT_EQ(output.size(), cv::Size(original_width, original_height));
  EXPECT_EQ(input.type(), output.type());
}

}  // namespace semantic_inference