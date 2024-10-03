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
#include <semantic_inference/image_utilities.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace semantic_inference {

struct ColorImageTestCase {
  bool rgb_order = true;
  bool chw_order = true;
  static constexpr float tolerance = 1.0e-6;
  static constexpr int height = 5;
  static constexpr int width = 4;

  cv::Mat getInput() const;
  cv::Mat getOutput() const;
  float getExpected(int row, int col, int channel) const;
};

struct ColorConversionFixture : public testing::TestWithParam<ColorImageTestCase> {
  ColorConversionFixture() {}
  virtual ~ColorConversionFixture() = default;
};

ColorImageTestCase color_conversion_test_cases[]{
    {true, true},    // rgb, chw
    {false, true},   // bgr, chw
    {true, false},   // rgb, hwc
    {false, false},  // bgr, hwc
};

INSTANTIATE_TEST_SUITE_P(
    ImageUtilities,
    ColorConversionFixture,
    testing::ValuesIn(color_conversion_test_cases),
    [](const testing::TestParamInfo<ColorConversionFixture::ParamType>& info) {
      const auto& param = info.param;
      std::string color_order = param.rgb_order ? "RGB" : "BGR";
      std::string layout = param.chw_order ? "ChannelsFirst" : "ChannelsLast";
      return color_order + layout;
    });

cv::Mat ColorImageTestCase::getInput() const {
  cv::Mat input(height, width, CV_8UC3);
  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      uint8_t value = r * input.cols * 3 + c * 3;
      input.at<cv::Vec3b>(r, c)[0] = value;
      input.at<cv::Vec3b>(r, c)[1] = value + 1;
      input.at<cv::Vec3b>(r, c)[2] = value + 2;
    }
  }

  return input;
}

cv::Mat ColorImageTestCase::getOutput() const {
  const std::vector<int> dims = chw_order ? std::vector<int>{3, height, width}
                                          : std::vector<int>{height, width, 3};
  return cv::Mat(dims, CV_32FC1);
}

float ColorImageTestCase::getExpected(int row, int col, int channel) const {
  float value = row * 3 * width + col * 3;
  return rgb_order ? value + (2 - channel) : value + channel;
}

TEST_P(ColorConversionFixture, FillColorImageCorrect) {
  const auto test_config = GetParam();
  const auto input = test_config.getInput();
  auto output = test_config.getOutput();

  ColorConverter::Config config;
  config.map_to_unit_range = false;
  config.normalize = false;
  config.rgb_order = test_config.rgb_order;

  const ColorConverter converter(config);
  converter.fillImage(input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      for (int channel = 0; channel < 3; ++channel) {
        const auto expected = test_config.getExpected(r, c, channel);
        const auto result = test_config.chw_order ? output.at<float>(channel, r, c)
                                                  : output.at<float>(r, c, channel);
        EXPECT_NEAR(result, expected, test_config.tolerance)
            << "r=" << r << ", c=" << c << ", channel=" << channel;
      }
    }
  }
}

TEST(ImageUtilities, FillDepthImage) {
  constexpr float tolerance = 1.0e-6;

  DepthConverter::Config config;
  config.mean = -1.0;
  config.stddev = 0.5;
  config.normalize = true;
  const DepthConverter converter(config);

  cv::Mat input(5, 4, CV_32FC1);
  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      input.at<float>(r, c) = r * input.cols + c;
    }
  }

  std::vector<int> dims{5, 4};
  cv::Mat output(dims, CV_32FC1);
  converter.fillImage(input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = ((r * input.cols + c) + 1.0f) * 2.0f;
      EXPECT_NEAR(output.at<float>(r, c), value, tolerance) << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageUtilities, ConvertColorCorrect) {
  constexpr float tolerance = 1.0e-5;
  {  // passthrough conversion is correct
    ColorConverter::Config config;
    config.normalize = false;
    config.map_to_unit_range = false;
    const ColorConverter converter(config);
    EXPECT_NEAR(converter.convert(255, 0), 255.0f, tolerance);
    EXPECT_NEAR(converter.convert(127, 0), 127.0f, tolerance);
    EXPECT_NEAR(converter.convert(0, 0), 0.0f, tolerance);
  }

  {  // unit range is correct
    ColorConverter::Config config;
    config.normalize = false;
    config.map_to_unit_range = true;
    const ColorConverter converter(config);
    EXPECT_NEAR(converter.convert(255, 0), 1.0f, tolerance);
    EXPECT_NEAR(converter.convert(127, 0), 127.0f / 255.0f, tolerance);
    EXPECT_NEAR(converter.convert(0, 0), 0.0f, tolerance);
  }

  {  // custom normalization parameters are correct
    ColorConverter::Config config;
    config.mean = {0.0f, 0.5f, 1.0f};
    config.stddev = {1.0f, 2.0f, 3.0f};
    config.map_to_unit_range = true;
    config.normalize = true;
    const ColorConverter converter(config);
    EXPECT_NEAR(converter.convert(255, 0), 1.0f, tolerance);
    EXPECT_NEAR(converter.convert(255, 1), 0.25f, tolerance);
    EXPECT_NEAR(converter.convert(255, 2), 0.0f, tolerance);
  }
}

}  // namespace semantic_inference
