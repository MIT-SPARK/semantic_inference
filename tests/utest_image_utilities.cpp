#include <gtest/gtest.h>
#include <semantic_recolor/image_utilities.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace semantic_recolor;

TEST(ImageConversion, fillNetworkImageNetworkOrderRgb) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.map_to_unit_range = false;
  config.normalize = false;
  config.network_uses_rgb_order = true;
  config.use_network_order = true;

  cv::Mat input(5, 4, CV_8UC3);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      uint8_t value = r * input.cols * 3 + c * 3;
      input.at<cv::Vec3b>(r, c)[0] = value;
      input.at<cv::Vec3b>(r, c)[1] = value + 1;
      input.at<cv::Vec3b>(r, c)[2] = value + 2;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkImage(config, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols * 3 + c * 3;
      EXPECT_NEAR(output.at<float>(0, r, c), value + 2.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(1, r, c), value + 1.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(2, r, c), value + 0.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageConversion, fillNetworkImageNetworkOrderBgr) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.map_to_unit_range = false;
  config.normalize = false;
  config.network_uses_rgb_order = false;
  config.use_network_order = true;

  cv::Mat input(5, 4, CV_8UC3);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      uint8_t value = r * input.cols * 3 + c * 3;
      input.at<cv::Vec3b>(r, c)[0] = value;
      input.at<cv::Vec3b>(r, c)[1] = value + 1;
      input.at<cv::Vec3b>(r, c)[2] = value + 2;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkImage(config, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols * 3 + c * 3;
      EXPECT_NEAR(output.at<float>(0, r, c), value + 0.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(1, r, c), value + 1.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(2, r, c), value + 2.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageConversion, fillNetworkImageRowMajorOrder) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.map_to_unit_range = false;
  config.normalize = false;
  config.network_uses_rgb_order = true;
  config.use_network_order = false;

  cv::Mat input(5, 4, CV_8UC3);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      uint8_t value = r * input.cols * 3 + c * 3;
      input.at<cv::Vec3b>(r, c)[0] = value;
      input.at<cv::Vec3b>(r, c)[1] = value + 1;
      input.at<cv::Vec3b>(r, c)[2] = value + 2;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkImage(config, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols * 3 + c * 3;
      EXPECT_NEAR(output.at<float>(r, c, 0), value + 2.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(r, c, 1), value + 1.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(r, c, 2), value + 0.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageConversion, fillNetworkImageRowMajorOrderBgr) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.map_to_unit_range = false;
  config.normalize = false;
  config.network_uses_rgb_order = false;
  config.use_network_order = false;

  cv::Mat input(5, 4, CV_8UC3);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      uint8_t value = r * input.cols * 3 + c * 3;
      input.at<cv::Vec3b>(r, c)[0] = value;
      input.at<cv::Vec3b>(r, c)[1] = value + 1;
      input.at<cv::Vec3b>(r, c)[2] = value + 2;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkImage(config, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols * 3 + c * 3;
      EXPECT_NEAR(output.at<float>(r, c, 0), value + 0.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(r, c, 1), value + 1.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
      EXPECT_NEAR(output.at<float>(r, c, 2), value + 2.0f, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageConversion, fillDepthImageNetworkOrder) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.use_network_order = true;

  cv::Mat input(5, 4, CV_32FC1);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      input.at<float>(r, c) = r * input.cols + c;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkDepthImage(config, {}, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols + c;
      EXPECT_NEAR(output.at<float>(0, r, c), value, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}

TEST(ImageConversion, fillDepthImageRowMajorOrder) {
  const float TEST_TOLERANCE = 1.0e-6;

  ModelConfig config;
  config.use_network_order = false;

  cv::Mat input(5, 4, CV_32FC1);
  config.height = input.rows;
  config.width = input.cols;

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      input.at<float>(r, c) = r * input.cols + c;
    }
  }

  cv::Mat output = cv::Mat(config.getInputMatDims(3), CV_32FC1);
  fillNetworkDepthImage(config, {}, input, output);

  for (int r = 0; r < input.rows; ++r) {
    for (int c = 0; c < input.cols; ++c) {
      float value = r * input.cols + c;
      EXPECT_NEAR(output.at<float>(r, c, 0), value, TEST_TOLERANCE)
          << "r=" << r << ", c=" << c;
    }
  }
}
