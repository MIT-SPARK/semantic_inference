#include <gtest/gtest.h>
#include <semantic_inference/model_config.h>

using namespace semantic_inference;

TEST(ModelConfig, TestInputAddressRgb) {
  ModelConfig config;
  config.network_uses_rgb_order = true;

  ModelConfig::ImageAddress addr;
  config.fillInputAddress(addr);

  EXPECT_EQ(addr[0], 2);
  EXPECT_EQ(addr[1], 1);
  EXPECT_EQ(addr[2], 0);
}

TEST(ModelConfig, TestInputAddressBgr) {
  ModelConfig config;
  config.network_uses_rgb_order = false;

  ModelConfig::ImageAddress addr;
  config.fillInputAddress(addr);

  EXPECT_EQ(addr[0], 0);
  EXPECT_EQ(addr[1], 1);
  EXPECT_EQ(addr[2], 2);
}

TEST(ModelConfig, TestGetValue) {
  const float GET_VALUE_TOLERANCE = 1.0e-5;
  ModelConfig config;
  config.normalize = false;
  config.map_to_unit_range = false;

  EXPECT_NEAR(config.getValue(255, 0), 255.0f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(127, 0), 127.0f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(0, 0), 0.0f, GET_VALUE_TOLERANCE);

  config.map_to_unit_range = true;
  EXPECT_NEAR(config.getValue(255, 0), 1.0f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(127, 0), 127.0f / 255.0f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(0, 0), 0.0f, GET_VALUE_TOLERANCE);

  config.mean = {0.0f, 0.5f, 1.0f};
  config.stddev = {1.0f, 2.0f, 3.0f};
  config.normalize = true;

  EXPECT_NEAR(config.getValue(255, 0), 1.0f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(255, 1), 0.25f, GET_VALUE_TOLERANCE);
  EXPECT_NEAR(config.getValue(255, 2), 0.0f, GET_VALUE_TOLERANCE);
}

TEST(ModelConfig, TestGetInputDimsNetworkOrder) {
  ModelConfig config;
  config.width = 5;
  config.height = 10;
  config.use_network_order = true;
  auto input_dims = config.getInputDims(2);
  auto cv_input_dims = config.getInputMatDims(2);

  ASSERT_EQ(cv_input_dims.size(), 3u);

  EXPECT_EQ(input_dims.d[0], 1);
  EXPECT_EQ(input_dims.d[1], 2);
  EXPECT_EQ(input_dims.d[2], 10);
  EXPECT_EQ(input_dims.d[3], 5);

  EXPECT_EQ(input_dims.d[1], cv_input_dims[0]);
  EXPECT_EQ(input_dims.d[2], cv_input_dims[1]);
  EXPECT_EQ(input_dims.d[3], cv_input_dims[2]);
}

TEST(ModelConfig, TestGetInputDimsRowMajor) {
  ModelConfig config;
  config.width = 5;
  config.height = 10;
  config.use_network_order = false;
  auto input_dims = config.getInputDims(2);
  auto cv_input_dims = config.getInputMatDims(2);

  ASSERT_EQ(cv_input_dims.size(), 3u);

  EXPECT_EQ(input_dims.d[0], 1);
  EXPECT_EQ(input_dims.d[1], 10);
  EXPECT_EQ(input_dims.d[2], 5);
  EXPECT_EQ(input_dims.d[3], 2);

  EXPECT_EQ(input_dims.d[1], cv_input_dims[0]);
  EXPECT_EQ(input_dims.d[2], cv_input_dims[1]);
  EXPECT_EQ(input_dims.d[3], cv_input_dims[2]);
}
