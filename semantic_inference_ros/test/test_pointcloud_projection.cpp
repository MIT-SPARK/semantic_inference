#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <semantic_inference_ros/pointcloud_projection.h>

#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace semantic_inference {

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::PointCloud2;

void fillCloud(const std::vector<Eigen::Vector3f>& points, PointCloud2& cloud) {
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  for (const auto& point : points) {
    auto& p_out = pcl_cloud.emplace_back();
    p_out.x = point.x();
    p_out.y = point.y();
    p_out.z = point.z();
  }

  pcl::toROSMsg(pcl_cloud, cloud);
}

void fillCloud(const std::vector<Eigen::Vector3f>& points,
               const std::vector<uint32_t>& labels,
               PointCloud2& cloud) {
  if (points.size() != labels.size()) {
    throw std::runtime_error("points and labels sizes do not match!");
  }

  pcl::PointCloud<pcl::PointXYZL> pcl_cloud;
  for (size_t i = 0; i < points.size(); ++i) {
    auto& p_out = pcl_cloud.emplace_back();
    p_out.x = points[i].x();
    p_out.y = points[i].y();
    p_out.z = points[i].z();
    p_out.label = labels[i];
  }

  pcl::toROSMsg(pcl_cloud, cloud);
}

void checkPoint(const pcl::PointXYZL& expected, const pcl::PointXYZL& result) {
  EXPECT_NEAR(expected.x, result.x, 1.0e-4f);
  EXPECT_NEAR(expected.y, result.y, 1.0e-4f);
  EXPECT_NEAR(expected.z, result.z, 1.0e-4f);
  EXPECT_EQ(expected.label, result.label);
}

void checkPoint(const pcl::PointXYZRGBL& expected, const pcl::PointXYZRGBL& result) {
  EXPECT_NEAR(expected.x, result.x, 1.0e-4f);
  EXPECT_NEAR(expected.y, result.y, 1.0e-4f);
  EXPECT_NEAR(expected.z, result.z, 1.0e-4f);
  EXPECT_EQ(expected.r, result.r);
  EXPECT_EQ(expected.g, result.g);
  EXPECT_EQ(expected.b, result.b);
  EXPECT_EQ(expected.label, result.label);
}

pcl::PointXYZRGBL makePoint(
    float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint32_t label) {
  pcl::PointXYZRGBL point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.r = r;
  point.g = g;
  point.b = b;
  point.label = label;
  return point;
}

TEST(PointcloudProjection, ProjectionCorrect) {
  PointCloud2 cloud;
  std::vector<Eigen::Vector3f> points{{1, 2, 3}, {4, 5, 6}, {4, 5, 2}, {0, 0, -1}};
  fillCloud(points, cloud);

  const cv::Mat img = 5 * cv::Mat::ones(3, 4, CV_8UC1);

  CameraInfo info;
  info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  info.width = 4;
  info.height = 3;

  const auto image_T_cloud = Eigen::Isometry3f::Identity();

  ProjectionConfig config;
  config.unknown_label = 1;

  PointCloud2 labeled;
  projectSemanticImage(config, info, img, cloud, image_T_cloud, labeled);

  pcl::PointCloud<pcl::PointXYZL> expected;
  expected.push_back(pcl::PointXYZL{1.0, 2.0, 3.0, 5});
  expected.push_back(pcl::PointXYZL{4.0, 5.0, 6.0, 5});
  expected.push_back(pcl::PointXYZL{4.0, 5.0, 2.0, 1});
  expected.push_back(pcl::PointXYZL{0.0, 0.0, -1.0, 1});

  pcl::PointCloud<pcl::PointXYZL> result;
  pcl::fromROSMsg(labeled, result);
  ASSERT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE("RESULT_CORRECT: " + std::to_string(i));
    checkPoint(expected.at(i), result.at(i));
  }
}

TEST(PointcloudProjection, ColorCorrect) {
  PointCloud2 cloud;
  std::vector<Eigen::Vector3f> points{{1, 2, 1}, {2, 1, 1}, {3, 0, 1}, {-1, -1, 1}};
  fillCloud(points, cloud);

  cv::Mat img = cv::Mat::zeros(3, 4, CV_16SC1);
  for (int r = 0; r < img.rows; ++r) {
    for (int c = 0; c < img.cols; ++c) {
      img.at<int16_t>(r, c) = r;
    }
  }

  const std::map<int16_t, std::array<uint8_t, 3>> cmap{{1, {255, 0, 0}},
                                                       {2, {0, 255, 0}}};
  const ImageRecolor recolor(ImageRecolor::Config(), cmap);

  CameraInfo info;
  info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  info.width = 4;
  info.height = 3;

  const auto image_T_cloud = Eigen::Isometry3f::Identity();

  ProjectionConfig config;
  config.unknown_label = 5;

  PointCloud2 labeled;
  projectSemanticImage(config, info, img, cloud, image_T_cloud, labeled, &recolor);

  pcl::PointCloud<pcl::PointXYZRGBL> expected;
  expected.push_back(makePoint(1.0, 2.0, 1.0, 0, 255, 0, 2));
  expected.push_back(makePoint(2.0, 1.0, 1.0, 255, 0, 0, 1));
  expected.push_back(makePoint(3.0, 0.0, 1.0, 0, 0, 0, 0));
  expected.push_back(makePoint(-1.0, -1.0, 1.0, 0, 0, 0, 5));

  pcl::PointCloud<pcl::PointXYZRGBL> result;
  pcl::fromROSMsg(labeled, result);
  ASSERT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE("RESULT_CORRECT: " + std::to_string(i));
    checkPoint(expected.at(i), result.at(i));
  }
}

TEST(PointcloudProjection, InputLabelsCorrect) {
  PointCloud2 cloud;
  std::vector<Eigen::Vector3f> points{{1, 2, 3}, {4, 5, 6}, {4, 5, 2}, {0, 0, -1}};
  std::vector<uint32_t> labels{3, 4, 6, 5};
  fillCloud(points, labels, cloud);

  const cv::Mat img = 5 * cv::Mat::ones(3, 4, CV_8UC1);

  CameraInfo info;
  info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  info.width = 4;
  info.height = 3;

  const auto image_T_cloud = Eigen::Isometry3f::Identity();

  ProjectionConfig config;
  config.unknown_label = 1;
  config.input_label_fieldname = "label";
  config.override_labels = {3};
  config.allowed_labels = {2};
  config.input_remapping = {{6, 2}};

  PointCloud2 labeled;
  projectSemanticImage(config, info, img, cloud, image_T_cloud, labeled);

  pcl::PointCloud<pcl::PointXYZL> expected;
  expected.push_back(pcl::PointXYZL{1.0, 2.0, 3.0, 3});
  expected.push_back(pcl::PointXYZL{4.0, 5.0, 6.0, 5});
  expected.push_back(pcl::PointXYZL{4.0, 5.0, 2.0, 2});
  expected.push_back(pcl::PointXYZL{0.0, 0.0, -1.0, 1});

  pcl::PointCloud<pcl::PointXYZL> result;
  pcl::fromROSMsg(labeled, result);
  ASSERT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE("RESULT_CORRECT: " + std::to_string(i));
    checkPoint(expected.at(i), result.at(i));
  }
}

}  // namespace semantic_inference
