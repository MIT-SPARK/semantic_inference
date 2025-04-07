#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <semantic_inference_ros/pointcloud_projection.h>

#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace semantic_inference {

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::PointCloud2;

void fillCloud(const std::vector<Eigen::Vector3f>& points,
               PointCloud2& cloud) {
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  for (const auto& point : points) {
    auto& p_out = pcl_cloud.emplace_back();
    p_out.x = point.x();
    p_out.y = point.y();
    p_out.z = point.z();
  }

  pcl::toROSMsg(pcl_cloud, cloud);
}

void checkPoint(const pcl::PointXYZL& expected, const pcl::PointXYZL& result) {
  EXPECT_NEAR(expected.x, result.x, 1.0e-4f);
  EXPECT_NEAR(expected.y, result.y, 1.0e-4f);
  EXPECT_NEAR(expected.z, result.z, 1.0e-4f);
  EXPECT_EQ(expected.label, result.label);
}

TEST(PointcloudProjection, ProjectionCorrect) {
  PointCloud2 cloud;
  std::vector<Eigen::Vector3f> points{{1, 2, 3}, {4, 5, 6}, {4, 5, 2}, {0, 0, -1}};
  fillCloud(points, cloud);

  const auto img = 5 * cv::Mat::ones(3, 4, CV_8UC1);

  CameraInfo info;
  info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  info.p = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  info.width = 4;
  info.height = 3;

  const auto image_T_cloud = Eigen::Isometry3f::Identity();

  PointCloud2 labeled;
  projectSemanticImage({true, false, 1}, info, img, cloud, image_T_cloud, labeled);

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

}  // namespace semantic_inference
