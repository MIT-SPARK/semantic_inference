#include "semantic_inference_ros/pointcloud_projection.h"

#include <config_utilities/config.h>

#include <image_geometry/pinhole_camera_model.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "semantic_inference_ros/ros_log_sink.h"

namespace semantic_inference {

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

void declare_config(ProjectionConfig& config) {
  using namespace config;
  name("ProjectionConfig::Config");
  field(config.use_lidar_frame, "use_lidar_frame");
  field(config.discard_out_of_view, "discard_out_of_view");
  field(config.unknown_label, "unknown_label");
}

void projectSemanticImage(const ProjectionConfig& config,
                          const CameraInfo& intrinsics,
                          const cv::Mat& image,
                          const PointCloud2& cloud,
                          const Eigen::Isometry3f& image_T_cloud,
                          PointCloud2& output) {
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(intrinsics);

  sensor_msgs::PointCloud2Modifier mod(output);
  // clang-format off
  mod.setPointCloud2Fields(4,
                           "x", 1, PointField::FLOAT32,
                           "y", 1, PointField::FLOAT32,
                           "z", 1, PointField::FLOAT32,
                           "label", 1, PointField::INT32);
  // clang-format on
  mod.resize(cloud.width, cloud.height);

  // techinically we could probably just have one iterator, but (small) chance that
  // someone sends a non-contiguous pointcloud
  auto x_in = sensor_msgs::PointCloud2ConstIterator<float>(cloud, "x");
  auto y_in = sensor_msgs::PointCloud2ConstIterator<float>(cloud, "y");
  auto z_in = sensor_msgs::PointCloud2ConstIterator<float>(cloud, "z");
  auto label_iter = sensor_msgs::PointCloud2Iterator<int32_t>(output, "label");
  while (x_in != x_in.end()) {
    const Eigen::Vector3f p_cloud(*x_in, *y_in, *z_in);
    const Eigen::Vector3f p_image = image_T_cloud * p_cloud;
    ++x_in;
    ++y_in;
    ++z_in;

    int u = -1;
    int v = -1;
    if (p_image.z() > 0.0f) {
      const auto& pixel =
          model.project3dToPixel(cv::Point3d(p_image.x(), p_image.y(), p_image.z()));
      u = std::round(pixel.x);
      v = std::round(pixel.y);
    }

    const auto in_view = u >= 0 && u < image.cols && v >= 0 && v < image.rows;
    *label_iter = in_view ? image.at<int32_t>(v, u) : config.unknown_label;
    ++label_iter;

    if (!in_view && config.discard_out_of_view) {
    }

    // TODO(nathan) optionally null out out-of-view points
  }
}

void colorPointcloud(const ImageRecolor& recolor, PointCloud2& input) {
  auto label_iter = sensor_msgs::PointCloud2ConstIterator<int32_t>(input, "label");
  auto color_iter = sensor_msgs::PointCloud2Iterator<float>(input, "rgb");
  while (label_iter != label_iter.end()) {
    const auto& color = recolor.getColor(*label_iter);
    color_iter[0] = color[0];
    color_iter[1] = color[1];
    color_iter[2] = color[2];
    ++label_iter;
    ++color_iter;
  }
}

}  // namespace semantic_inference
