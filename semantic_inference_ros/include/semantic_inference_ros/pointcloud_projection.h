#include <semantic_inference/image_recolor.h>

#include <cstdint>
#include <string>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Geometry>

namespace semantic_inference {

struct ProjectionConfig {
  bool use_lidar_frame = true;
  bool discard_out_of_view = false;
  int16_t unknown_label = 0;
};

void declare_config(ProjectionConfig& config);

void projectSemanticImage(const ProjectionConfig& config,
                          const sensor_msgs::msg::CameraInfo& intrinsics,
                          const cv::Mat& image,
                          const sensor_msgs::msg::PointCloud2& cloud,
                          const Eigen::Isometry3f& image_T_cloud,
                          sensor_msgs::msg::PointCloud2& output);

void colorPointcloud(const ImageRecolor& recolor, sensor_msgs::msg::PointCloud2& input);

}  // namespace semantic_inference
