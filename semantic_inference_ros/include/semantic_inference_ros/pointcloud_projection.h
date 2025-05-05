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
  // NOTE(hlim): `ground_label` is optional.
  // If it is set to a value > 0, it is assumed that a ground-segmented
  // point cloud is provided.
  // This activates ground labeling for points outside the image FoV.
  int16_t ground_label = -1;
};

void declare_config(ProjectionConfig& config);

bool projectSemanticImage(const ProjectionConfig& config,
                          const sensor_msgs::msg::CameraInfo& intrinsics,
                          const cv::Mat& image,
                          const sensor_msgs::msg::PointCloud2& cloud,
                          const Eigen::Isometry3f& image_T_cloud,
                          sensor_msgs::msg::PointCloud2& output,
                          const ImageRecolor* recolor = nullptr);

}  // namespace semantic_inference
