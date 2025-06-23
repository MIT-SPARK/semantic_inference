#include <semantic_inference/image_recolor.h>

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Geometry>

namespace semantic_inference {

struct ProjectionConfig {
  //! Output point positions in LiDAR frame (versus camera frame)
  bool use_lidar_frame = true;
  //! Drop points outside of the field of view of the camera
  bool discard_out_of_view = false;
  //! Label to use for points with no label information
  uint32_t unknown_label = 0;
  //! Optional fieldname for labels contained in the input pointcloud
  std::string input_label_fieldname = "";
  //! Set of input pointcloud labels that take priority over the label image if in view
  std::set<uint32_t> override_labels;
  //! Set of input pointcloud labels that can be forwarded to the output
  std::set<uint32_t> allowed_labels;
  //! Input label remapping for input pointcloud
  std::unordered_map<uint32_t, uint32_t> input_remapping;

  bool isOverride(uint32_t label) const { return override_labels.count(label); }
  bool isAllowed(uint32_t label) const { return allowed_labels.count(label); }
  std::optional<uint32_t> remapInput(std::optional<uint32_t> orig) const;
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
