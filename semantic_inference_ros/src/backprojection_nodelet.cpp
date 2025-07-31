#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/commandline.h>
#include <config_utilities/types/path.h>
#include <ianvs/image_subscription.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/transform_listener.h>

#include <cstdint>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/node.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "semantic_inference_ros/pointcloud_projection.h"
#include "semantic_inference_ros/ros_log_sink.h"

namespace semantic_inference {

using message_filters::sync_policies::ApproximateTime;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
using OptPose = std::optional<Eigen::Isometry3f>;

struct BackprojectionNode : public rclcpp::Node {
 public:
  using SyncPolicy = ApproximateTime<Image, CameraInfo, PointCloud2>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;

  struct Config {
    ProjectionConfig projection;
    ImageRecolor::Config recolor;
    size_t input_queue_size = 1;
    size_t output_queue_size = 1;
    bool show_config = true;
    std::string camera_frame;
    std::string lidar_frame;
  } const config;

  explicit BackprojectionNode(const rclcpp::NodeOptions& options);

 private:
  OptPose getTransform(const std::string& parent_link,
                       const std::string& child_link,
                       const rclcpp::Time& stamp);

  void callback(const Image::ConstSharedPtr& image_msg,
                const CameraInfo::ConstSharedPtr& info_msg,
                const PointCloud2::ConstSharedPtr& cloud_msg);

  ianvs::ImageSubscription image_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  message_filters::Subscriber<PointCloud2> cloud_sub_;
  std::unique_ptr<Sync> sync_;
  std::unique_ptr<ImageRecolor> recolor_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
};

void declare_config(BackprojectionNode::Config& config) {
  using namespace config;
  name("BackprojectionNode::Config");
  field(config.projection, "projection");
  field(config.recolor, "recolor");
  field(config.input_queue_size, "input_queue_size");
  field(config.output_queue_size, "output_queue_size");
  field(config.show_config, "show_config");
  field(config.camera_frame, "camera_frame");
  field(config.lidar_frame, "lidar_frame");
  check(config.input_queue_size, GT, 0, "input_queue_size");
  check(config.output_queue_size, GT, 0, "output_queue_size");
}

BackprojectionNode::BackprojectionNode(const rclcpp::NodeOptions& options)
    : Node("backprojection_node", options),
      config(config::fromCLI<Config>(options.arguments())),
      image_sub_(*this),
      tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_) {
  using namespace std::placeholders;

  logging::Logger::addSink("ros", std::make_shared<RosLogSink>(get_logger()));
  logging::setConfigUtilitiesLogger();
  if (config.show_config) {
    SLOG(INFO) << "\n" << config::toString(config);
  }

  config::checkValid(config);
  if (!config.recolor.colormap_path.empty() &&
      std::filesystem::exists(config.recolor.colormap_path)) {
    recolor_ = std::make_unique<ImageRecolor>(config.recolor);
  }

  pub_ = create_publisher<PointCloud2>("labeled_cloud", config.output_queue_size);

  const rclcpp::QoS qos(config.input_queue_size);
  // these are designed to default to the same namespaces as the input to the inference
  // node
  image_sub_.subscribe("semantic/image_raw", config.input_queue_size);
  info_sub_.subscribe(this, "color/camera_info", qos.get_rmw_qos_profile());
  cloud_sub_.subscribe(this, "cloud", qos.get_rmw_qos_profile());
  sync_ = std::make_unique<Sync>(SyncPolicy(10), image_sub_, info_sub_, cloud_sub_);
  sync_->registerCallback(std::bind(&BackprojectionNode::callback, this, _1, _2, _3));
}

OptPose BackprojectionNode::getTransform(const std::string& parent_link,
                                         const std::string& child_link,
                                         const rclcpp::Time& stamp) {
  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_buffer_.lookupTransform(parent_link, child_link, stamp);
  } catch (const std::exception& exception) {
    SLOG(WARNING) << "Failed to lookup tf: " << exception.what();
    return std::nullopt;
  }

  const auto tf_double = tf2::transformToEigen(tf_stamped);
  return tf_double.cast<float>();
}

void BackprojectionNode::callback(const Image::ConstSharedPtr& image_msg,
                                  const CameraInfo::ConstSharedPtr& info_msg,
                                  const PointCloud2::ConstSharedPtr& cloud_msg) {
  // Find transform from cloud to image frame
  const rclcpp::Time stamp(cloud_msg->header.stamp);
  const auto image_T_cloud = getTransform(
      !config.camera_frame.empty() ? config.camera_frame : image_msg->header.frame_id,
      !config.lidar_frame.empty() ? config.lidar_frame : cloud_msg->header.frame_id,
      stamp);
  if (!image_T_cloud) {
    return;
  }

  // Convert image
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(image_msg);
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  auto output = std::make_unique<PointCloud2>();
  const auto valid = projectSemanticImage(config.projection,
                                          *info_msg,
                                          img_ptr->image,
                                          *cloud_msg,
                                          image_T_cloud.value(),
                                          *output,
                                          recolor_.get());
  if (!valid) {
    return;
  }

  output->header = cloud_msg->header;
  output->header.frame_id = config.projection.use_lidar_frame
                                ? cloud_msg->header.frame_id
                                : image_msg->header.frame_id;
  pub_->publish(std::move(output));
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::BackprojectionNode)
