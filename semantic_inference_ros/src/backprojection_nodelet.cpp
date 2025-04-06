#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/commandline.h>
// #include <config_utilities/types/path.h>
#include <glog/logging.h>
#include <ianvs/image_subscription.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <semantic_inference/image_recolor.h>
#include <tf2_ros/transform_listener.h>

#include <cstdint>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <image_geometry/pinhole_camera_model.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "semantic_inference_ros/ros_log_sink.h"
#include <Eigen/Geometry>

namespace semantic_inference {

using message_filters::Synchronizer;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

struct ProjectionConfig {
  bool use_lidar_frame = true;
  bool discard_out_of_view = false;
  int16_t unknown_label = 0;
};

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

struct BackprojectionNode : public rclcpp::Node {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, CameraInfo, PointCloud2>;

  struct Config {
    ProjectionConfig projection;
    size_t input_queue_size = 1;
    size_t output_queue_size = 1;
    bool show_config = true;
  } const config;

  explicit BackprojectionNode(const rclcpp::NodeOptions& options);

 private:
  Eigen::Isometry3f getTransform(const std::string& parent_link,
                                 const std::string& child_link,
                                 const rclcpp::Time& stamp);

  void callback(const Image::ConstSharedPtr& image_msg,
                const CameraInfo::ConstSharedPtr& info_msg,
                const PointCloud2::ConstSharedPtr& cloud_msg);

  ianvs::ImageSubscription image_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  message_filters::Subscriber<PointCloud2> cloud_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
};

void declare_config(BackprojectionNode::Config& config) {
  using namespace config;
  name("BackprojectionNode::Config");
  field(config.projection, "projection");
  field(config.input_queue_size, "input_queue_size");
  field(config.output_queue_size, "output_queue_size");
  field(config.show_config, "show_config");
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

  pub_ = create_publisher<PointCloud2>("labeled_cloud", config.output_queue_size);

  const rclcpp::QoS qos(config.input_queue_size);
  image_sub_.subscribe("image", config.input_queue_size);
  info_sub_.subscribe(this, "camera_info", qos.get_rmw_qos_profile());
  cloud_sub_.subscribe(this, "cloud", qos.get_rmw_qos_profile());

  sync.reset(
      new Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, info_sub_, cloud_sub_));
  sync->registerCallback(std::bind(&BackprojectionNode::callback, this, _1, _2, _3));
}

Eigen::Isometry3f BackprojectionNode::getTransform(const std::string& parent_link,
                                                   const std::string& child_link,
                                                   const rclcpp::Time& stamp) {
  const auto& tf_stamped = tf_buffer_.lookupTransform(parent_link, child_link, stamp);
  const auto tf_double = tf2::transformToEigen(tf_stamped);
  return tf_double.cast<float>();
}

void BackprojectionNode::callback(const Image::ConstSharedPtr& image_msg,
                                  const CameraInfo::ConstSharedPtr& info_msg,
                                  const PointCloud2::ConstSharedPtr& cloud_msg) {
  // Find transform from cloud to image frame
  const rclcpp::Time stamp(cloud_msg->header.stamp);
  const auto image_T_cloud =
      getTransform(image_msg->header.frame_id, cloud_msg->header.frame_id, stamp);

  // Convert image
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(image_msg);
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }

  cv::Mat semantic_labels;
  if (img_ptr->image.type() == CV_16SC1) {
    semantic_labels = img_ptr->image.clone();
  } else {
    img_ptr->image.convertTo(semantic_labels, CV_16SC1);
  }

  auto output = std::make_unique<PointCloud2>();
  projectSemanticImage(config.projection,
                       *info_msg,
                       semantic_labels,
                       *cloud_msg,
                       image_T_cloud,
                       *output);

  output->header = cloud_msg->header;
  output->header.frame_id = config.projection.use_lidar_frame
                                ? cloud_msg->header.frame_id
                                : image_msg->header.frame_id;
  pub_->publish(std::move(output));
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::BackprojectionNode)
