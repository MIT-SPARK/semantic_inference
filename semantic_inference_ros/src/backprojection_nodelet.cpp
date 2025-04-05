#include <config_utilities/config.h>
#include <config_utilities/config_utilities.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <image_geometry/pinhole_camera_model.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
// #include <pcl/common/transforms.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl_ros/point_cloud.h>
#include <tf2_ros/transform_listener.h>

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Geometry>

namespace semantic_inference {

using message_filters::Synchronizer;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;

class SemanticProjector {
 public:
  struct Config {
    bool discard_out_of_view = false;
    bool create_color = true;
    int16_t unknown_label = 0;
    std::string output_cloud_reference_frame = "camera";  // "lidar" or "camera"
    std::string colormap_path;
  } const config;

  explicit SemanticProjector(const Config& config);

  void projectSemanticImage(const cv::Mat& semantic_image,
                            const PointCloud2& cloud,
                            const Eigen::Affine3d& image_T_cloud,
                            PointCloud2& output) const;

  void setCamInfo(const CameraInfo& msg) { cam_model_.fromCameraInfo(msg); }

 private:
  image_geometry::PinholeCameraModel cam_model_;
  std::string output_cloud_reference_frame_ = "camera";
};

SemanticProjector::SemanticProjector(const Config& config)
    : config(config::checkValid(config)) {
  output_cloud_reference_frame_ = config.output_cloud_reference_frame;
}

void SemanticProjector::projectSemanticImage(const cv::Mat& semantic_image,
                                             const PointCloud2& cloud,
                                             const Eigen::Affine3d& image_T_cloud,
                                             PointCloud2& output) const {
  sensor_msgs::PointCloud2Modifier mod(output);
  mod.setPointCloud2FieldsByString("xyz", 
  mod.resize(cloud.width, cloud.height);
  output_cloud.clear();
  output_cloud.reserve(transformed_cloud.size());
  colored_cloud.clear();
  colored_cloud.reserve(transformed_cloud.size());
  for (size_t i = 0; i < transformed_cloud.size(); i++) {
    const auto& pt = transformed_cloud.points[i];
    pcl::PointXYZL pt_labeled;
    pt_labeled.x = pt.x;
    pt_labeled.y = pt.y;
    pt_labeled.z = pt.z;
    pt_labeled.label = config.unknown_label;

    if (pt.z < 0) {
      if (!config.discard_out_of_view) {
        output_cloud.points.emplace_back(pt_labeled);
        if (create_color_) {
          colored_cloud.points.emplace_back(labeledPointToColored(pt_labeled));
        }
      }
      continue;
    }

    const auto& pixel_uv = cam_model_.project3dToPixel(cv::Point3d(pt.x, pt.y, pt.z));
    int u = static_cast<int>(pixel_uv.x);
    int v = static_cast<int>(pixel_uv.y);

    if (u < 0 || u >= semantic_image.cols || v < 0 || v >= semantic_image.rows) {
      if (!config.discard_out_of_view) {
        output_cloud.points.emplace_back(pt_labeled);
        if (create_color_) {
          colored_cloud.points.emplace_back(labeledPointToColored(pt_labeled));
        }
      }
      continue;
    }

    pt_labeled.label = semantic_image.at<int16_t>(v, u);
    output_cloud.points.emplace_back(pt_labeled);
    if (create_color_) {
      colored_cloud.points.emplace_back(labeledPointToColored(pt_labeled));
    }
  }

  if (output_cloud_reference_frame_ == "lidar") {
    const auto output_tmp = output_cloud;
    const auto colored_tmp = colored_cloud;
    pcl::transformPointCloud(
        output_tmp, output_cloud, image_T_cloud.cast<float>().inverse());
    pcl::transformPointCloud(
        colored_tmp, colored_cloud, image_T_cloud.cast<float>().inverse());
  } else if (output_cloud_reference_frame_ != "camera") {
    throw std::invalid_argument("Unsupported parameter detected.");
  }
}

// NOTE(hyungtae) You might not need this function, as Hydra already has
// `color_mesh_by_label` option
void SemanticProjector::projectSemanticImage(
    const cv::Mat& semantic_image,
    const pcl::PointCloud<InputPointType>& cloud,
    const Eigen::Affine3d& image_T_cloud,
    pcl::PointCloud<pcl::PointXYZRGBL>& output_cloud) const {
  pcl::PointCloud<InputPointType> transformed_cloud;
  pcl::transformPointCloud(cloud, transformed_cloud, image_T_cloud.cast<float>());

  output_cloud.clear();
  output_cloud.reserve(transformed_cloud.size());
  for (size_t i = 0; i < transformed_cloud.size(); i++) {
    const auto& pt = transformed_cloud.points[i];
    pcl::PointXYZRGBL pt_labeled;
    pt_labeled.x = pt.x;
    pt_labeled.y = pt.y;
    pt_labeled.z = pt.z;
    pt_labeled.r = 0;
    pt_labeled.g = 0;
    pt_labeled.b = 0;
    pt_labeled.label = config.unknown_label;

    if (pt.z < 0) {
      if (!config.discard_out_of_view) {
        if (create_color_) {
          labeledPointToLabeledAndColored(pt_labeled);
        }
        output_cloud.points.emplace_back(pt_labeled);
      }
      continue;
    }

    const auto& pixel_uv = cam_model_.project3dToPixel(cv::Point3d(pt.x, pt.y, pt.z));
    int u = static_cast<int>(pixel_uv.x);
    int v = static_cast<int>(pixel_uv.y);

    if (u < 0 || u >= semantic_image.cols || v < 0 || v >= semantic_image.rows) {
      if (!config.discard_out_of_view) {
        if (create_color_) {
          labeledPointToLabeledAndColored(pt_labeled);
        }
        output_cloud.points.emplace_back(pt_labeled);
      }
      continue;
    }

    pt_labeled.label = semantic_image.at<int16_t>(v, u);
    if (create_color_) {
      labeledPointToLabeledAndColored(pt_labeled);
    }
    output_cloud.points.emplace_back(pt_labeled);
  }

  if (output_cloud_reference_frame_ == "lidar") {
    const auto output_tmp = output_cloud;
    pcl::transformPointCloud(
        output_tmp, output_cloud, image_T_cloud.cast<float>().inverse());
  } else if (output_cloud_reference_frame_ != "camera") {
    throw std::invalid_argument("Unsupported parameter detected.");
  }
}

pcl::PointXYZRGB SemanticProjector::labeledPointToColored(
    const pcl::PointXYZL& pt_labeled) const {
  pcl::PointXYZRGB colored_pt;
  colored_pt.x = pt_labeled.x;
  colored_pt.y = pt_labeled.y;
  colored_pt.z = pt_labeled.z;

  if (!color_map_.count(pt_labeled.label)) {
    if (pt_labeled.label != static_cast<uint32_t>(config.unknown_label)) {
      LOG_EVERY_N(ERROR, 100) << "Encountered unknown label: " << pt_labeled.label;
    }

    return colored_pt;
  }

  const auto& color = color_map_.at(pt_labeled.label);
  colored_pt.r = static_cast<uint8_t>(color[0]);
  colored_pt.g = static_cast<uint8_t>(color[1]);
  colored_pt.b = static_cast<uint8_t>(color[2]);
  return colored_pt;
}

void SemanticProjector::labeledPointToLabeledAndColored(
    pcl::PointXYZRGBL& pt_labeled) const {
  if (!color_map_.count(pt_labeled.label)) {
    LOG_EVERY_N(ERROR, 100) << "Encountered unknown label: " << pt_labeled.label;
    return;
  }

  const auto& color = color_map_.at(pt_labeled.label);
  pt_labeled.r = static_cast<uint8_t>(color[0]);
  pt_labeled.g = static_cast<uint8_t>(color[1]);
  pt_labeled.b = static_cast<uint8_t>(color[2]);
}

struct BackprojectionNodelet : public rclcpp::Node {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, CameraInfo, PointCloud2>;

  struct Config {
    size_t image_queue_size = 1;
    size_t ptcld_queue_size = 3;
    size_t output_queue_size = 1;
    std::string output_cloud_reference_frame = "camera";  // "lidar" or "camera"
    bool create_color_with_label = false;
    SemanticProjector::Config projector;
    bool show_config = true;
  };

  explicit BackprojectionNodelet(const rclcpp::NodeOptions& options);

 private:
  Eigen::Affine3d getTransform(std::string parent_link,
                               std::string child_link,
                               const ros::Time& stamp);

  void callback(const Image::ConstSharedPtr& image_msg,
                const CameraInfo::ConstSharedPtr& info_msg,
                const PointCloud2::ConstSharedPtr& cloud_msg);

  Config config_;

  message_filters::Subscriber<Image> image_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  message_filters::Subscriber<PointCloud2> cloud_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  rclcpp::Publisher<PointCloud2>::SharedPtr output_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr colored_pub_;

  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  std::unique_ptr<SemanticProjector> projector_;
};

void BackprojectionNodelet::onInit() {
  auto nh = getPrivateNodeHandle();
  config_ = config::fromRos<Config>(nh);
  if (config_.show_config) {
    ROS_INFO_STREAM("config:\n" << config::toString(config_));
  }
  config::checkValid(config_);

  // Do not use image_transport for now since we want to receive raw semantic images
  // semantic_recolor published compressed semantic images have incorrect step size
  image_sub_.subscribe(nh, "semantic_image", config_.image_queue_size);

  info_sub_.subscribe(nh, "camera_info", config_.image_queue_size);

  cloud_sub_.subscribe(nh, "cloud", config_.ptcld_queue_size);

  sync.reset(
      new Synchronizer<SyncPolicy>(SyncPolicy(10), image_sub_, info_sub_, cloud_sub_));
  sync->registerCallback(
      boost::bind(&BackprojectionNodelet::callback, this, _1, _2, _3));

  output_pub_ = nh.advertise<pcl::PointCloud<pcl::PointXYZL>>(
      "semantic_inference", config_.output_queue_size);
  colored_pub_ = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
      "semantic_colored_pointcloud", config_.output_queue_size);

  tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));

  projector_.reset(new SemanticProjector(config_.projector));
}

Eigen::Affine3d BackprojectionNodelet::getTransform(std::string parent_link,
                                                    std::string child_link,
                                                    const ros::Time& stamp) {
  const auto& tf_stamped = tf_buffer_.lookupTransform(parent_link, child_link, stamp);
  return tf2::transformToEigen(tf_stamped);
}

void BackprojectionNodelet::callback(const Image::ConstSharedPtr& image_msg,
                                     const CameraInfo::ConstSharedPtr& info_msg,
                                     const PointCloud2::ConstSharedPtr& cloud_msg) {
  if (!projector_->camModelInit()) {
    projector_->setCamInfo(info_msg);
  }

  // Find transform from cloud to image frame
  const auto& eigen_tf = getTransform(image_msg->header.frame_id,
                                      cloud_msg->header.frame_id,
                                      ros::Time().fromNSec(cloud_msg->header.stamp));

  // Convert image
  cv_bridge::CvImageConstPtr img_ptr;
  try {
    img_ptr = cv_bridge::toCvShare(image_msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
    return;
  }

  cv::Mat semantic_labels;
  if (img_ptr->image.type() == CV_16SC1) {
    semantic_labels = img_ptr->image.clone();
  } else {
    img_ptr->image.convertTo(semantic_labels, CV_16SC1);
  }

  if (config_.create_color_with_label) {
    pcl::PointCloud<pcl::PointXYZRGBL> output_cloud;
    projector_->projectSemanticImage(
        semantic_labels, *cloud_msg, eigen_tf, output_cloud);

    output_cloud.header = cloud_msg->header;
    output_cloud.header.frame_id = config_.output_cloud_reference_frame == "camera"
                                       ? image_msg->header.frame_id
                                       : cloud_msg->header.frame_id;
    output_pub_.publish(output_cloud);
  } else {
    pcl::PointCloud<pcl::PointXYZL> output_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
    projector_->projectSemanticImage(
        semantic_labels, *cloud_msg, eigen_tf, output_cloud, colored_cloud);

    output_cloud.header = cloud_msg->header;
    output_cloud.header.frame_id = config_.output_cloud_reference_frame == "camera"
                                       ? image_msg->header.frame_id
                                       : cloud_msg->header.frame_id;
    output_pub_.publish(output_cloud);

    if (colored_cloud.size() > 0 && colored_pub_.getNumSubscribers() > 0) {
      colored_cloud.header = cloud_msg->header;
      colored_cloud.header.frame_id = config_.output_cloud_reference_frame == "camera"
                                          ? image_msg->header.frame_id
                                          : cloud_msg->header.frame_id;
      colored_pub_.publish(colored_cloud);
    }
  }
}

void declare_config(SemanticProjector::Config& config) {
  using namespace config;
  name("SemanticProjector::Config");
  field(config.discard_out_of_view, "discard_out_of_view");
  field(config.create_color, "create_color");
  field(config.unknown_label, "unknown_label");
  field(config.output_cloud_reference_frame, "output_cloud_reference_frame");
  field(config.colormap_path, "colormap_path");
}

void declare_config(BackprojectionNodelet::Config& config) {
  using namespace config;
  name("BackprojectionNodelet::Config");
  field(config.output_queue_size, "output_queue_size");
  field(config.ptcld_queue_size, "ptcld_queue_size");
  field(config.image_queue_size, "image_queue_size");
  field(config.output_cloud_reference_frame, "output_cloud_reference_frame");
  field(config.create_color_with_label, "create_color_with_label");
  field(config.projector, "projector");
  field(config.show_config, "show_config");
  check(config.output_queue_size, GE, 0, "output_queue_size");
  check(config.ptcld_queue_size, GE, 0, "ptcld_queue_size");
  check(config.image_queue_size, GE, 0, "image_queue_size");
}

}  // namespace semantic_inference

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(semantic_inference::BackprojectionNodelet)
