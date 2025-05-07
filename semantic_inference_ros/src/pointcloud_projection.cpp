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

using LabelIteratorVariant =
    std::variant<sensor_msgs::PointCloud2ConstIterator<int8_t>,
                 sensor_msgs::PointCloud2ConstIterator<uint8_t>,
                 sensor_msgs::PointCloud2ConstIterator<int16_t>,
                 sensor_msgs::PointCloud2ConstIterator<uint16_t>,
                 sensor_msgs::PointCloud2ConstIterator<int32_t>,
                 sensor_msgs::PointCloud2ConstIterator<uint32_t>>;

namespace {
struct OutputPosIter {
 public:
  explicit OutputPosIter(PointCloud2& cloud)
      : x_iter_(cloud, "x"), y_iter_(cloud, "y"), z_iter_(cloud, "z") {}

  operator bool() const { return x_iter_ != x_iter_.end(); }

  void set(const Eigen::Vector3f& p) {
    *x_iter_ = p.x();
    *y_iter_ = p.y();
    *z_iter_ = p.z();
  }

  OutputPosIter& operator++() {
    ++x_iter_;
    ++y_iter_;
    ++z_iter_;
    return *this;
  }

 private:
  // techinically we could probably just have one iterator, but (small) chance that
  // someone sends a non-contiguous pointcloud
  sensor_msgs::PointCloud2Iterator<float> x_iter_;
  sensor_msgs::PointCloud2Iterator<float> y_iter_;
  sensor_msgs::PointCloud2Iterator<float> z_iter_;
};

struct InputPosIter {
 public:
  explicit InputPosIter(const PointCloud2& cloud)
      : x_iter_(cloud, "x"), y_iter_(cloud, "y"), z_iter_(cloud, "z") {}

  operator bool() const { return x_iter_ != x_iter_.end(); }
  Eigen::Vector3f operator*() { return Eigen::Vector3f(*x_iter_, *y_iter_, *z_iter_); }
  InputPosIter& operator++() {
    ++x_iter_;
    ++y_iter_;
    ++z_iter_;
    return *this;
  }

 private:
  // techinically we could probably just have one iterator, but (small) chance that
  // someone sends a non-contiguous pointcloud
  sensor_msgs::PointCloud2ConstIterator<float> x_iter_;
  sensor_msgs::PointCloud2ConstIterator<float> y_iter_;
  sensor_msgs::PointCloud2ConstIterator<float> z_iter_;
};

void initOutput(const PointCloud2& input, PointCloud2& output, bool use_color) {
  sensor_msgs::PointCloud2Modifier mod(output);
  // clang-format off
  if (use_color) {
    mod.setPointCloud2Fields(5,
                             "x", 1, PointField::FLOAT32,
                             "y", 1, PointField::FLOAT32,
                             "z", 1, PointField::FLOAT32,
                             "rgba", 1, PointField::UINT32,
                             "label", 1, PointField::UINT32);
  } else {
    mod.setPointCloud2Fields(4,
                             "x", 1, PointField::FLOAT32,
                             "y", 1, PointField::FLOAT32,
                             "z", 1, PointField::FLOAT32,
                             "label", 1, PointField::UINT32);
  }
  // clang-format on
  mod.resize(input.width, input.height);
}

}  // namespace

void declare_config(ProjectionConfig& config) {
  using namespace config;
  name("ProjectionConfig::Config");
  field(config.use_lidar_frame, "use_lidar_frame");
  field(config.discard_out_of_view, "discard_out_of_view");
  field(config.unknown_label, "unknown_label");
  field(config.input_label_fieldname, "input_label_fieldname");
  field(config.passthrough_labels, "passthrough_labels");
}

std::optional<LabelIteratorVariant> getLabelIterIfExists(
    const PointCloud2& cloud, const std::string& field_name) {
  for (const auto& field : cloud.fields) {
    if (field.name == field_name) {
      switch (field.datatype) {
        case PointField::INT8:
          return sensor_msgs::PointCloud2ConstIterator<int8_t>(cloud, field_name);
        case PointField::UINT8:
          return sensor_msgs::PointCloud2ConstIterator<uint8_t>(cloud, field_name);
        case PointField::INT16:
          return sensor_msgs::PointCloud2ConstIterator<int16_t>(cloud, field_name);
        case PointField::UINT16:
          return sensor_msgs::PointCloud2ConstIterator<uint16_t>(cloud, field_name);
        case PointField::INT32:
          return sensor_msgs::PointCloud2ConstIterator<int32_t>(cloud, field_name);
        case PointField::UINT32:
          return sensor_msgs::PointCloud2ConstIterator<uint32_t>(cloud, field_name);
        default:
          throw std::runtime_error("Unsupported label field datatype: " +
                                   std::to_string(field.datatype));
      }
    }
  }
  return std::nullopt;
}

bool projectSemanticImage(const ProjectionConfig& config,
                          const CameraInfo& intrinsics,
                          const cv::Mat& image,
                          const PointCloud2& cloud,
                          const Eigen::Isometry3f& image_T_cloud,
                          PointCloud2& output,
                          const ImageRecolor* recolor) {
  std::function<int32_t(int, int)> getter;
  switch (image.type()) {
    case CV_8UC1:
      getter = [image](int r, int c) -> int32_t { return image.at<uint8_t>(r, c); };
      break;
    case CV_8SC1:
      getter = [image](int r, int c) -> int32_t { return image.at<int8_t>(r, c); };
      break;
    case CV_16UC1:
      getter = [image](int r, int c) -> int32_t { return image.at<uint16_t>(r, c); };
      break;
    case CV_16SC1:
      getter = [image](int r, int c) -> int32_t { return image.at<int16_t>(r, c); };
      break;
    case CV_32SC1:
      getter = [image](int r, int c) -> int32_t { return image.at<int32_t>(r, c); };
      break;
    default:
      SLOG(ERROR) << "Unknown label type: " << image.type();
      return false;
  }

  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(intrinsics);
  initOutput(cloud, output, recolor != nullptr);

  auto pos_in_iter = InputPosIter(cloud);
  auto pos_out_iter = OutputPosIter(output);
  auto label_out_iter = sensor_msgs::PointCloud2Iterator<int32_t>(output, "label");

  // Optional to label points outside the image FoV.
  auto label_in_iter = getLabelIterIfExists(cloud, config.input_label_fieldname);

  const auto invalid_point =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
  while (pos_in_iter) {
    const Eigen::Vector3f p_cloud = *pos_in_iter;
    const Eigen::Vector3f p_image = image_T_cloud * p_cloud;

    int u = -1;
    int v = -1;
    if (p_image.z() > 0.0f) {
      const auto& pixel =
          model.project3dToPixel(cv::Point3d(p_image.x(), p_image.y(), p_image.z()));
      u = std::round(pixel.x);
      v = std::round(pixel.y);
    }

    const auto in_view = u >= 0 && u < image.cols && v >= 0 && v < image.rows;

    if (in_view) {
      *label_out_iter = getter(v, u);
    } else if (label_in_iter) {
      int16_t input_label = std::visit(
          [](auto& iter) { return static_cast<int16_t>(*iter); }, *label_in_iter);

      if (config.passthrough_labels.count(input_label) > 0) {
        *label_out_iter = input_label;
      } else {
        *label_out_iter = config.unknown_label;
      }
    } else {
      *label_out_iter = config.unknown_label;
    }

    if (!in_view && config.discard_out_of_view) {
      pos_out_iter.set(invalid_point);
    } else {
      pos_out_iter.set(config.use_lidar_frame ? p_cloud : p_image);
    }

    ++pos_in_iter;
    ++pos_out_iter;
    ++label_out_iter;
    if (label_in_iter) {
      std::visit([](auto& iter) { ++iter; }, *label_in_iter);
    }
  }

  if (!recolor) {
    return true;
  }

  auto labels_out = sensor_msgs::PointCloud2ConstIterator<int32_t>(output, "label");
  auto color_iter = sensor_msgs::PointCloud2Iterator<uint8_t>(output, "rgba");
  while (labels_out != labels_out.end()) {
    const auto& color = *labels_out == config.unknown_label
                            ? recolor->default_color
                            : recolor->getColor(*labels_out);
    // annoyingly BGR order even if field is RGBA
    color_iter[0] = color[2];
    color_iter[1] = color[1];
    color_iter[2] = color[0];
    color_iter[3] = 255u;
    ++labels_out;
    ++color_iter;
  }

  return true;
}

}  // namespace semantic_inference
