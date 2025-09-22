#include "semantic_inference_ros/pointcloud_projection.h"

#include <config_utilities/config.h>

#include <optional>

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

  void discard() {
    *x_iter_ = std::numeric_limits<float>::quiet_NaN();
    *y_iter_ = std::numeric_limits<float>::quiet_NaN();
    *z_iter_ = std::numeric_limits<float>::quiet_NaN();
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

struct InputLabelIterBase {
  virtual ~InputLabelIterBase() = default;
  virtual bool has_next() const = 0;
  virtual void next() = 0;
  virtual uint32_t get() const = 0;
};

template <typename T>
struct InputLabelIter : InputLabelIterBase {
  InputLabelIter(const PointCloud2& cloud, const std::string& field_name)
      : iter_(cloud, field_name) {}

  bool has_next() const override { return iter_ != iter_.end(); }
  void next() override { ++iter_; }
  uint32_t get() const override { return *iter_; }

 private:
  sensor_msgs::PointCloud2ConstIterator<T> iter_;
};

std::unique_ptr<InputLabelIterBase> iterFromFields(const PointCloud2& cloud,
                                                   const std::string& field_name) {
  if (field_name.empty()) {
    return nullptr;
  }

  for (const auto& field : cloud.fields) {
    if (field.name != field_name) {
      continue;
    }

    switch (field.datatype) {
      case PointField::INT8:
        return std::make_unique<InputLabelIter<int8_t>>(cloud, field_name);
      case PointField::UINT8:
        return std::make_unique<InputLabelIter<uint8_t>>(cloud, field_name);
      case PointField::INT16:
        return std::make_unique<InputLabelIter<int16_t>>(cloud, field_name);
      case PointField::UINT16:
        return std::make_unique<InputLabelIter<uint16_t>>(cloud, field_name);
      case PointField::INT32:
        return std::make_unique<InputLabelIter<int32_t>>(cloud, field_name);
      case PointField::UINT32:
        return std::make_unique<InputLabelIter<uint32_t>>(cloud, field_name);
      default:
        SLOG(ERROR) << "Unknown label type: " << std::to_string(field.datatype);
        return nullptr;
    }
  }

  std::stringstream ss;
  ss << "[";
  auto iter = cloud.fields.begin();
  while (iter != cloud.fields.end()) {
    ss << "'" << iter->name << "'";
    ++iter;
    if (iter != cloud.fields.end()) {
      ss << ", ";
    }
  }
  ss << "]";

  SLOG(ERROR) << "Missing field '" << field_name << "' (available: " << ss.str() << ")";
  return nullptr;
}

struct InputLabelAdapter {
  InputLabelAdapter(const PointCloud2& cloud, const std::string& field_name)
      : iter_(iterFromFields(cloud, field_name)) {}

  operator bool() const { return iter_ && iter_->has_next(); }

  std::optional<uint32_t> operator*() const {
    if (iter_) {
      return iter_->get();
    } else {
      return std::nullopt;
    }
  }

  InputLabelAdapter& operator++() {
    if (iter_) {
      iter_->next();
    }
    return *this;
  }

 private:
  std::unique_ptr<InputLabelIterBase> iter_;
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

struct LabelImageAdapter {
  explicit LabelImageAdapter(const cv::Mat& img) : image(img) {
    switch (image.type()) {
      case CV_8UC1:
        getter = [this](int r, int c) -> uint32_t { return image.at<uint8_t>(r, c); };
        break;
      case CV_8SC1:
        getter = [this](int r, int c) -> uint32_t { return image.at<int8_t>(r, c); };
        break;
      case CV_16UC1:
        getter = [this](int r, int c) -> uint32_t { return image.at<uint16_t>(r, c); };
        break;
      case CV_16SC1:
        getter = [this](int r, int c) -> uint32_t { return image.at<int16_t>(r, c); };
        break;
      case CV_32SC1:
        getter = [this](int r, int c) -> uint32_t { return image.at<int32_t>(r, c); };
        break;
      default:
        SLOG(ERROR) << "Unknown label type: " << image.type();
    }
  }

  operator bool() const { return static_cast<bool>(getter); }

  uint32_t operator()(int row, int col) const { return getter(row, col); }

  cv::Mat image;
  std::function<uint32_t(int, int)> getter;
};

void recolorCloud(PointCloud2& output,
                  const ImageRecolor& recolor,
                  uint32_t unknown_label) {
  auto labels = sensor_msgs::PointCloud2ConstIterator<uint32_t>(output, "label");
  auto colors = sensor_msgs::PointCloud2Iterator<uint8_t>(output, "rgba");
  while (labels != labels.end()) {
    const auto unknown = static_cast<uint32_t>(*labels) == unknown_label;
    const auto& color = unknown ? recolor.default_color : recolor.getColor(*labels);
    // annoyingly BGR order even if field is RGBA
    colors[0] = color[2];
    colors[1] = color[1];
    colors[2] = color[0];
    colors[3] = 255u;
    ++labels;
    ++colors;
  }
}

}  // namespace

struct LabelConverter {
  static int32_t toIntermediate(uint32_t orig, std::string&) { return orig; }

  static void fromIntermediate(const int32_t& intermediate,
                               uint32_t& value,
                               std::string&) {
    value = intermediate;
  }
};

void declare_config(ProjectionConfig& config) {
  using namespace config;
  name("ProjectionConfig::Config");
  field(config.use_lidar_frame, "use_lidar_frame");
  field(config.discard_out_of_view, "discard_out_of_view");
  field<LabelConverter>(config.unknown_label, "unknown_label");
  field(config.input_label_fieldname, "input_label_fieldname");
  field(config.override_labels, "override_labels");
  field(config.allowed_labels, "allowed_labels");
  field(config.input_remapping, "input_remapping");
}

std::optional<uint32_t> ProjectionConfig::remapInput(
    std::optional<uint32_t> orig) const {
  if (!orig) {
    return std::nullopt;
  }

  const auto label = orig.value();
  const auto iter = input_remapping.find(label);
  return iter == input_remapping.end() ? label : iter->second;
}

bool projectSemanticImage(const ProjectionConfig& config,
                          const CameraInfo& intrinsics,
                          const cv::Mat& image,
                          const PointCloud2& cloud,
                          const Eigen::Isometry3f& image_T_cloud,
                          PointCloud2& output,
                          const ImageRecolor* recolor) {
  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(intrinsics);

  auto pos_in_iter = InputPosIter(cloud);
  const LabelImageAdapter img_wrapper(image);
  // iterator over label field in input pointcloud if it exists
  InputLabelAdapter label_in_iter(cloud, config.input_label_fieldname);

  initOutput(cloud, output, recolor != nullptr);
  auto pos_out_iter = OutputPosIter(output);
  auto label_out_iter = sensor_msgs::PointCloud2Iterator<uint32_t>(output, "label");

  while (pos_in_iter) {
    const Eigen::Vector3f p_cloud = *pos_in_iter;
    const Eigen::Vector3f p_image = image_T_cloud * p_cloud;

    int u = -1;
    int v = -1;
    if (p_image.z() > 0.0f) {
      const cv::Point3d p_cv(p_image.x(), p_image.y(), p_image.z());
      const auto& pixel = model.project3dToPixel(p_cv);
      u = std::round(pixel.x);
      v = std::round(pixel.y);
    }

    const auto in_view = u >= 0 && u < image.cols && v >= 0 && v < image.rows;
    const uint32_t label_in =
        config.remapInput(*label_in_iter).value_or(config.unknown_label);
    if (in_view) {
      *label_out_iter = config.isOverride(label_in) ? label_in : img_wrapper(v, u);
    } else {
      *label_out_iter = config.isAllowed(label_in) ? label_in : config.unknown_label;
    }

    if (!in_view && config.discard_out_of_view) {
      pos_out_iter.discard();
    } else {
      pos_out_iter.set(config.use_lidar_frame ? p_cloud : p_image);
    }

    ++pos_in_iter;
    ++pos_out_iter;
    ++label_in_iter;
    ++label_out_iter;
  }

  if (recolor) {
    recolorCloud(output, *recolor, config.unknown_label);
  }

  return true;
}

}  // namespace semantic_inference
