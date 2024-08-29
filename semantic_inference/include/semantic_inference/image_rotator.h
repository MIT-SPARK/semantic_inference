#include <opencv2/core.hpp>

namespace semantic_inference {

enum class RotationType {
  NONE,
  ROTATE_90_CLOCKWISE,
  ROTATE_180,
  ROTATE_90_COUNTERCLOCKWISE,
};

struct RotationInfo {
  bool needs_rotation = false;
  cv::RotateFlags pre_rotation;
  cv::RotateFlags post_rotation;
};

struct ImageRotator {
  struct Config {
    RotationType rotation = RotationType::NONE;
  } const config;

  ImageRotator();
  explicit ImageRotator(const Config& config);
  ImageRotator(const ImageRotator& other);
  ImageRotator& operator=(const ImageRotator& other);

  operator bool() const;
  cv::Mat rotate(const cv::Mat& original) const;
  cv::Mat derotate(const cv::Mat& original) const;

 private:
  RotationInfo info_;
};

void declare_config(ImageRotator::Config& config);

}  // namespace semantic_inference
