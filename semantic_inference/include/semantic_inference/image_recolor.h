#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <opencv2/core/mat.hpp>
#include <set>
#include <string>
#include <vector>

namespace semantic_inference {

struct GroupInfo {
  std::string name;
  std::vector<int16_t> labels;
};

class ImageRecolor {
 public:
  struct Config {
    std::vector<GroupInfo> groups;
    std::vector<uint8_t> default_color{0, 0, 0};
    int16_t default_id = -1;
    int16_t offset = 0;
    std::filesystem::path colormap_path;
  } const config;

  explicit ImageRecolor(const Config& config,
                        const std::map<int16_t, std::array<uint8_t, 3>>& colormap = {});

  static ImageRecolor fromHLS(int16_t num_classes,
                              float luminance = 0.8,
                              float saturation = 0.8);

  void recolorImage(const cv::Mat& classes, cv::Mat& output) const;

  void relabelImage(const cv::Mat& classes, cv::Mat& output) const;

 protected:
  void fillColor(int16_t class_id, uint8_t* pixel, size_t pixel_size = 3) const;

  int16_t getRemappedLabel(int16_t class_id) const;

 private:
  std::map<int16_t, int16_t> label_remapping_;
  std::map<int16_t, std::array<uint8_t, 3>> color_map_;
  mutable std::set<int16_t> seen_unknown_labels_;
};

void declare_config(ImageRecolor::Config& config);

}  // namespace semantic_inference
