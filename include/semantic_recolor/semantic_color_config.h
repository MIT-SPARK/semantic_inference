#pragma once
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

// forward declare for now
namespace ros {
class NodeHandle;
}

namespace semantic_recolor {

struct ColorLabelPair {
  std::vector<double> color;
  std::vector<int> labels;
};

class SemanticColorConfig {
 public:
  SemanticColorConfig();

  explicit SemanticColorConfig(const ros::NodeHandle& nh);

  void initialize(const std::map<int, ColorLabelPair>& classes,
                  const std::vector<double>& default_color);

  void recolorImage(const cv::Mat& classes, cv::Mat& output) const;

  void relabelImage(const cv::Mat& classes, cv::Mat& output) const;

  void show(std::ostream& out) const;

 protected:
  void fillColor(int16_t class_id, uint8_t* pixel, size_t pixel_size = 3) const;

  int16_t getRemappedLabel(int16_t class_id) const;

 private:
  bool initialized_;
  int16_t default_id_;

  std::map<int16_t, int16_t> label_remapping_;
  std::map<int16_t, std::array<uint8_t, 3>> color_map_;
  std::array<uint8_t, 3> default_color_;

  mutable std::set<int16_t> seen_unknown_labels_;
};

}  // namespace semantic_recolor
