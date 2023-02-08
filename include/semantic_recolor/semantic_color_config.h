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

  void fillColor(int32_t class_id, uint8_t* pixel, size_t pixel_size = 3) const;

  void fillImage(const cv::Mat& classes, cv::Mat& output) const;

  void show(std::ostream& out) const;

 private:
  bool initialized_;
  std::map<int32_t, std::vector<uint8_t>> color_map_;
  std::vector<uint8_t> default_color_;
  mutable std::set<int32_t> seen_unknown_labels_;
};

}  // namespace semantic_recolor
