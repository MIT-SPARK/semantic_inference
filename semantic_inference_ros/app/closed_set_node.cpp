#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "semantic_inference_ros/segmentation_nodelet.h"

int main(int argc, char* argv[]) {
  const auto before = argc;
  rclcpp::init(argc, argv);
  std::cerr << "before: " << before << ", after: " << argc << std::endl;
  std::cerr << "args:" << std::endl;
  for (int i = 0; i < argc; ++i) {
    std::cerr << "  - " << i << ": " << argv[i] << std::endl;
  }

  const rclcpp::NodeOptions options;  // should pull from default context
  rclcpp::spin(std::make_shared<semantic_inference::SegmentationNode>(options));
  rclcpp::shutdown();
  return 0;
}
