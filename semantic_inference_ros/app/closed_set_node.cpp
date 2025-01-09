#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <config_utilities/parsing/commandline.h>
#include "semantic_inference_ros/segmentation_nodelet.h"

int main(int argc, char* argv[]) {
  config::initContext(argc, argv);
  std::cerr << "args:" << std::endl;
  for (int i = 0; i < argc; ++i) {
    std::cerr << "  - " << i << ": " << argv[i] << std::endl;
  }

  rclcpp::init(argc, argv);

  const rclcpp::NodeOptions options;  // should pull from default context
  rclcpp::spin(std::make_shared<semantic_inference::SegmentationNode>(options));
  rclcpp::shutdown();
  return 0;
}
