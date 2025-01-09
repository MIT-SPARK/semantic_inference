#include <config_utilities/parsing/commandline.h>

#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include "semantic_inference_ros/segmentation_nodelet.h"

int main(int argc, char* argv[]) {
  config::Settings().print_width = 300;
  config::initContext(argc, argv);

  rclcpp::init(argc, argv);

  const rclcpp::NodeOptions options;
  auto node = std::make_shared<semantic_inference::SegmentationNode>(options);
  node->start();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
