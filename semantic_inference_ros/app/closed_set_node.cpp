#include <config_utilities/parsing/context.h>

#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include "semantic_inference_ros/segmentation_nodelet.h"

int main(int argc, char* argv[]) {
  config::initContext(argc, argv);

  rclcpp::init(argc, argv);

  {  // node lifetime scope
    const rclcpp::NodeOptions options;
    auto node = std::make_shared<semantic_inference::SegmentationNode>(options);
    node->start();
    rclcpp::spin(node);
  }  // end node lifetime scope

  rclcpp::shutdown();
  return 0;
}
