#include "semantic_recolor_nodelet/segmenter.h"
#include "semantic_recolor_nodelet/utilities.h"

#include <ros/ros.h>

using namespace semantic_recolor;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle nh("~");
  DemoConfig demo_config(nh);
  SegmentationConfig config = readSegmenterConfig(nh);

  cv::Mat img = cv::imread(demo_config.input_file);
  if (img.empty()) {
    ROS_FATAL_STREAM("Image not found: " << demo_config.input_file);
    return 1;
  }

  TrtSegmenter segmenter(config);
  if (!segmenter.init()) {
    ROS_FATAL("Failed to initialize segmenter");
    return 1;
  }

  ros::WallTime start = ros::WallTime::now();
  if (!segmenter.infer(img)) {
    ROS_FATAL("Failed to run inference");
    return 1;
  }
  ros::WallTime end = ros::WallTime::now();

  ROS_INFO_STREAM("Ran inference in " << (start - end).toSec() << "[s]");

  const cv::Mat& classes = segmenter.getClasses();
  showStatistics(classes);
  outputDemoImage(demo_config, classes);

  return 0;
}
