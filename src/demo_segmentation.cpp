#include "semantic_recolor/segmenter.h"
#include "semantic_recolor/utilities.h"

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

  if (!segmenter.infer(img)) {
    ROS_FATAL("Failed to run inference");
    return 1;
  }

  ros::WallTime start = ros::WallTime::now();
  size_t num_valid = 0;
  for (int iter = 0; iter < demo_config.num_timing_inferences; ++iter) {
    num_valid += (segmenter.infer(img)) ? 1 : 0;
  }
  ros::WallTime end = ros::WallTime::now();

  double average_period_s =
      (end - start).toSec() / static_cast<double>(demo_config.num_timing_inferences);
  double percent_valid = static_cast<double>(num_valid) /
                         static_cast<double>(demo_config.num_timing_inferences);

  ROS_INFO_STREAM("Inference took an average of "
                  << average_period_s << " [s] over "
                  << demo_config.num_timing_inferences << " total iterations of which "
                  << percent_valid * 100.0 << "% were valid");

  const cv::Mat& classes = segmenter.getClasses();
  showStatistics(classes);
  outputDemoImage(demo_config, classes);

  return 0;
}
