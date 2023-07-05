#include <ros/ros.h>

#include "semantic_recolor/ros_utilities.h"
#include "semantic_recolor/segmenter.h"

using namespace semantic_recolor;

struct DemoConfig {
  DemoConfig(const ros::NodeHandle& nh) {
    if (!nh.getParam("input_file", input_file)) {
      ROS_FATAL("Missing input_file");
      throw std::runtime_error("missing param!");
    }

    if (!nh.getParam("output_file", output_file)) {
      ROS_FATAL("Missing output_file");
      throw std::runtime_error("missing param!");
    }

    nh.getParam("saturation", saturation);
    nh.getParam("luminance", luminance);
    nh.getParam("max_classes", max_classes);
    nh.getParam("num_timing_inferences", num_timing_inferences);
  }

  std::string input_file;
  std::string output_file;
  double saturation = 0.85;
  double luminance = 0.75;
  int max_classes = 150;
  int num_timing_inferences = 10;
};

void outputDemoImage(const DemoConfig& config, const cv::Mat& classes) {
  cv::Mat new_image_hls(classes.rows, classes.cols, CV_32FC3);
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      float* pixel = new_image_hls.ptr<float>(r, c);

      const int remainder = classes.at<int32_t>(r, c) % config.max_classes;
      const double ratio =
          static_cast<double>(remainder) / static_cast<double>(config.max_classes);
      pixel[0] = ratio * 360.0;
      pixel[1] = config.luminance;
      pixel[2] = config.saturation;
    }
  }

  cv::Mat new_image;
  cv::cvtColor(new_image_hls, new_image, cv::COLOR_HLS2BGR);
  new_image *= 255.0;
  cv::imwrite(config.output_file, new_image);
}

void showStatistics(const cv::Mat& classes) {
  std::map<int32_t, size_t> counts;
  std::vector<int32_t> unique_classes;
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      int32_t class_id = classes.at<int32_t>(r, c);
      if (!counts.count(class_id)) {
        counts[class_id] = 0;
        unique_classes.push_back(class_id);
      }

      counts[class_id]++;
    }
  }

  double total = static_cast<double>(classes.rows * classes.cols);
  std::sort(unique_classes.begin(),
            unique_classes.end(),
            [&](const int32_t& lhs, const int32_t& rhs) {
              return counts[lhs] > counts[rhs];
            });

  std::stringstream ss;
  ss << " Class pixel percentages:" << std::endl;
  for (const int32_t id : unique_classes) {
    ss << "  - " << id << ": " << static_cast<double>(counts[id]) / total * 100.0 << "%"
       << std::endl;
  }

  ROS_INFO_STREAM(ss.str());
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle nh("~");
  DemoConfig demo_config(nh);
  const auto config = readModelConfig(nh);

  cv::Mat img = cv::imread(demo_config.input_file);
  if (img.empty()) {
    ROS_FATAL_STREAM("Image not found: " << demo_config.input_file);
    return 1;
  }

  SemanticSegmenter segmenter(config);
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
