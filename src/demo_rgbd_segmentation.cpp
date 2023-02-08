#include <ros/package.h>

#include "semantic_recolor/rgbd_segmenter.h"
#include "semantic_recolor/yaml_utilities.h"

using namespace semantic_recolor;

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

  std::cout << ss.str() << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr
        << "[FATAL]: invalid usage. Example usage: " << std::endl
        << "\tdemo_rgbd_segmentation MODEL_NAME COLOR_CONFIG RGB_IMAGE DEPTH_IMAGE"
        << std::endl;
    return 1;
  }

  const std::string model_name = argv[1];
  const std::string package_path = ros::package::getPath("semantic_recolor");
  const std::string model_config_path = package_path + "/config/" + argv[1] + ".yaml";
  const std::string model_onnx_path = package_path + "/models/" + argv[1] + ".onnx";
  const std::string model_engine_path = package_path + "/engines/" + argv[1] + ".trt";
  const std::string color_config_path = package_path + "/" + argv[2];
  const std::string rgb_path = argv[3];
  const std::string depth_path = argv[4];

  cv::Mat rgb = cv::imread(rgb_path);
  if (rgb.empty()) {
    std::cerr << "[FATAL]: invalid rgb path " << rgb_path << std::endl;
    return 1;
  }

  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  if (depth_path.empty()) {
    std::cerr << "[FATAL]: invalid depth path " << depth_path << std::endl;
    return 1;
  }

  std::cout << "Using rgb: " << rgb.cols << " x " << rgb.rows << " x " << rgb.channels()
            << std::endl;
  std::cout << "Using depth: " << depth.cols << " x " << depth.rows << " x "
            << depth.channels() << std::endl;

  YAML::Node config_node;
  try {
    config_node = YAML::LoadFile(model_config_path);
  } catch (const std::exception& e) {
    std::cerr << "[FATAL]: invalid config @ " << model_config_path << ": " << e.what()
              << std::endl;
    return 1;
  }
  ModelConfig config = readModelConfigFromYaml(config_node);
  config.model_file = model_onnx_path;
  config.engine_file = model_engine_path;
  config.use_ros_logging = false;
  config.log_severity = Severity::kERROR;
  DepthConfig depth_config = readDepthModelConfigFromYaml(config_node);

  printModelConfig(config);
  printDepthModelConfig(depth_config);

  YAML::Node color_config_node;
  try {
    color_config_node = YAML::LoadFile(color_config_path);
  } catch (const std::exception& e) {
    std::cerr << "[FATAL]: invalid color config @ " << color_config_path << ": "
              << e.what() << std::endl;
    return 1;
  }

  SemanticColorConfig color_config = readSemanticColorConfigFromYaml(color_config_node);

  TrtRgbdSegmenter segmenter(config, depth_config);
  if (!segmenter.init()) {
    std::cerr << "[FATAL]: failed to init segmenter!" << std::endl;
    return 1;
  }

  if (!segmenter.infer(rgb, depth)) {
    std::cerr << "[FATAL]: failed to run segmenter!" << std::endl;
    return 1;
  }

  const cv::Mat& classes = segmenter.getClasses();
  showStatistics(classes);

  cv::Mat semantic_image(rgb.rows, rgb.cols, CV_8UC3);
  color_config.fillImage(classes, semantic_image);
  cv::Mat rgb_mat;
  cv::cvtColor(semantic_image, rgb_mat, cv::COLOR_BGR2RGB);
  cv::imwrite("semantic_labels.png", rgb_mat);
  cv::Mat to_save;
  classes.convertTo(to_save, CV_8UC1);
  cv::imwrite("semantic_classes.png", to_save);

  cv::imshow("segmentation", rgb_mat);
  cv::waitKey(0);

  return 0;
}
