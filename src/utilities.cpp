#include "semantic_recolor_nodelet/utilities.h"

#include <opencv2/imgproc.hpp>

namespace semantic_recolor {

SegmentationConfig readSegmenterConfig(const ros::NodeHandle &nh) {
  SegmentationConfig config;

  if (!nh.getParam("model_file", config.model_file)) {
    ROS_FATAL("Missing model_file");
    throw std::runtime_error("missing param!");
  }

  nh.getParam("engine_file", config.engine_file);
  nh.getParam("width", config.width);
  nh.getParam("height", config.height);
  nh.getParam("input_name", config.input_name);
  nh.getParam("output_name", config.output_name);
  nh.getParam("mean", config.mean);
  nh.getParam("stddev", config.stddev);

  return config;
}

void outputDemoImage(const DemoConfig &config, const cv::Mat &classes) {
  cv::Mat new_image_hls(classes.rows, classes.cols, CV_32FC3);
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      float *pixel = new_image_hls.ptr<float>(r, c);

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

void showStatistics(const cv::Mat &classes) {
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
            [&](const int32_t &lhs, const int32_t &rhs) {
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

void fillNetworkImage(const SegmentationConfig &cfg,
                      const cv::Mat &input,
                      cv::Mat &output) {
  cv::Mat img;
  cv::resize(input, img, cv::Size(cfg.width, cfg.height));

  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      const uint8_t *pixel = img.ptr<uint8_t>(row, col);
      output.at<float>(0, row, col) =
          (static_cast<float>(pixel[2]) / 255.0f - cfg.mean[0]) / cfg.stddev[0];
      output.at<float>(1, row, col) =
          (static_cast<float>(pixel[1]) / 255.0f - cfg.mean[1]) / cfg.stddev[1];
      output.at<float>(2, row, col) =
          (static_cast<float>(pixel[0]) / 255.0f - cfg.mean[2]) / cfg.stddev[2];
    }
  }
}

}  // namespace semantic_recolor
