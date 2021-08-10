#include <onnxruntime_cxx_api.h>
#include <ros/ros.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <memory>
#include <numeric>

cv::Mat makeDebugImg(const cv::Mat &classes,
                     double saturation = 0.85,
                     double luminance = 0.75) {
  double max_class = 0.0;
  double min_class = 0.0;
  cv::minMaxLoc(classes, &min_class, &max_class);
  ROS_INFO_STREAM("Min class: " << min_class << " Max class: " << max_class);

  if (max_class - min_class == 0.0) {
    ROS_WARN_STREAM("Min and max class are the same: " << max_class);
    return cv::Mat::zeros(classes.rows, classes.cols, CV_8UC3);
  }
  const double class_diff = max_class - min_class;

  cv::Mat new_image_hls(classes.rows, classes.cols, CV_32FC3);
  for (int r = 0; r < classes.rows; ++r) {
    for (int c = 0; c < classes.cols; ++c) {
      float *pixel = new_image_hls.ptr<float>(r, c);
      double ratio =
          static_cast<double>(classes.at<int32_t>(r, c) - min_class) / class_diff;
      pixel[0] = ratio * 360.0;
      pixel[1] = luminance;
      pixel[2] = saturation;
    }
  }

  cv::Mat new_image;
  cv::cvtColor(new_image_hls, new_image, cv::COLOR_HLS2BGR);
  cv::imshow("new image", new_image);
  cv::waitKey(0);
  return new_image;
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

cv::Mat getImage(const std::string &input_file, int width, int height) {
  cv::Mat file_img = cv::imread(input_file);
  if (file_img.empty()) {
    ROS_FATAL_STREAM("Image not found: " << input_file);
    return file_img;
  }

  cv::Mat img(file_img.rows, file_img.cols, CV_8UC3);
  ROS_INFO_STREAM("Image: " << file_img.rows << " x " << file_img.cols << " x "
                            << file_img.channels());
  if (file_img.channels() == 3) {
    cv::cvtColor(file_img, img, cv::COLOR_BGR2RGB);
  } else {
    cv::cvtColor(file_img, img, cv::COLOR_BGRA2RGB);
  }
  cv::Mat infer_img;
  cv::resize(img, infer_img, cv::Size(width, height));

  cv::Mat img_float;
  infer_img.convertTo(img_float, CV_32FC3);
  ROS_INFO_STREAM("Image: " << img_float.rows << " x " << img_float.cols << " x "
                            << img_float.channels());

  std::vector<float> mean{0.485f, 0.456f, 0.406f};
  std::vector<float> stddev{0.229f, 0.224f, 0.225f};

  for (int r = 0; r < img_float.rows; ++r) {
    for (int c = 0; c < img_float.cols; ++c) {
      float *pixel = img_float.ptr<float>(r, c);
      for (int channel = 0; channel < 3; ++channel) {
        pixel[channel] =
            ((pixel[channel] / 255.0) - mean.at(channel)) / stddev.at(channel);
      }
    }
  }

  return img_float;
}

template <typename T>
std::string displayVector(const std::vector<T> &vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    ss << vec[i] << ", ";
  }
  if (vec.size()) {
    ss << vec.back();
  }
  ss << "]";
  return ss.str();
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle nh("~");

  std::string model_path;
  nh.getParam("model_path", model_path);

  int width = 640;
  nh.getParam("image_width", width);

  int height = 360;
  nh.getParam("image_height", height);

  std::string input_file = "input.png";
  nh.getParam("input_file", input_file);

  std::string output_file = "output.png";
  nh.getParam("output_file", output_file);

  cv::Mat img = getImage(input_file, width, height);
  if (img.empty()) {
    return 1;
  }

  const char *input_names[] = {"input.1"};
  const char *output_names[] = {"4464"};

  Ort::Env env;
  Ort::SessionOptions options;
  options.SetLogSeverityLevel(1)
      .SetIntraOpNumThreads(4)
      .SetInterOpNumThreads(4)
      .SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, model_path.c_str(), options);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::AllocatorWithDefaultOptions allocator;

  ROS_INFO("IO Names");
  ROS_INFO("========");
  size_t num_in = session.GetInputCount();
  for (size_t i = 0; i < num_in; ++i) {
    ROS_INFO_STREAM(
        "Input #"
        << i << ": " << session.GetInputName(i, allocator) << " @ "
        << displayVector(
               session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
  }

  size_t num_out = session.GetOutputCount();
  for (size_t i = 0; i < num_out; ++i) {
    ROS_INFO_STREAM(
        "Output #"
        << i << ": " << session.GetOutputName(i, allocator) << " @ "
        << displayVector(
               session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
  }

  const size_t input_size = img.rows * img.step[0];
  std::array<int64_t, 4> input_shape{1, 3, height, width};
  Ort::Value input =
      Ort::Value::CreateTensor<float>(memory_info,
                                      reinterpret_cast<float *>(img.data),
                                      input_size,
                                      input_shape.data(),
                                      input_shape.size());

  cv::Mat classes(height, width, CV_32S);

  const size_t output_size = classes.rows * classes.step[0];
  std::array<int64_t, 2> output_shape{height, width};
  Ort::Value output =
      Ort::Value::CreateTensor<int32_t>(memory_info,
                                        reinterpret_cast<int32_t *>(classes.data),
                                        output_size,
                                        output_shape.data(),
                                        output_shape.size());

  // TODO(nathan) runoptions
  Ort::RunOptions run_options;
  session.Run(run_options, input_names, &input, 1, output_names, &output, 1);

  // showStatistics(classes);

  // cv::Mat color_classes = makeDebugImg(classes);
  // cv::imwrite(output_file, color_classes);

  return 0;
}
