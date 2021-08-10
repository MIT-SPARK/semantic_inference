#include "semantic_recolor_nodelet/utilities.h"

#include <onnxruntime_cxx_api.h>
#include <ros/ros.h>

#include <fstream>
#include <memory>
#include <numeric>

using namespace semantic_recolor;

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle nh("~");
  TestConfig config = readTestConfig(nh);

  cv::Mat img = getImage(config);

  const char *input_names[] = {"input.1"};
  const char *output_names[] = {"4464"};

  Ort::Env env;
  Ort::SessionOptions options;
  options.SetLogSeverityLevel(1)
      .SetIntraOpNumThreads(4)
      .SetInterOpNumThreads(4)
      .SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, config.model_path.c_str(), options);

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
  std::array<int64_t, 4> input_shape{1, 3, config.height, config.width};
  Ort::Value input =
      Ort::Value::CreateTensor<float>(memory_info,
                                      reinterpret_cast<float *>(img.data),
                                      input_size,
                                      input_shape.data(),
                                      input_shape.size());

  cv::Mat classes(config.height, config.width, CV_32S);

  const size_t output_size = classes.rows * classes.step[0];
  std::array<int64_t, 2> output_shape{config.height, config.width};
  Ort::Value output =
      Ort::Value::CreateTensor<int32_t>(memory_info,
                                        reinterpret_cast<int32_t *>(classes.data),
                                        output_size,
                                        output_shape.data(),
                                        output_shape.size());

  // TODO(nathan) runoptions
  Ort::RunOptions run_options;
  session.Run(run_options, input_names, &input, 1, output_names, &output, 1);

  showStatistics(classes);
  outputDebugImg(config, classes);

  return 0;
}
