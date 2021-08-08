#include <ros/ros.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <memory>
#include <numeric>

using nvinfer1::ILogger;
using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger {
 public:
  explicit Logger(Severity severity = Severity::kINFO) : min_severity_(severity) {}

  void log(Severity severity, const char *msg) noexcept override {
    if (severity < min_severity_) {
      return;
    }

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        ROS_FATAL_STREAM(msg);
        break;
      case Severity::kERROR:
        ROS_ERROR_STREAM(msg);
        break;
      case Severity::kWARNING:
        ROS_WARN_STREAM(msg);
        break;
      case Severity::kINFO:
        ROS_INFO_STREAM(msg);
        break;
      case Severity::kVERBOSE:
      default:
        ROS_DEBUG_STREAM(msg);
        break;
    }
  }

 private:
  Severity min_severity_;
};

using TrtRuntime = nvinfer1::IRuntime;
using TrtEngine = nvinfer1::ICudaEngine;
using TrtContext = nvinfer1::IExecutionContext;

template <typename T>
struct CudaMemoryHolder {
  explicit CudaMemoryHolder(const nvinfer1::Dims &dims) : dims(dims) {
    size =
        std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) *
        sizeof(T);

    void *raw_memory = nullptr;
    if (cudaMalloc(&raw_memory, size) != cudaSuccess) {
      return;
    }

    memory.reset(reinterpret_cast<T *>(raw_memory));
  }

  struct Deleter {
    void operator()(T *object) {
      if (object != nullptr) {
        cudaFree(object);
      }
    }
  };

  std::unique_ptr<T, Deleter> memory;
  size_t size;
  nvinfer1::Dims dims;
};

size_t getFileSize(std::istream &to_check) {
  to_check.seekg(0, std::istream::end);
  size_t size = to_check.tellg();
  to_check.seekg(0, std::ifstream::beg);
  return size;
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle nh("~");

  std::string model;
  nh.getParam("model_file", model);

  int width;
  nh.getParam("image_width", width);

  int height;
  nh.getParam("image_height", height);

  std::string input_file = "input.png";
  nh.getParam("input_image", input_file);

  std::string output_file = "output.png";
  nh.getParam("output_image", output_file);

  std::ifstream engine_file(model, std::ios::binary);
  if (engine_file.fail()) {
    ROS_FATAL_STREAM("Model file: " << model << " not found!");
    return 1;
  }

  const size_t engine_size = getFileSize(engine_file);
  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  Logger logger(Severity::kVERBOSE);
  std::unique_ptr<TrtRuntime> trt_runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<TrtEngine> engine(
      trt_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
  if (!engine) {
    ROS_FATAL_STREAM("Runtime creation failed");
    return 1;
  }

  std::unique_ptr<TrtContext> context(engine->createExecutionContext());
  if (!context) {
    ROS_FATAL_STREAM("Failed to create execution context");
    return 1;
  }

  auto input_idx = engine->getBindingIndex("input");
  if (input_idx == -1) {
    ROS_FATAL_STREAM("Failed to get input index");
    return 1;
  }

  nvinfer1::Dims4 input_dims{1, 3, height, width};
  context->setBindingDimensions(input_idx, input_dims);

  auto output_idx = engine->getBindingIndex("output");
  if (output_idx == -1) {
    ROS_FATAL_STREAM("Failed to get input index");
    return 1;
  }

  auto output_dims = context->getBindingDimensions(output_idx);

  if (engine->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
    ROS_FATAL_STREAM("Input type doesn't match expected: "
                     << static_cast<int32_t>(engine->getBindingDataType(input_idx))
                     << " != " << static_cast<int32_t>(nvinfer1::DataType::kFLOAT));
  }

  if (engine->getBindingDataType(output_idx) != nvinfer1::DataType::kINT32) {
    ROS_FATAL_STREAM("Output type doesn't match expected: "
                     << static_cast<int32_t>(engine->getBindingDataType(output_idx))
                     << " != " << static_cast<int32_t>(nvinfer1::DataType::kINT32));
  }

  CudaMemoryHolder<float> input(input_dims);
  CudaMemoryHolder<int32_t> output(output_dims);

  cv::Mat img = cv::imread(input_file);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::Mat img_float;
  img.convertTo(img_float, CV_32FC1);
  img_float /= 255.0;

  std::vector<float> mean{0.485, 0.456, 0.406};
  std::vector<float> stddev{0.229, 0.224, 0.225};
  for (int r = 0; r < img_float.rows; ++r) {
    for (int c = 0; c < img_float.cols; ++c) {
      img_float.at<float>(r, c, 0) =
          (img_float.at<float>(r, c, 0) - mean[0]) / stddev[0];
      img_float.at<float>(r, c, 1) =
          (img_float.at<float>(r, c, 1) - mean[1]) / stddev[1];
      img_float.at<float>(r, c, 2) =
          (img_float.at<float>(r, c, 2) - mean[2]) / stddev[2];
    }
  }

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    ROS_FATAL_STREAM("Creating cuda stream failed!");
    return 1;
  }

  if (cudaMemcpyAsync(input.memory.get(),
                      img_float.data,
                      input.size,
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    ROS_FATAL_STREAM("Failed to copy image to gpu!");
    return 1;
  }

  void *bindings[] = {input.memory.get(), output.memory.get()};
  bool status = context->enqueueV2(bindings, stream, nullptr);
  if (!status) {
    ROS_FATAL_STREAM("Inference failed!");
    return 1;
  }

  std::vector<int> classes(output.size / sizeof(int32_t));
  if (cudaMemcpyAsync(classes.data(),
                      output.memory.get(),
                      classes.size() * sizeof(int32_t),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess) {
    ROS_FATAL_STREAM("Copying output failed");
  }

  cudaStreamSynchronize(stream);

  // TODO(nathan) output an image and report stats

  return 0;
}
