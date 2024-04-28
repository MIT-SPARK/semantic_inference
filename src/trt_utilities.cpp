#include "trt_utilities.h"

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <map>

#include "semantic_recolor/logging.h"

namespace semantic_recolor {

using nvinfer1::IHostMemory;
using nvinfer1::INetworkDefinition;
using nvinfer1::IRuntime;
using nvinfer1::NetworkDefinitionCreationFlag;
using nvonnxparser::IParser;

Severity stringToSeverity(const std::string& severity) {
  std::map<std::string, Severity> level_map{
      {"INTERNAL_ERROR", Severity::kINTERNAL_ERROR},
      {"ERROR", Severity::kERROR},
      {"WARNING", Severity::kWARNING},
      {"INFO", Severity::kINFO},
      {"VERBOSE", Severity::kVERBOSE}};
  auto iter = level_map.find(severity);
  return iter == level_map.end() ? Severity::kINFO : iter->second;
}

class LoggingShim : public nvinfer1::ILogger {
 public:
  static LoggingShim& instance(const std::string& severity = "INFO") {
    if (!s_instance_) {
      s_instance_.reset(new LoggingShim());
    }

    s_instance_->setSeverity(severity);
    return *s_instance_;
  }

  void log(Severity severity, const char* msg) noexcept override {
    if (severity > min_severity_) {
      return;
    }

    logging::Level level;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        level = logging::Level::FATAL;
        break;
      case Severity::kERROR:
        level = logging::Level::ERROR;
        break;
      case Severity::kWARNING:
        level = logging::Level::WARNING;
        break;
      case Severity::kINFO:
        level = logging::Level::INFO;
        break;
      case Severity::kVERBOSE:
      default:
        level = logging::Level::DEBUG;
        break;
    }

    // disable file-like formatting by specifying negative line number
    logging::LogEntry entry(level, "TensorRT", -1);
    entry << msg;
  }

  void setSeverity(const std::string& severity) {
    min_severity_ = stringToSeverity(severity);
  }

 private:
  Severity min_severity_;
  inline static std::unique_ptr<LoggingShim> s_instance_ = nullptr;
  LoggingShim() : min_severity_(Severity::kINFO) {}
};

void* CudaMemoryManager::alloc(size_t size) {
  void* raw_ptr = nullptr;
  auto error = cudaMalloc(&raw_ptr, size);
  if (error != cudaSuccess) {
    SLOG(ERROR) << "Failed to allocate " << size << " bytes on device";
  }

  return raw_ptr;
}

void CudaMemoryManager::Delete::operator()(void* object) {
  if (object != nullptr) {
    cudaFree(object);
  }
}

Shape Shape::updateFrom(const cv::Mat& input) {
  int new_width = width == -1 ? input.cols : width;
  int new_height = height == -1 ? input.rows : height;
  return {new_width, new_height, channels, chw_order};
}

size_t Shape::numel() const {
  if (width == -1 || height == -1) {
    return 0;
  }

  return width * height * channels.value_or(1);
}

bool Shape::operator==(const Shape& other) const {
  return other.width == width && other.height == height && other.channels == channels;
}

bool ImageMemoryPair::updateShape(const Shape& new_shape) {
  if (new_shape.width == -1 || new_shape.height == -1) {
    SLOG(ERROR) << "Cannot create matrices with unspecified sizes!";
    return false;
  }

  if (shape == new_shape) {
    return true;
  }

  shape = new_shape;

  std::vector<int> dims;
  if (shape.chw_order) {
    dims = {shape.channels.value_or(1), shape.height, shape.width};
  } else {
    dims = {shape.height, shape.width, shape.channels.value_or(1)};
  }

  host_image = cv::Mat(dims, CV_32FC1);
  device_image.reset(reinterpret_cast<float*>(CudaMemoryManager::alloc(size())));
  return true;
}

size_t ImageMemoryPair::size() const { return shape.numel() * sizeof(float); }

std::string toString(const nvinfer1::Dims& dims) {
  std::stringstream ss;
  ss << "[";
  for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
    ss << dims.d[i] << " x ";
  }
  ss << dims.d[dims.nbDims - 1] << "]";
  return ss.str();
}

std::string toString(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kFP8:
      return "FP8";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    case nvinfer1::DataType::kINT4:
      return "INT4";
#endif
    default:
      return "UNKNOWN: '" + std::to_string(static_cast<int>(dtype)) + "'";
  }
}

std::string toString(nvinfer1::TensorIOMode mode) {
  switch (mode) {
    case nvinfer1::TensorIOMode::kINPUT:
      return "in";
    case nvinfer1::TensorIOMode::kOUTPUT:
      return "out";
    default:
    case nvinfer1::TensorIOMode::kNONE:
      return "none";
  }
}

RuntimePtr getRuntime(const std::string& log_severity) {
  auto& logger = LoggingShim::instance(log_severity);
  return RuntimePtr(nvinfer1::createInferRuntime(logger));
}

EnginePtr deserializeEngine(IRuntime& runtime,
                            const std::filesystem::path& engine_path) {
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (engine_file.fail()) {
    SLOG(INFO) << "Engine file: " << engine_path << " not found!";
    return nullptr;
  }

  const auto engine_size = std::filesystem::file_size(engine_path);
  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  EnginePtr engine(runtime.deserializeCudaEngine(engine_data.data(), engine_size));
  if (!engine) {
    SLOG(ERROR) << "Engine deserialization failed for " << engine_path;
    return nullptr;
  }

  return engine;
}

EnginePtr buildEngineFromOnnx(IRuntime& runtime,
                              const std::filesystem::path& model_path,
                              const std::filesystem::path& engine_path,
                              const std::string& verbosity) {
  auto& logger = LoggingShim::instance(verbosity);
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  int flags = 0;
#if NV_TENSORRT_MAJOR < 10
  // required for exported models previously to tensorrt 10
  flags = (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(flags));
  std::unique_ptr<IParser> parser(nvonnxparser::createParser(*net, logger));
  parser->parseFromFile(model_path.c_str(), static_cast<int>(Severity::kVERBOSE));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    SLOG(ERROR) << "Parser Error #" << i << ": " << parser->getError(i)->desc();
  }

  if (net->getNbOutputs() != 1) {
    SLOG(ERROR) << "Network does not have a single output: " << net->getNbOutputs()
                << " != 1";
    return nullptr;
  }

  auto output = net->getOutput(0);
  auto output_dtype = output->getType();
  if (output_dtype != nvinfer1::DataType::kINT32) {
    SLOG(WARNING) << "Add extra output cast layer to INT32";
    auto layer = net->addCast(*output, nvinfer1::DataType::kINT32);
    net->unmarkOutput(*output);
    layer->getOutput(0)->setName("output_int32");
    net->markOutput(*layer->getOutput(0));
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  std::unique_ptr<IHostMemory> memory(builder->buildSerializedNetwork(*net, *config));
  if (!memory) {
    SLOG(FATAL) << "Failed to build network for " << model_path;
    return nullptr;
  }

  std::ofstream fout(engine_path, std::ios::binary);
  fout.write(reinterpret_cast<char*>(memory->data()), memory->size());

  EnginePtr engine(runtime.deserializeCudaEngine(memory->data(), memory->size()));
  if (!engine) {
    SLOG(FATAL) << "Engine creation failed from " << model_path;
    return nullptr;
  }

  return engine;
}

}  // namespace semantic_recolor
