#include "semantic_recolor/trt_utilities.h"

#include <NvOnnxParser.h>
#include <ros/ros.h>

#include <fstream>

namespace semantic_recolor {

using TrtNetworkDef = nvinfer1::INetworkDefinition;
using nvinfer1::NetworkDefinitionCreationFlag;

inline size_t getFileSize(std::istream &to_check) {
  to_check.seekg(0, to_check.end);
  size_t size = to_check.tellg();
  to_check.seekg(0, to_check.beg);
  return size;
}

void Logger::log(Severity severity, const char *msg) noexcept {
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

std::ostream &operator<<(std::ostream &out, const nvinfer1::Dims &dims) {
  out << "[";
  for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
    out << dims.d[i] << " x ";
  }
  out << dims.d[dims.nbDims - 1] << "]";
  return out;
}

std::ostream &operator<<(std::ostream &out, const nvinfer1::DataType &dtype) {
  out << "[";
  if (dtype == nvinfer1::DataType::kFLOAT) {
    out << "kFLOAT";
  } else if (dtype == nvinfer1::DataType::kHALF) {
    out << "kHALF";
  } else if (dtype == nvinfer1::DataType::kINT8) {
    out << "kINT8";
  } else if (dtype == nvinfer1::DataType::kINT32) {
    out << "kINT32";
  } else {
    out << "UNKNOWN";
  }
  out << "]";
  return out;
}

std::unique_ptr<TrtEngine> deserializeEngine(TrtRuntime &runtime,
                                             const std::string &engine_path) {
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (engine_file.fail()) {
    ROS_INFO_STREAM("Engine file: " << engine_path << " not found!");
    return nullptr;
  }

  const size_t engine_size = getFileSize(engine_file);
  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  std::unique_ptr<TrtEngine> engine(
      runtime.deserializeCudaEngine(engine_data.data(), engine_size));
  if (!engine) {
    ROS_FATAL_STREAM("Engine creation failed");
    return nullptr;
  }

  return engine;
}

std::unique_ptr<TrtEngine> buildEngineFromOnnx(TrtRuntime &runtime,
                                               Logger &logger,
                                               const std::string &model_path,
                                               const std::string &engine_path) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  const auto network_flags =
      1u << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<TrtNetworkDef> network(builder->createNetworkV2(network_flags));

  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));
  parser->parseFromFile(model_path.c_str(), static_cast<int>(Severity::kINFO));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    ROS_ERROR_STREAM("Parser Error #" << i << ": " << parser->getError(i)->desc());
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

  std::unique_ptr<nvinfer1::IHostMemory> memory(
      builder->buildSerializedNetwork(*network, *config));
  if (!memory) {
    ROS_FATAL_STREAM("Failed to build network");
    return nullptr;
  }

  std::ofstream fout(engine_path, std::ios::binary);
  fout.write(reinterpret_cast<char *>(memory->data()), memory->size());

  std::unique_ptr<TrtEngine> engine(
      runtime.deserializeCudaEngine(memory->data(), memory->size()));
  if (!engine) {
    ROS_FATAL_STREAM("Engine creation failed");
    return nullptr;
  }
  return engine;
}

}  // namespace semantic_recolor
