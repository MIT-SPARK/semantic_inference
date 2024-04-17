#include "semantic_recolor/trt_utilities.h"

#include <NvOnnxParser.h>
#include <ros/ros.h>

#include <fstream>

namespace semantic_recolor {

using TrtNetworkDef = nvinfer1::INetworkDefinition;
using nvinfer1::NetworkDefinitionCreationFlag;

inline size_t getFileSize(std::istream& to_check) {
  to_check.seekg(0, to_check.end);
  size_t size = to_check.tellg();
  to_check.seekg(0, to_check.beg);
  return size;
}

void Logger::log(Severity severity, const char* msg) noexcept {
  if (severity < min_severity_) {
    return;
  }

  if (!use_ros_) {
    std::cerr << msg << std::endl;
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

std::ostream& operator<<(std::ostream& out, const nvinfer1::Dims& dims) {
  out << "[";
  for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
    out << dims.d[i] << " x ";
  }
  out << dims.d[dims.nbDims - 1] << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, const nvinfer1::DataType& dtype) {
  out << "[";
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      out << "kFLOAT";
      break;
    case nvinfer1::DataType::kHALF:
      out << "kHALF";
      break;
    case nvinfer1::DataType::kINT8:
      out << "kINT8";
      break;
    case nvinfer1::DataType::kINT32:
      out << "kINT32";
      break;
    case nvinfer1::DataType::kBOOL:
      out << "kBOOL";
      break;
    case nvinfer1::DataType::kUINT8:
      out << "kUINT8";
      break;
    case nvinfer1::DataType::kFP8:
      out << "kFP8";
      break;
    default:
      out << "UNKNOWN: '" << static_cast<int>(dtype) << "'";
      break;
  }
  out << "]";
  return out;
}

std::unique_ptr<TrtEngine> deserializeEngine(TrtRuntime& runtime,
                                             const std::string& engine_path) {
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

std::unique_ptr<TrtEngine> buildEngineFromOnnx(TrtRuntime& runtime,
                                               Logger& logger,
                                               const std::string& model_path,
                                               const std::string& engine_path,
                                               bool set_builder_flags) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  std::unique_ptr<TrtNetworkDef> network(builder->createNetworkV2(0));

  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));
  parser->parseFromFile(model_path.c_str(), static_cast<int>(Severity::kINFO));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    ROS_ERROR_STREAM("Parser Error #" << i << ": " << parser->getError(i)->desc());
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (set_builder_flags) {
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
    config->setFlag(nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
  }

  std::unique_ptr<nvinfer1::IHostMemory> memory(
      builder->buildSerializedNetwork(*network, *config));
  if (!memory) {
    ROS_FATAL_STREAM("Failed to build network");
    return nullptr;
  }

  std::ofstream fout(engine_path, std::ios::binary);
  fout.write(reinterpret_cast<char*>(memory->data()), memory->size());

  std::unique_ptr<TrtEngine> engine(
      runtime.deserializeCudaEngine(memory->data(), memory->size()));
  if (!engine) {
    ROS_FATAL_STREAM("Engine creation failed");
    return nullptr;
  }
  return engine;
}

}  // namespace semantic_recolor
