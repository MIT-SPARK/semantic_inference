#include "semantic_recolor/trt_utilities.h"

#include <ros/ros.h>

namespace semantic_recolor {

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

}  // namespace semantic_recolor
