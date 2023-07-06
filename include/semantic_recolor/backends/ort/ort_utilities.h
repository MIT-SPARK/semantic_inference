#pragma once
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace semantic_recolor {

struct FieldInfo {
  std::string name;
  std::vector<int64_t> dims;
  ONNXTensorElementDataType type;
  std::string type_name;

  Ort::Value makeOrtValue(const OrtMemoryInfo* mem_info, const cv::Mat& tensor) const;

  void validateTensor(const cv::Mat& tensor) const;

  bool tensorMatchesType(const cv::Mat& tensor) const;

  std::vector<int64_t> filteredDims() const;
};

std::vector<FieldInfo> getSessionInputs(const Ort::Session* session,
                                        OrtAllocator* allocator);

std::vector<FieldInfo> getSessionOutputs(const Ort::Session* session,
                                         OrtAllocator* allocator);

std::string getElementTypeString(ONNXTensorElementDataType type);

std::ostream& operator<<(std::ostream& out, const FieldInfo& info);

}  // namespace semantic_recolor
