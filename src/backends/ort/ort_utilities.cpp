#include "semantic_recolor/backends/ort/ort_utilities.h"

#include <glog/logging.h>

#include <functional>
#include <sstream>

namespace semantic_recolor {

using SizeGetter = std::function<size_t(const Ort::Session*)>;
using NameGetter =
    std::function<Ort::AllocatedStringPtr(const Ort::Session*, size_t, OrtAllocator*)>;
using InfoGetter = std::function<Ort::TypeInfo(const Ort::Session*, size_t)>;

std::vector<FieldInfo> readFieldInfo(const Ort::Session* session,
                                     OrtAllocator* allocator,
                                     const SizeGetter& size_getter,
                                     const NameGetter& name_getter,
                                     const InfoGetter& info_getter) {
  if (!session || !allocator) {
    return {};
  }

  std::vector<FieldInfo> fields;
  const size_t num_fields = size_getter(session);
  for (size_t i = 0; i < num_fields; ++i) {
    auto str_ptr = name_getter(session, i, allocator);
    if (!str_ptr) {
      throw std::domain_error("invalid name for field found!");
    }

    const auto type_info = info_getter(session, i);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto type = tensor_info.GetElementType();

    fields.push_back({std::string(str_ptr.get()),
                      tensor_info.GetShape(),
                      type,
                      getElementTypeString(type)});
  }

  return fields;
}

std::vector<FieldInfo> getSessionInputs(const Ort::Session* session,
                                        OrtAllocator* allocator) {
  return readFieldInfo(session,
                       allocator,
                       &Ort::Session::GetInputCount,
                       &Ort::Session::GetInputNameAllocated,
                       &Ort::Session::GetInputTypeInfo);
}

std::vector<FieldInfo> getSessionOutputs(const Ort::Session* session,
                                         OrtAllocator* allocator) {
  return readFieldInfo(session,
                       allocator,
                       &Ort::Session::GetOutputCount,
                       &Ort::Session::GetOutputNameAllocated,
                       &Ort::Session::GetOutputTypeInfo);
}

std::string getElementTypeString(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return "undefined";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    default:
      return "invalid";
  }
}

std::ostream& operator<<(std::ostream& out, const FieldInfo& info) {
  out << info.name << " -> " << info.type_name << "[";

  auto iter = info.dims.begin();
  while (iter != info.dims.end()) {
    if (*iter == -1) {
      out << "N";
    } else {
      out << *iter;
    }

    ++iter;

    if (iter != info.dims.end()) {
      out << " x ";
    }
  }
  out << "]";
  return out;
}

bool FieldInfo::tensorMatchesType(const cv::Mat& tensor) const {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return tensor.type() == CV_32F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return tensor.type() == CV_8U;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return tensor.type() == CV_8S;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return tensor.type() == CV_16U;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return tensor.type() == CV_16S;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return tensor.type() == CV_32S;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return tensor.type() == CV_64F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    default:
      return false;
  }
}

std::vector<int64_t> getFilteredTensorDims(const cv::Mat& tensor) {
  std::vector<int64_t> filtered;
  for (int i = 0; i < tensor.dims; ++i) {
    if (tensor.size[i] != 1) {
      filtered.push_back(tensor.size[i]);
    }
  }
  return filtered;
}

std::vector<int64_t> FieldInfo::filteredDims() const {
  std::vector<int64_t> filtered;
  for (const auto dim : dims) {
    if (dim != 1) {
      filtered.push_back(dim);
    }
  }
  return filtered;
}

std::string cvDepthToString(int depth) {
  if (depth == CV_8U) {
    return "CV_8U";
  }
  if (depth == CV_8S) {
    return "CV_8S";
  }
  if (depth == CV_16U) {
    return "CV_16U";
  }
  if (depth == CV_16S) {
    return "CV_16S";
  }
  if (depth == CV_32S) {
    return "CV_32S";
  }
  if (depth == CV_32F) {
    return "CV_32F";
  }
  if (depth == CV_64F) {
    return "CV_64F";
  }
  return "Unkown";
}

void FieldInfo::validateTensor(const cv::Mat& tensor) const {
  const auto field_dims = filteredDims();
  const auto tensor_dims = getFilteredTensorDims(tensor);

  if (field_dims.size() != tensor_dims.size()) {
    LOG(ERROR) << "size mismatch for field: " << *this << " vs. " << tensor.size;
    throw std::invalid_argument("tensor rank does not match field rank");
  }

  for (size_t i = 0; i < field_dims.size(); ++i) {
    if (field_dims[i] != tensor_dims[i]) {
      LOG(ERROR) << "size mismatch for field: " << *this << " vs. " << tensor.size;
      throw std::invalid_argument("tensor dimensions do not match field dimensions");
    }
  }
}

template <typename T>
Ort::Value makeOrtTensor(const FieldInfo& field,
                         const OrtMemoryInfo* mem_info,
                         const cv::Mat& tensor) {
  return Ort::Value::CreateTensor<T>(mem_info,
                                     const_cast<T*>(tensor.ptr<T>()),
                                     tensor.total(),
                                     field.dims.data(),
                                     field.dims.size());
}

Ort::Value FieldInfo::makeOrtValue(const OrtMemoryInfo* mem_info,
                                   const cv::Mat& tensor) const {
  if (!mem_info) {
    throw std::domain_error("invalid memory info");
  }

  if (tensor.channels() > 1) {
    throw std::domain_error("multi-channel matrix not supported");
  }

  validateTensor(tensor);
  if (!tensorMatchesType(tensor)) {
    LOG(ERROR) << "type mismatch for field: " << *this << " vs. "
               << cvDepthToString(tensor.depth());
    throw std::invalid_argument("tensor and field type mismatch");
  }

  const auto input_type = tensor.type();
  if (input_type == CV_32F) {
    return makeOrtTensor<float>(*this, mem_info, tensor);
  } else if (input_type == CV_64F) {
    return makeOrtTensor<double>(*this, mem_info, tensor);
  } else if (input_type == CV_8U) {
    return makeOrtTensor<uint8_t>(*this, mem_info, tensor);
  } else if (input_type == CV_16U) {
    return makeOrtTensor<uint16_t>(*this, mem_info, tensor);
  } else if (input_type == CV_8S) {
    return makeOrtTensor<int8_t>(*this, mem_info, tensor);
  } else if (input_type == CV_16S) {
    return makeOrtTensor<int16_t>(*this, mem_info, tensor);
  } else if (input_type == CV_32S) {
    return makeOrtTensor<int32_t>(*this, mem_info, tensor);
  } else {
    throw std::invalid_argument("tensor type is unhandled!");
  }
}

Ort::Value FieldInfo::makeOrtValue(OrtAllocator* allocator) const {
  return Ort::Value::CreateTensor(allocator, dims.data(), dims.size(), type);
}

template <typename LHS, typename RHS>
void convert(const Ort::Value& value, cv::Mat& tensor) {
  auto t_ptr = tensor.ptr<RHS>();
  const auto v_ptr = value.GetTensorData<LHS>();
  for (size_t i = 0; i < tensor.total(); ++i) {
    *(t_ptr + i) = static_cast<RHS>(*(v_ptr + i));
  }
}

void FieldInfo::copyValueToTensor(const Ort::Value& value, cv::Mat& tensor) const {
  if (tensor.channels() > 1) {
    throw std::domain_error("multi-channel matrix not supported");
  }

  validateTensor(tensor);
  if (tensorMatchesType(tensor)) {
    const auto size = tensor.elemSize() * tensor.total();
    std::memcpy(tensor.ptr(), value.GetTensorData<uint8_t>(), size);
    return;
  }

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 && tensor.depth() == CV_32S) {
    convert<uint32_t, int32_t>(value, tensor);
    return;
  }

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && tensor.depth() == CV_32S) {
    convert<int64_t, int32_t>(value, tensor);
    return;
  }

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 && tensor.depth() == CV_32S) {
    convert<uint64_t, int32_t>(value, tensor);
    return;
  }

  LOG(ERROR) << "Cannot convert value for field: " << *this << " to "
             << cvDepthToString(tensor.depth());
  throw std::domain_error("cannot convert types");
}

}  // namespace semantic_recolor
