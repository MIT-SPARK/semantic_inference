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

  if (!tensorMatchesType(tensor)) {
    LOG(ERROR) << "type mismatch for field: " << *this << " vs. "
               << cvDepthToString(tensor.depth());
    throw std::invalid_argument("tensor and field type mismatch");
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

/*std::optional<Tensor::Type> OrtToTensorType(ONNXTensorElementDataType type) {*/
/*switch (type) {*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:*/
/*return Tensor::Type::FLOAT32;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:*/
/*return Tensor::Type::UINT8;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:*/
/*return Tensor::Type::INT8;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:*/
/*return Tensor::Type::UINT16;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:*/
/*return Tensor::Type::INT16;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:*/
/*return Tensor::Type::INT32;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:*/
/*return Tensor::Type::INT64;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:*/
/*return Tensor::Type::FLOAT64;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:*/
/*return Tensor::Type::UINT32;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:*/
/*return Tensor::Type::UINT64;*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:*/
/*case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:*/
/*default:*/
/*return std::nullopt;*/
/*}*/
/*}*/

/*Tensor FieldInfo::getTensor() const {*/
/*for (const auto dim_size : dims) {*/
/*if (dim_size == -1) {*/
/*std::stringstream ss;*/
/*ss << "cannot make a tensor with dynamic dimensions"*/
/*<< " for field " << name;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/
/*}*/

/*auto tensor_type = OrtToTensorType(type);*/
/*if (!tensor_type) {*/
/*std::stringstream ss;*/
/*ss << "cannot make tensor of corresponding type to " << type_name << " for field "*/
/*<< name;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/

/*return Tensor(dims, *tensor_type);*/
/*}*/

/*Tensor FieldInfo::getDynamicTensor(const std::vector<size_t>& dims_to_read,*/
/*const std::vector<int64_t>& output_dims) const {*/
/*size_t dynamic_index = 0;*/
/*std::vector<int64_t> new_dims;*/
/*for (size_t i = 0; i < dims.size(); ++i) {*/
/*if (dims[i] != -1) {*/
/*new_dims.push_back(dims[i]);*/
/*continue;*/
/*}*/

/*if (dynamic_index >= dims_to_read.size()) {*/
/*std::stringstream ss;*/
/*ss << "dynamic size required for axis " << i << " for field " << name;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/

/*const auto output_index = dims_to_read[dynamic_index];*/
/*++dynamic_index;*/
/*if (output_index >= output_dims.size()) {*/
/*std::stringstream ss;*/
/*ss << "output dimensions missing index " << output_index << " for field " << name;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/

/*const auto new_dim = output_dims[output_index];*/
/*if (new_dim < 0) {*/
/*std::stringstream ss;*/
/*ss << "invalid output dimension at index " << output_index << ": " << new_dim;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/

/*new_dims.push_back(new_dim);*/
/*}*/

/*auto tensor_type = OrtToTensorType(type);*/
/*if (!tensor_type) {*/
/*std::stringstream ss;*/
/*ss << "cannot make tensor of corresponding type to " << type_name << " for field "*/
/*<< name;*/
/*throw std::invalid_argument(ss.str());*/
/*}*/

/*return Tensor(new_dims, *tensor_type);*/
/*}*/

}  // namespace semantic_recolor
