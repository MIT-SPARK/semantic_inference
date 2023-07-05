#include "semantic_recolor/backends/ort/ort_backend.h"

#include "semantic_recolor/backends/ort/ort_utilities.h"

namespace semantic_recolor {

class OrtBackendImpl {
 public:
  explicit OrtBackendImpl()
      : mem_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                             OrtMemType::OrtMemTypeDefault)) {
    allocator_.reset(new Ort::AllocatorWithDefaultOptions());
  }

  void init(const ModelConfig& config, const std::string& model_path) {
    env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort_backend"));

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1);

    session_.reset(new Ort::Session(*env_, model_path.c_str(), options));

    // nn_img_ = cv::Mat(config_.getInputMatDims(3), CV_32FC1);
    // classes_ = cv::Mat(config_.height, config_.width, CV_32S);
  }

  void run(const cv::Mat& input, cv::Mat& output) const {
    run({{input_names_.front(), input}}, output);
  }

  void run(const std::map<std::string, cv::Mat>& inputs, cv::Mat& output) const {
    std::vector<Ort::Value> input_values;
    for (const auto& field : input_fields_) {
      const auto iter = inputs.find(field.name);
      if (iter == inputs.end()) {
        std::stringstream ss;
        ss << "missing input " << field.name << " from provided input tensors!";
        throw std::invalid_argument(ss.str());
      }

      input_values.push_back(field.makeOrtValue(mem_info_, iter->second));
    }

    std::vector<Ort::Value> output_values;
    output_values.push_back(output_field_.makeOrtValue(mem_info_, output));

    session_->Run(Ort::RunOptions(nullptr),
                  input_names_.data(),
                  input_values.data(),
                  input_names_.size(),
                  &output_name_,
                  output_values.data(),
                  1);
  }

 private:
  std::vector<const char*> input_names_;
  std::vector<FieldInfo> input_fields_;

  const char* output_name_;
  FieldInfo output_field_;

  Ort::MemoryInfo mem_info_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
};

OrtBackend::OrtBackend() { impl_ = std::make_unique<OrtBackendImpl>(); }

void OrtBackend::init(const ModelConfig& config, const std::string& model_path) {
  impl_->init(config, model_path);
}

void OrtBackend::run(const cv::Mat& input, cv::Mat& output) const {
  impl_->run(input, output);
}

}  // namespace semantic_recolor
