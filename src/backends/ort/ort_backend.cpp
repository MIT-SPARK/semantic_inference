#include "semantic_recolor/backends/ort/ort_backend.h"

#include <glog/logging.h>

#include "semantic_recolor/backends/ort/ort_utilities.h"

namespace semantic_recolor {

struct GlogSingleton {
  static GlogSingleton& instance() {
    if (!instance_) {
      instance_.reset(new GlogSingleton());
    }
    return *instance_;
  }

  ~GlogSingleton() = default;

  void setLogLevel(int log_level, int verbosity = 0) {
    FLAGS_minloglevel = log_level;
    FLAGS_v = verbosity;
  }

 private:
  GlogSingleton() {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging("semantic_recolor");
    google::InstallFailureSignalHandler();
  }

  static std::unique_ptr<GlogSingleton> instance_;
};

std::unique_ptr<GlogSingleton> GlogSingleton::instance_;

class OrtBackendImpl {
 public:
  explicit OrtBackendImpl()
      : mem_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                             OrtMemType::OrtMemTypeDefault)) {
    allocator_.reset(new Ort::AllocatorWithDefaultOptions());
    GlogSingleton::instance().setLogLevel(0, 0);
  }

  bool init(const ModelConfig& config, const std::string& model_path) {
    env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort_backend"));

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1);
    session_.reset(new Ort::Session(*env_, model_path.c_str(), options));

    input_fields_ = getSessionInputs(session_.get(), *allocator_);
    for (const auto& field : input_fields_) {
      input_names_.push_back(field.name.c_str());
    }

    const auto outputs = getSessionOutputs(session_.get(), *allocator_);
    if (outputs.size() != 1) {
      LOG(ERROR) << "Model does not have a single output!";
      return false;
    }

    output_field_ = outputs.at(0);
    output_name_ = output_field_.name.c_str();

    LOG(INFO) << "Loaded model from " << model_path;
    std::stringstream ss;
    ss << "Model inputs:" << std::endl;
    for (const auto& input : input_fields_) {
      ss << " - " << input << std::endl;
    }
    LOG(INFO) << ss.str();
    LOG(INFO) << "Model output: " << output_field_;
    return true;
  }

  bool run(const cv::Mat& input, cv::Mat& output) const {
    CHECK_GT(input_names_.size(), 0) << "no input names parsed";
    return run({{input_names_.front(), input}}, output);
  }

  bool run(const std::map<std::string, cv::Mat>& inputs, cv::Mat& output) const {
    std::vector<Ort::Value> input_values;
    for (const auto& field : input_fields_) {
      const auto iter = inputs.find(field.name);
      if (iter == inputs.end()) {
        LOG(ERROR) << "missing input " << field.name << " from provided input tensors!";
        return false;
      }

      input_values.push_back(field.makeOrtValue(mem_info_, iter->second));
    }

    Ort::Value output_value = output_field_.makeOrtValue(*allocator_);
    session_->Run(Ort::RunOptions(nullptr),
                  input_names_.data(),
                  input_values.data(),
                  input_names_.size(),
                  &output_name_,
                  &output_value,
                  1);

    output_field_.copyValueToTensor(output_value, output);
    return true;
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

bool OrtBackend::init(const ModelConfig& config, const std::string& model_path) {
  return impl_->init(config, model_path);
}

bool OrtBackend::run(const cv::Mat& input, cv::Mat& output) const {
  return impl_->run(input, output);
}

}  // namespace semantic_recolor
