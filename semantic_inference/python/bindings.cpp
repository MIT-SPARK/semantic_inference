#include <config_utilities/parsing/yaml.h>
#include <pybind11/pybind11.h>
#include <semantic_inference/segmenter.h>

namespace semantic_inference::python {

namespace py = pybind11;

PYBIND11_MODULE(_semantic_inference_bindings, m) {
  py::class_<ColorConverter::Config>(m, "ColorConfig")
      .def_readwrite("mean", &ColorConverter::Config::mean)
      .def_readwrite("stddev", &ColorConverter::Config::stddev)
      .def_readwrite("map_to_unit_range", &ColorConverter::Config::map_to_unit_range)
      .def_readwrite("normalize", &ColorConverter::Config::normalize)
      .def_readwrite("rgb_order", &ColorConverter::Config::rgb_order);

  py::class_<ModelConfig>(m, "ModelConfig")
      .def_readwrite("log_severity", &ModelConfig::log_severity)
      .def_readwrite("force_rebuild", &ModelConfig::force_rebuild)
      .def_readwrite("min_optimization_size", &ModelConfig::min_optimization_size)
      .def_readwrite("max_optimization_size", &ModelConfig::max_optimization_size)
      .def_readwrite("target_optimization_size", &ModelConfig::target_optimization_size)
      .def_readwrite("color", &ModelConfig::color)
      .def_static("from_file", [](const std::filesystem::path& filepath) {
        return config::fromYamlFile<ModelConfig>(filepath);
      });

  py::class_<Segmenter>(m, "Segmenter")
      .def(py::init(
          [](const std::filesystem::path& model_file, const ModelConfig& config) {
            Segmenter::Config conf;
            conf.model = config;
            conf.model.model_file = model_file;
            return std::make_unique<Segmenter>(conf);
          }));
}

}  // namespace semantic_inference::python
