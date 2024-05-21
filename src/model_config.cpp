#include "semantic_recolor/model_config.h"

#include <config_utilities/config.h>
#include <config_utilities/types/path.h>

namespace semantic_recolor {

void declare_config(ModelConfig& config) {
  using namespace config;
  name("ModelConfig");
  // params
  field<Path>(config.model_file, "model_file");
  field<Path>(config.engine_file, "engine_file");
  field(config.log_severity, "log_severity");
  field(config.force_rebuild, "force_rebuild");
  field(config.color, "color");
  field(config.depth, "depth");
  // checks
  check<Path::Exists>(config.model_file, "model_file");
  checkIsOneOf(config.log_severity,
               {"INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE"},
               "log_severity");
}

}  // namespace semantic_recolor
