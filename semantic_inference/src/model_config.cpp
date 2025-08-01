/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * -------------------------------------------------------------------------- */

#include "semantic_inference/model_config.h"

#include <config_utilities/config.h>
#include <config_utilities/types/path.h>

#include <cstdlib>

namespace semantic_inference {

std::filesystem::path getModelDirectory() {
  const auto env_path = std::getenv("SEMANTIC_INFERENCE_MODEL_DIR");
  if (env_path) {
    return std::filesystem::absolute(std::filesystem::path(env_path));
  }

  const auto home_path = std::getenv("HOME");
  if (!home_path) {
    throw std::runtime_error(
        "cannot infer default model location via HOME! Please set "
        "SEMANTIC_INFERENCE_MODEL_DIR instead!");
  }

  return std::filesystem::path(home_path) / ".semantic_inference";
}

std::filesystem::path ModelConfig::model_path() const {
  return getModelDirectory() / model_file;
}

std::filesystem::path ModelConfig::engine_path() const {
  return model_path().replace_extension(".trt");
}

void declare_config(ModelConfig& config) {
  using namespace config;
  name("ModelConfig");
  // params
  field<Path>(config.model_file, "model_file");
  field(config.log_severity, "log_severity");
  field(config.force_rebuild, "force_rebuild");
  field(config.color, "color");
  field(config.depth, "depth");
  // checks
  check<Path::Exists>(config.model_path(), "model_file");
  checkIsOneOf(config.log_severity,
               {"INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE"},
               "log_severity");
}

}  // namespace semantic_inference
