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

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/types/path.h>

#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "semantic_inference/image_recolor.h"
#include "semantic_inference/logging.h"
#include "semantic_inference/model_config.h"
#include "semantic_inference/segmenter.h"

using namespace semantic_inference;

struct DemoConfig {
  std::filesystem::path input_file;
  std::filesystem::path output_file;
  double saturation = 0.85;
  double luminance = 0.75;
  int max_classes = 150;
  int num_timing_inferences = 10;
  Segmenter::Config segmenter;
};

void declare_config(DemoConfig& config) {
  using namespace config;
  name("DemoConfig");
  field<Path>(config.input_file, "input_file");
  field<Path>(config.output_file, "output_file");
  field(config.saturation, "saturation");
  field(config.luminance, "luminance");
  field(config.max_classes, "max_classes");
  field(config.num_timing_inferences, "num_timing_inferences");
  field(config.segmenter, "segmenter");
  check<Path::Exists>(config.input_file, "input_file");
}

void outputDemoImage(const DemoConfig& config,
                     const cv::Mat& color,
                     const cv::Mat& labels) {
  const auto recolor =
      ImageRecolor::fromHLS(config.max_classes, config.luminance, config.saturation);

  cv::Mat new_image(color.rows, color.cols, CV_8UC3);
  recolor.recolorImage(labels, new_image);

  std::filesystem::path output_path;
  if (config.output_file.empty()) {
    output_path = config.input_file.parent_path() / config.input_file.stem();
    output_path += "_labels.png";
  } else {
    output_path = config.output_file;
  }

  SLOG(INFO) << "Writing output to " << output_path;
  cv::imwrite(output_path.string(), new_image);
}

int main(int argc, char* argv[]) {
  logging::Logger::addSink("cout",
                           std::make_shared<logging::CoutSink>(logging::Level::INFO));

  if (argc <= 1) {
    SLOG(FATAL) << "Invalid usage! Usage is demo_segmentation CONFIG_PATH";
    return EXIT_FAILURE;
  }

  std::filesystem::path config_path(argv[1]);
  if (!std::filesystem::exists(config_path)) {
    SLOG(FATAL) << "Config path " << config_path << " does not exist!";
    return EXIT_FAILURE;
  }

  const auto config = config::fromYamlFile<DemoConfig>(config_path);
  SLOG(INFO) << "\n" << config::toString(config);

  cv::Mat img = cv::imread(config.input_file.string());
  if (img.empty()) {
    SLOG(FATAL) << "Image not found: " << config.input_file;
    return 1;
  }

  Segmenter segmenter(config.segmenter);
  auto result = segmenter.infer(img);
  if (!result) {
    SLOG(FATAL) << "Failed to run inference";
    return 1;
  }

  const auto start = std::chrono::high_resolution_clock::now();
  size_t num_valid = 0;
  for (int iter = 0; iter < config.num_timing_inferences; ++iter) {
    num_valid += (segmenter.infer(img)) ? 1 : 0;
  }
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end - start;

  double average_period_s =
      elapsed_s.count() / static_cast<double>(config.num_timing_inferences);
  double percent_valid = static_cast<double>(num_valid) /
                         static_cast<double>(config.num_timing_inferences);

  SLOG(INFO) << "Inference took an average of " << average_period_s << " [s] over "
             << config.num_timing_inferences << " total iterations of which "
             << percent_valid * 100.0 << "% were valid";

  SLOG(INFO) << getLabelPercentages(result.labels);
  outputDemoImage(config, img, result.labels);
  return EXIT_SUCCESS;
}
