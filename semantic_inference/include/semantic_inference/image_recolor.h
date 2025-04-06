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

#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace semantic_inference {

using Label = int16_t;

struct GroupInfo {
  std::string name;
  std::vector<Label> labels;
};

class ImageRecolor {
 public:
  struct Config {
    std::vector<GroupInfo> groups;
    std::vector<uint8_t> default_color{0, 0, 0};
    Label default_id = -1;
    Label offset = 0;
    std::filesystem::path colormap_path;
    bool skip_header = true;
    char delimiter = ',';
  } const config;

  explicit ImageRecolor(const Config& config,
                        const std::map<Label, std::array<uint8_t, 3>>& colormap = {});

  static ImageRecolor fromHLS(size_t num_classes,
                              float luminance = 0.8,
                              float saturation = 0.8);

  void recolorImage(const cv::Mat& classes, cv::Mat& output) const;

  void relabelImage(const cv::Mat& classes, cv::Mat& output) const;

  const std::array<uint8_t, 3>& getColor(Label label) const;

  const std::array<uint8_t, 3> default_color;
  const std::map<Label, Label> label_remapping;
  const std::map<Label, std::array<uint8_t, 3>> color_map;

 protected:
  void fillColor(Label class_id, uint8_t* pixel, size_t pixel_size = 3) const;

  Label getRemappedLabel(Label class_id) const;

 private:
  mutable std::set<Label> seen_unknown_labels_;
};

void declare_config(ImageRecolor::Config& config);

}  // namespace semantic_inference
