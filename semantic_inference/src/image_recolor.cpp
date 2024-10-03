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

#include "semantic_inference/image_recolor.h"

#include <config_utilities/config.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>

#include <fstream>
#include <opencv2/imgproc.hpp>

#include "semantic_inference/logging.h"

namespace semantic_inference {

namespace fs = std::filesystem;

std::string vecToString(const std::vector<std::string>& vec) {
  std::stringstream ss;
  ss << "[";

  auto iter = vec.begin();
  while (iter != vec.end()) {
    ss << *iter;
    ++iter;
    if (iter != vec.end()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::map<int16_t, std::array<uint8_t, 3>> loadColormap(const fs::path& filepath,
                                                       bool skip_first = true,
                                                       char delimiter = ',') {
  std::ifstream file(filepath.string());
  if (!file.good()) {
    SLOG(ERROR) << "Couldn't open file: " << filepath;
    return {};
  }

  std::map<int16_t, std::array<uint8_t, 3>> cmap;
  size_t row_number = 0;
  std::string curr_line;
  while (std::getline(file, curr_line)) {
    if (skip_first && !row_number) {
      row_number++;
      continue;
    }

    std::stringstream ss(curr_line);
    std::vector<std::string> columns;
    std::string column;
    while (std::getline(ss, column, delimiter)) {
      columns.push_back(column);
      column = "";
    }

    if (columns.size() != 6) {
      SLOG(ERROR) << "Row " << row_number << " is invalid: [" << vecToString(columns)
                  << "]";
      continue;
    }

    // We expect the CSV to have header: name, red, green, blue, alpha, id
    const uint8_t r = std::atoi(columns[1].c_str());
    const uint8_t g = std::atoi(columns[2].c_str());
    const uint8_t b = std::atoi(columns[3].c_str());
    const int16_t id = std::atoi(columns[5].c_str());
    cmap[id] = {r, g, b};
    row_number++;
  }

  return cmap;
}

std::array<uint8_t, 3> getColorFromHLS(float ratio, float luminance, float saturation) {
  cv::Mat hls(1, 1, CV_32FC3);
  hls.at<float>(0) = 360.0 * ratio;
  hls.at<float>(1) = luminance;
  hls.at<float>(2) = saturation;

  cv::Mat bgr;
  cv::cvtColor(hls, bgr, cv::COLOR_HLS2BGR);
  return {static_cast<uint8_t>(255 * bgr.at<float>(2)),
          static_cast<uint8_t>(255 * bgr.at<float>(1)),
          static_cast<uint8_t>(255 * bgr.at<float>(0))};
}

ImageRecolor::ImageRecolor(const Config& config,
                           const std::map<int16_t, std::array<uint8_t, 3>>& colormap)
    : config(config::checkValid(config)), color_map_(colormap) {
  if (std::filesystem::exists(config.colormap_path)) {
    color_map_ = loadColormap(config.colormap_path);
  }

  const auto num_classes = config.groups.size() + 1;
  for (size_t i = 0; i < config.groups.size(); ++i) {
    const auto& group = config.groups[i];
    auto iter = color_map_.find(i);
    if (iter == color_map_.end()) {
      SLOG(WARNING) << "Missing color for group '" << group.name << "'";
      color_map_[i] = getColorFromHLS(static_cast<float>(i) / num_classes, 0.7, 0.7);
    }

    for (const auto& label : group.labels) {
      label_remapping_[label + config.offset] = i;
    }
  }
}

ImageRecolor ImageRecolor::fromHLS(int16_t num_classes,
                                   float luminance,
                                   float saturation) {
  Config config;
  std::map<int16_t, std::array<uint8_t, 3>> colormap;
  for (int16_t i = 0; i < num_classes; ++i) {
    GroupInfo info;
    info.name = "group_" + std::to_string(i);
    info.labels = {i};

    config.groups.push_back(info);
    colormap[i] = getColorFromHLS(
        static_cast<float>(i) / static_cast<float>(num_classes), luminance, saturation);
  }

  return ImageRecolor(config, colormap);
}

void ImageRecolor::relabelImage(const cv::Mat& classes, cv::Mat& output) const {
  if (output.type() != CV_16S) {
    return;
  }

  // opencv doesn't allow resizing of 32S images...
  cv::Mat resized_classes;
  classes.convertTo(resized_classes, CV_16S);
  if (classes.rows != output.rows || classes.cols != output.cols) {
    // interpolating class labels doesn't make sense so use NEAREST
    cv::resize(resized_classes,
               resized_classes,
               cv::Size(output.cols, output.rows),
               0.0f,
               0.0f,
               cv::INTER_NEAREST);
  }

  for (int r = 0; r < resized_classes.rows; ++r) {
    for (int c = 0; c < resized_classes.cols; ++c) {
      int16_t* pixel = output.ptr<int16_t>(r, c);
      const auto class_id = resized_classes.at<int16_t>(r, c);
      *pixel = getRemappedLabel(class_id);
    }
  }
}

void ImageRecolor::recolorImage(const cv::Mat& classes, cv::Mat& output) const {
  if (output.type() != CV_8UC3) {
    return;
  }

  // opencv doesn't allow resizing of 32S images...
  cv::Mat resized_classes;
  classes.convertTo(resized_classes, CV_16S);
  if (classes.rows != output.rows || classes.cols != output.cols) {
    // interpolating class labels doesn't make sense so use NEAREST
    cv::resize(resized_classes,
               resized_classes,
               cv::Size(output.cols, output.rows),
               0.0f,
               0.0f,
               cv::INTER_NEAREST);
  }

  for (int r = 0; r < resized_classes.rows; ++r) {
    for (int c = 0; c < resized_classes.cols; ++c) {
      uint8_t* pixel = output.ptr<uint8_t>(r, c);
      const auto class_id = resized_classes.at<int16_t>(r, c);
      fillColor(class_id, pixel);
    }
  }
}

void ImageRecolor::fillColor(int16_t class_id,
                             uint8_t* pixel,
                             size_t pixel_size) const {
  const auto iter = color_map_.find(class_id);
  if (iter != color_map_.end()) {
    std::memcpy(pixel, iter->second.data(), pixel_size);
    return;
  }

  if (!seen_unknown_labels_.count(class_id)) {
    SLOG(ERROR) << "Encountered class id without color: " << class_id;
    seen_unknown_labels_.insert(class_id);
  }

  std::memcpy(pixel, config.default_color.data(), pixel_size);
}

int16_t ImageRecolor::getRemappedLabel(int16_t class_id) const {
  const auto iter = label_remapping_.find(class_id);
  if (iter != label_remapping_.end()) {
    return iter->second;
  }

  if (!seen_unknown_labels_.count(class_id)) {
    SLOG(ERROR) << "Encountered unhandled class id: " << class_id;
    seen_unknown_labels_.insert(class_id);
  }

  return config.default_id;
}

void declare_config(GroupInfo& config) {
  using namespace config;
  name("GroupInfo");
  field(config.labels, "labels");
  field(config.name, "name");
}

void declare_config(ImageRecolor::Config& config) {
  using namespace config;
  name("ImageRecolor::Config");
  field(config.groups, "groups");
  field(config.default_color, "default_color");
  field(config.default_id, "default_id");
  field(config.offset, "offset");
  field<Path>(config.colormap_path, "colormap_path");
  check(config.default_color.size(), EQ, 3u, "color");
}

}  // namespace semantic_inference
