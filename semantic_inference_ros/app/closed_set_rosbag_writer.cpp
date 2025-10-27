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
#include <ianvs/bag_reader.h>
#include <semantic_inference/image_rotator.h>
#include <semantic_inference/logging.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <CLI/CLI.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace semantic_inference {

using cv_bridge::CvImage;
using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;

template <typename T>
cv_bridge::CvImageConstPtr imageFromMsg(const ianvs::BagMessage& msg,
                                        const std::string& encoding,
                                        const rclcpp::Serialization<T>& serialization) {
  const auto serialized = msg.serialized();
  auto deserialized = std::make_shared<T>();
  serialization.deserialize_message(&serialized, deserialized.get());
  try {
    return cv_bridge::toCvCopy(deserialized, encoding);
  } catch (const cv_bridge::Exception& e) {
    SLOG(ERROR) << "cv_bridge exception: " << e.what();
    return nullptr;
  }
}

struct ImageDeserializer {
  static cv_bridge::CvImageConstPtr deserialize(const ianvs::BagMessage& msg,
                                                const std::string& encoding) {
    if (msg.is<Image>()) {
      return imageFromMsg(msg, encoding, uncompressed);
    }

    if (msg.is<CompressedImage>()) {
      return imageFromMsg(msg, encoding, compressed);
    }

    SLOG(ERROR) << "Unknown message type '" << msg.type() << "'";
    return nullptr;
  }

  inline static const rclcpp::Serialization<Image> uncompressed = {};
  inline static const rclcpp::Serialization<CompressedImage> compressed = {};
};

Segmenter::Config loadSegmentationConfig(const std::string& model_name,
                                         const std::string& model_verbosity) {
  using ament_index_cpp::get_package_share_directory;
  const auto package_dir = get_package_share_directory("semantic_inference_ros");
  const auto model_config_dir =
      std::filesystem::path(package_dir) / "config" / "models";
  const auto model_config_path = model_config_dir / (model_name + ".yaml");
  if (!std::filesystem::exists(model_config_path)) {
    throw std::runtime_error("Invalid model config '" + model_config_path.string() +
                             "'");
  }

  auto config = config::fromYamlFile<Segmenter::Config>(model_config_path, "segmenter");
  config.model.model_file = model_name + ".onnx";
  config.model.log_severity = model_verbosity;
  return config;
}

struct AppArgs {
  struct TopicConfig {
    std::string input;
    std::string output;
    RotationType rotation = RotationType::NONE;

    static TopicConfig fromArg(const std::string& arg);
  };

  std::string model_name = "ade20k-efficientvit_seg_l2";
  std::string model_verbosity = "WARNING";

  bool show_config = false;
  bool segmentation_only = false;
  bool quiet = false;
  bool overwrite = false;

  std::filesystem::path path;
  std::vector<std::string> topics;
  std::string suffix = "_semantics";
  std::filesystem::path output;

  void add_to_app(CLI::App& app);
  std::filesystem::path output_path() const;
  std::map<std::string, TopicConfig> topic_map() const;
};

AppArgs::TopicConfig AppArgs::TopicConfig::fromArg(const std::string& arg) {
  auto pos = arg.find(':');
  if (pos == std::string::npos) {
    return {arg, arg + "/labels"};
  }

  const auto old_name = arg.substr(0, pos);
  const auto rest = arg.substr(pos + 1);
  pos = rest.find(':');
  if (pos == std::string::npos) {
    return {old_name, rest};
  }

  const auto new_name = rest.substr(0, pos);
  const auto rotation_constant = std::stoi(rest.substr(pos + 1));

  RotationType rotation;
  if (rotation_constant == 90) {
    rotation = RotationType::ROTATE_90_CLOCKWISE;
  } else if (rotation_constant == 180) {
    rotation = RotationType::ROTATE_180;
  } else if (rotation_constant == -90 || rotation_constant == 270) {
    rotation = RotationType::ROTATE_90_COUNTERCLOCKWISE;
  } else {
    throw std::runtime_error("Invalid rotation constant for topic '" + old_name +
                             "': '" + std::to_string(rotation_constant) + "'");
  }

  return {old_name, new_name, rotation};
}

void AppArgs::add_to_app(CLI::App& app) {
  app.add_flag("--show-config", show_config, "display segmentation config");
  app.add_flag("--segmentation-only", segmentation_only, "don't copy bag to output");
  app.add_flag("--quiet", quiet, "disable logging");
  app.add_flag("-f,--force", overwrite, "remove output if it exists");

  app.add_option("bag_path", path)->required()->description("Bag to open");
  app.add_option("-o,--outpt", output)->description("Optional output path");
  app.add_option("-t,--topics", topics)->description("Topics to run inference on");
  app.add_option("-m,--model", model_name)->description("Model to use");
  app.add_option("-v,--model-verbosity", model_verbosity)
      ->description("Model verbosity");
}

std::filesystem::path AppArgs::output_path() const {
  if (!output.empty()) {
    return output;
  }

  auto actual_path = path;
  if (!std::filesystem::is_directory(path)) {
    actual_path = path.parent_path();
  }

  return actual_path.parent_path() / (path.stem().string() + suffix);
}

std::map<std::string, AppArgs::TopicConfig> AppArgs::topic_map() const {
  std::map<std::string, TopicConfig> remapping;
  for (const auto& topic : topics) {
    const auto topic_config = TopicConfig::fromArg(topic);
    remapping[topic_config.input] = topic_config;
  }

  return remapping;
}

class ClosedSetRosbagWriter {
 public:
  explicit ClosedSetRosbagWriter(const AppArgs& args);

  void run() const;

  const AppArgs args;

 private:
  cv_bridge::CvImage::Ptr runSegmentation(const cv_bridge::CvImage& img,
                                          RotationType rotation) const;

  std::unique_ptr<Segmenter> segmenter_;
};

struct ProgressBar {
  ProgressBar(size_t total, const std::string& prefix = "")
      : total(total), prefix(prefix) {}

  void next() {
    ++count;
    const auto percent = static_cast<double>(count) / total;
    if (percent - last_percent < min_diff) {
      return;
    }

    print(percent, false);
    last_percent = percent;
  }

  void finish() { print(1.0, true); }

  void print(double percent, bool clear) {
    if (!prefix.empty()) {
      std::cout << prefix << ": ";
    }

    const auto bars = static_cast<size_t>(std::floor(percent * width));
    std::cout << "[" << std::string(bars, '#');
    if (bars <= width) {
      std::cout << std::string(width - bars, ' ');
    }

    std::cout << "] " << std::fixed << std::setw(5) << std::setprecision(1)
              << 100 * percent << "%";

    if (clear) {
      std::cout << std::endl;
    } else {
      std::cout << "\r";
      std::cout.flush();
    }
  }

  const size_t total;
  std::string prefix;
  double last_percent = 0.0;
  double min_diff = 0.001;
  size_t width = 60;
  size_t count = 0;
};

ClosedSetRosbagWriter::ClosedSetRosbagWriter(const AppArgs& args) : args(args) {
  const auto config = loadSegmentationConfig(args.model_name, args.model_verbosity);
  if (args.show_config) {
    SLOG(INFO) << config::toString(config);
  }

  if (!config::isValid(config, true)) {
    throw std::runtime_error("Invalid config!");
  }

  segmenter_ = std::make_unique<Segmenter>(config);
}

void ClosedSetRosbagWriter::run() const {
  const auto topic_remapping = args.topic_map();
  if (topic_remapping.empty()) {
    throw std::runtime_error("No topics specified!");
  }

  const auto output_path = args.output_path();
  if (std::filesystem::exists(output_path) && args.overwrite) {
    SLOG(WARNING) << "Removing existing output " << output_path;
    std::filesystem::remove_all(output_path);
  }

  if (!args.quiet) {
    std::stringstream ss;
    ss << "Segmenting bag " << args.path << " to " << output_path;
    if (!args.segmentation_only) {
      ss << " (copying all topics)";
    }

    ss << "\nSegmentation topics:\n";
    for (const auto& [old_topic, new_topic] : topic_remapping) {
      ss << " - " << old_topic << " -> " << new_topic.output << "\n";
    }
    SLOG(INFO) << ss.str();
  }

  ianvs::BagReader reader(args.path);
  if (!reader) {
    return;
  }

  rosbag2_cpp::Writer writer;
  writer.open(args.output_path());

  ProgressBar bar(reader.message_count(), "Processing bag");
  std::set<std::string> seen;
  ianvs::BagMessage::Ptr msg;
  do {
    msg = reader.next();
    if (!msg) {
      continue;
    }

    bar.next();
    const auto topic = msg->topic();
    if (!args.segmentation_only) {
      if (!seen.count(topic)) {
        writer.create_topic(msg->metadata);
        seen.insert(topic);
      }

      writer.write(msg->contents);
    }

    auto iter = topic_remapping.find(topic);
    if (iter == topic_remapping.end()) {
      continue;
    }

    const auto img = ImageDeserializer::deserialize(*msg, "rgb8");
    if (!img) {
      SLOG(ERROR) << "Failed to deserialize image!";
      continue;
    }

    const auto labels = runSegmentation(*img, iter->second.rotation);
    if (!labels) {
      continue;
    }

    // NOTE(nathan) no need to create topic if we're writing a known type
    const auto msg_out = labels->toImageMsg();
    const rclcpp::Time msg_time(msg->contents->recv_timestamp);
    writer.write(*msg_out, iter->second.output, msg_time);
  } while (msg);

  bar.finish();
}

CvImage::Ptr ClosedSetRosbagWriter::runSegmentation(const CvImage& image,
                                                    RotationType rotation) const {
  SLOG(DEBUG) << "Encoding: " << image.encoding << " size: " << image.image.cols
              << " x " << image.image.rows << " x " << image.image.channels()
              << " is right type? " << (image.image.type() == CV_8UC3 ? "yes" : "no");

  const ImageRotator rotator(ImageRotator::Config{rotation});
  const auto rotated = rotator.rotate(image.image);
  const auto result = segmenter_->infer(rotated);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return nullptr;
  }

  const auto derotated = rotator.derotate(result.labels);
  auto labels = std::make_shared<cv_bridge::CvImage>();
  labels->header = image.header;
  labels->encoding = "16SC1";  // 16-bit signed, single channel
  derotated.convertTo(labels->image, CV_16S);
  return labels;
}

}  // namespace semantic_inference

using semantic_inference::ClosedSetRosbagWriter;
using semantic_inference::Segmenter;

auto main(int argc, char* argv[]) -> int {
  logging::Logger::addSink("cout", std::make_shared<logging::SimpleSink>());
  logging::setConfigUtilitiesLogger();

  CLI::App app("Utility to play a rosbag after modfying and publishing transforms");
  app.allow_extras();
  app.get_formatter()->column_width(50);

  semantic_inference::AppArgs args;
  args.add_to_app(app);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  ClosedSetRosbagWriter writer(args);
  writer.run();
  return 0;
}
