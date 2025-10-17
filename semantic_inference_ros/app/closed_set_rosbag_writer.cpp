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
#include <semantic_inference/image_rotator.h>
#include <semantic_inference/logging.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>
#include <ianvs/bag_reader.h>

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

std::filesystem::path get_model_config(const std::string& model_name) {
  using ament_index_cpp::get_package_share_directory;
  const auto package_dir = get_package_share_directory("semantic_inference_ros");
  const auto model_dir = std::filesystem::path(package_dir) / "config" / "models";
  return model_dir / (model_name + ".yaml");
}

struct BagConfig {
  std::filesystem::path path;
  std::vector<std::string> topics;
  bool write_all_topics = true;
  std::string suffix = "_semantics";
  std::filesystem::path output;

  std::filesystem::path output_path() const {
    if (!output.empty()) {
      return output;
    }

    auto actual_path = path;
    if (!std::filesystem::is_directory(path)) {
      actual_path = path.parent_path();
    }

    return actual_path.parent_path() / (path.stem().string() + suffix);
  }

  std::map<std::string, std::string> topic_map() const {
    std::map<std::string, std::string> remapping;
    for (const auto& topic : topics) {
      SLOG(ERROR) << "Got topic: " << topic;
      auto pos = topic.find(':');
      if (pos == std::string::npos) {
        remapping[topic] = topic + "/labels";
      } else {
        const auto new_name = topic.substr(pos + 1);
        const auto old_name = topic.substr(0, pos);
        remapping[old_name] = new_name;
      }
    }

    return remapping;
  }
};

void declare_config(BagConfig& config) {
  using namespace config;
  name("BagConfig");
  field<Path::Absolute>(config.path, "bag_path");
  field(config.write_all_topics, "write_all_topics");
  check<Path::Exists>(config.path, "bag_path");
}

class ClosedSetRosbagWriter {
 public:
  explicit ClosedSetRosbagWriter(const Segmenter::Config& config);

  void processBag(const BagConfig& bag_info) const;

 private:
  cv_bridge::CvImage::Ptr runSegmentation(const cv_bridge::CvImage& img) const;

  std::unique_ptr<Segmenter> segmenter_;
};

ClosedSetRosbagWriter::ClosedSetRosbagWriter(const Segmenter::Config& config) {
  SLOG(INFO) << config::toString(config);
  if (!config::isValid(config, true)) {
    throw std::runtime_error("Invalid config!");
  }

  try {
    segmenter_ = std::make_unique<Segmenter>(config);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }
}

template <typename T>
std::string msg_type_name() {
  return rosidl_generator_traits::name<T>();
}

struct ImageDeserializer {
  static cv_bridge::CvImageConstPtr deserialize(const ianvs::BagMessage& msg,
                                                const std::string& encoding) {
    cv_bridge::CvImageConstPtr img_ptr;

    const auto serialized = msg.serialized();
    if (msg.type() == msg_type_name<Image>()) {
      auto deserialized = std::make_shared<Image>();
      uncompressed.deserialize_message(&serialized, deserialized.get());
      try {
        img_ptr = cv_bridge::toCvCopy(deserialized, encoding);
      } catch (const cv_bridge::Exception& e) {
        SLOG(ERROR) << "cv_bridge exception: " << e.what();
        return nullptr;
      }
    } else if (msg.type() == msg_type_name<CompressedImage>()) {
      auto deserialized = std::make_shared<CompressedImage>();
      compressed.deserialize_message(&serialized, deserialized.get());
      try {
        img_ptr = cv_bridge::toCvCopy(deserialized, encoding);
      } catch (const cv_bridge::Exception& e) {
        SLOG(ERROR) << "cv_bridge exception: " << e.what();
        return nullptr;
      }
    } else {
      SLOG(ERROR) << "Unknown message type '" << msg.type() << "'";
      return nullptr;
    }

    return img_ptr;
  }

  inline static const rclcpp::Serialization<Image> uncompressed = {};
  inline static const rclcpp::Serialization<CompressedImage> compressed = {};
};

void ClosedSetRosbagWriter::processBag(const BagConfig& bag) const {
  if (!config::isValid(bag)) {
    SLOG(ERROR) << "Invalid bag:\n" << config::toString(bag);
    return;
  }

  SLOG(INFO) << "Opening bag " << bag.path;
  if (bag.write_all_topics) {
    SLOG(INFO) << "(copying all topics)";
  }

  ianvs::BagReader reader(bag.path);
  if (!reader) {
    return;
  }

  rosbag2_cpp::Writer writer;
  writer.open(bag.output_path());

  SLOG(INFO) << "Processing bag!";
  const auto topic_remapping = bag.topic_map();

  SLOG(INFO) << "Segmentation topics:";
  for (const auto& [old_topic, new_topic] : topic_remapping) {
    SLOG(INFO) << " - " << old_topic << " -> " << new_topic;
  }

  std::set<std::string> seen;
  ianvs::BagMessage::Ptr msg;
  do {
    msg = reader.next();
    if (!msg) {
      continue;
    }

    const auto topic = msg->topic();
    if (bag.write_all_topics) {
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

    const auto labels = runSegmentation(*img);
    if (!labels) {
      continue;
    }

    // NOTE(nathan) no need to create topic if we're writing a known type
    const auto msg_out = labels->toImageMsg();
    const rclcpp::Time msg_time = msg_out->header.stamp;
    writer.write(*msg_out, iter->second, msg_time);
  } while (msg);

  for (const auto& seen_topic : seen) {
    const auto segment = topic_remapping.count(seen_topic);
    SLOG(ERROR) << "Saw " << seen_topic << " (segment: " << std::boolalpha << segment
                << ")";
  }

  SLOG(INFO) << "Finished processing bag " << bag.path;
}

CvImage::Ptr ClosedSetRosbagWriter::runSegmentation(const CvImage& image) const {
  SLOG(DEBUG) << "Encoding: " << image.encoding << " size: " << image.image.cols
              << " x " << image.image.rows << " x " << image.image.channels()
              << " is right type? " << (image.image.type() == CV_8UC3 ? "yes" : "no");

  // TODO(nathan) handle rotation per topic
  // const auto rotated = image_rotator_.rotate(image.image);
  const auto result = segmenter_->infer(image.image);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return nullptr;
  }

  // const auto derotated = image_rotator_.derotate(result.labels);

  auto labels = std::make_shared<cv_bridge::CvImage>();
  labels->header = image.header;
  result.labels.convertTo(labels->image, CV_16S);
  return labels;
}

}  // namespace semantic_inference

struct SimpleSink : logging::LogSink {
  SimpleSink(logging::Level level = logging::Level::INFO, bool with_prefix = false)
      : level(level), with_prefix(with_prefix) {}
  virtual ~SimpleSink() = default;
  void dispatch(const logging::LogEntry& entry) const override {
    if (entry.level < level) {
      // skip ignored entries
      return;
    }

    std::stringstream ss;
    if (with_prefix) {
      ss << entry.prefix();
    }

    ss << entry.message();
    std::cout << ss.str() << std::endl;
  }

  const logging::Level level;
  const bool with_prefix;
};

using semantic_inference::BagConfig;
using semantic_inference::ClosedSetRosbagWriter;
using semantic_inference::Segmenter;

struct AppArgs {
  void add_to_app(CLI::App& app) {
    app.add_option("bag_path", bag.path)->required()->description("Bag to open");
    app.add_option("-o,--outpt", bag.output)->description("Optional output path");
    app.add_flag("--write-all-topics,!--no-write-all-topics",
                 bag.write_all_topics,
                 "write all other topics to bag");
    app.add_option("-t,--topics", bag.topics)
        ->description("Topics to run inference on");
    app.add_option("-m,--model", model_name)->description("Model to use");
  }

  std::string model_name = "ade20k-efficientvit_seg_l2";
  BagConfig bag;
};

auto main(int argc, char* argv[]) -> int {
  logging::Logger::addSink("cout", std::make_shared<SimpleSink>());
  logging::setConfigUtilitiesLogger();

  CLI::App app("Utility to play a rosbag after modfying and publishing transforms");
  app.allow_extras();
  app.get_formatter()->column_width(50);

  AppArgs args;
  args.add_to_app(app);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  const std::filesystem::path this_path(
      ament_index_cpp::get_package_share_directory("semantic_inference_ros"));
  const auto model_dir = this_path / "config" / "models";

  const auto model_config = model_dir / (args.model_name + ".yaml");
  if (!std::filesystem::exists(model_config)) {
    throw std::runtime_error("Invalid model config '" + model_config.string() + "'");
  }

  auto config = config::fromYamlFile<Segmenter::Config>(model_config, "segmenter");
  config.model.model_file = args.model_name + ".onnx";

  ClosedSetRosbagWriter writer(config);
  writer.processBag(args.bag);
  return 0;
}
