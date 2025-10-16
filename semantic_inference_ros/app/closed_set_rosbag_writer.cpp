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
#include <config_utilities/parsing/commandline.h>
#include <config_utilities/types/path.h>
#include <semantic_inference/image_rotator.h>
#include <semantic_inference/logging.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace semantic_inference {

using sensor_msgs::msg::Image;

struct MessageInfo {
  std::shared_ptr<rosbag2_storage::SerializedBagMessage> contents;
  const rosbag2_storage::TopicMetadata* metadata = nullptr;
  std::string topic() const { return contents ? contents->topic_name : ""; }
  std::string type() const { return metadata ? metadata->type : ""; }
  operator bool() const { return contents != nullptr && metadata != nullptr; }
  rclcpp::SerializedMessage serialized() const {
    return rclcpp::SerializedMessage(*contents->serialized_data);
  }
};

struct BagReader {
  BagReader(const std::filesystem::path& bagpath);
  MessageInfo next() const;
  operator bool() const { return reader != nullptr; }

  std::unique_ptr<rosbag2_cpp::Reader> reader;
  std::map<std::string, rosbag2_storage::TopicMetadata> lookup;
};

BagReader::BagReader(const std::filesystem::path& bagpath) {
  rosbag2_storage::StorageOptions opts;
  opts.uri = bagpath;
  reader = rosbag2_transport::ReaderWriterFactory::make_reader(opts);
  if (!reader) {
    return;
  }

  reader->open(opts);

  const auto metadata = reader->get_all_topics_and_types();
  for (const auto& data : metadata) {
    lookup[data.name] = data;
  }
}

MessageInfo BagReader::next() const {
  while (reader->has_next()) {
    auto msg = reader->read_next();
    if (!msg) {
      continue;
    }

    auto iter = lookup.find(msg->topic_name);
    if (iter == lookup.end()) {
      SLOG(ERROR) << "no find metadata for topic '" << msg->topic_name << "'";
      continue;
    }

    return {msg, &iter->second};
  }

  return {};
}

struct BagConfig {
  std::filesystem::path path;
  std::string topic;
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
};

void declare_config(BagConfig& config) {
  using namespace config;
  name("BagConfig");
  field<Path::Absolute>(config.path, "bag_path");
  field(config.write_all_topics, "write_all_topics");
  check<Path::Exists>(config.path, "bag_path");
  checkCondition(!config.topic.empty(), "topic non-empty");
}

class ClosedSetRosbagWriter {
 public:
  struct Config {
    bool show_config = true;
    Segmenter::Config segmenter;
    ImageRotator::Config image_rotator;
  } const config;

  explicit ClosedSetRosbagWriter(const Config& config);

  void processBag(const BagConfig& bag_info) const;

 private:
  cv::Mat runSegmentation(const cv::Mat& img) const;

  std::unique_ptr<Segmenter> segmenter_;
  ImageRotator image_rotator_;
};

void declare_config(ClosedSetRosbagWriter::Config& config) {
  using namespace config;
  name("ClosedSetRosbagWriter::Config");
  field(config.segmenter, "segmenter");
  field(config.image_rotator, "image_rotator");
  field(config.show_config, "show_config");
}

ClosedSetRosbagWriter::ClosedSetRosbagWriter(const Config& config)
    : config(config), image_rotator_(config.image_rotator) {
  if (config.show_config) {
    SLOG(INFO) << "\n" << config::toString(config);
  }

  config::checkValid(config);

  try {
    segmenter_ = std::make_unique<Segmenter>(config.segmenter);
  } catch (const std::exception& e) {
    SLOG(ERROR) << "Exception: " << e.what();
    throw e;
  }
}

template <typename T>
std::string msg_type_name() {
  return rosidl_generator_traits::data_type<T>();
}

struct ImageDeserializer {
  static cv::Mat deserialize(const MessageInfo& msg, const std::string& encoding) {
    cv_bridge::CvImageConstPtr img_ptr;

    const auto serialized = msg.serialized();
    if (msg.type() == msg_type_name<sensor_msgs::msg::Image>()) {
      auto deserialized = std::make_shared<sensor_msgs::msg::Image>();
      uncompressed.deserialize_message(&serialized, deserialized.get());
      try {
        img_ptr = cv_bridge::toCvCopy(deserialized, encoding);
      } catch (const cv_bridge::Exception& e) {
        SLOG(ERROR) << "cv_bridge exception: " << e.what();
        return cv::Mat();
      }
    } else if (msg.type() == msg_type_name<sensor_msgs::msg::CompressedImage>()) {
      auto deserialized = std::make_shared<sensor_msgs::msg::CompressedImage>();
      compressed.deserialize_message(&serialized, deserialized.get());
      try {
        img_ptr = cv_bridge::toCvCopy(deserialized, encoding);
      } catch (const cv_bridge::Exception& e) {
        SLOG(ERROR) << "cv_bridge exception: " << e.what();
        return cv::Mat();
      }
    } else {
      SLOG(ERROR) << "Unknown message type '" << msg.type() << "'";
      return cv::Mat();
    }

    SLOG(DEBUG) << "Encoding: " << img_ptr->encoding << " size: " << img_ptr->image.cols
                << " x " << img_ptr->image.rows << " x " << img_ptr->image.channels()
                << " is right type? "
                << (img_ptr->image.type() == CV_8UC3 ? "yes" : "no");

    return img_ptr->image;
  }

  static const rclcpp::Serialization<sensor_msgs::msg::Image> uncompressed;
  static const rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed;
};

void ClosedSetRosbagWriter::processBag(const BagConfig& bag) const {
  if (!config::isValid(bag)) {
    SLOG(ERROR) << "Invalid bag:\n" << config::toString(bag);
    return;
  }

  rosbag2_storage::StorageOptions opts_out;
  opts_out.uri = bag.output_path();
  rosbag2_cpp::writers::SequentialWriter writer;
  writer.open(opts_out, rosbag2_cpp::ConverterOptions{});

  BagReader reader(bag.path);
  if (!reader) {
    return;
  }

  std::set<std::string> seen;
  MessageInfo msg;
  do {
    msg = reader.next();
    if (!msg) {
      continue;
    }

    const auto topic = msg.topic();
    if (bag.write_all_topics) {
      if (!seen.count(topic)) {
        writer.create_topic(*msg.metadata);
        seen.insert(topic);
      }

      writer.write(msg.contents);
    }

    if (topic != bag.topic) {
      continue;
    }

    const auto img = ImageDeserializer::deserialize(msg, "rgb8");
    if (img.empty()) {
      SLOG(ERROR) << "Failed to deserialize image!";
      continue;
    }

    const auto labels = runSegmentation(img);
    if (labels.empty()) {
      continue;
    }

    // TODO(nathan) write image to bag
  } while (msg);
}

cv::Mat ClosedSetRosbagWriter::runSegmentation(const cv::Mat& image) const {
  const auto rotated = image_rotator_.rotate(image);
  const auto result = segmenter_->infer(rotated);
  if (!result) {
    SLOG(ERROR) << "failed to run inference!";
    return cv::Mat();
  }

  return image_rotator_.derotate(result.labels);
}

}  // namespace semantic_inference

auto main(int argc, char* argv[]) -> int {
  logging::Logger::addSink("cout",
                           std::make_shared<logging::CoutSink>(logging::Level::INFO));
  logging::setConfigUtilitiesLogger();

  const auto config =
      config::fromCLI<semantic_inference::ClosedSetRosbagWriter::Config>(argc, argv);
  semantic_inference::ClosedSetRosbagWriter writer(config);

  return 0;
}
