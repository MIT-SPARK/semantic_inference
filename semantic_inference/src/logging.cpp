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

#include "semantic_inference/logging.h"

#include <config_utilities/internal/logger.h>

#include <filesystem>
#include <iostream>

namespace logging {

Logger& Logger::instance() {
  if (!s_instance_) {
    s_instance_.reset(new Logger());
  }

  return *s_instance_;
}

void Logger::addSink(const std::string& name, const LogSink::Ptr& sink) {
  auto& self = instance();
  self.sinks_[name] = sink;
}

void Logger::dispatchLogEntry(const LogEntry& entry) {
  auto& self = instance();
  for (const auto& name_sink_pair : self.sinks_) {
    name_sink_pair.second->dispatch(entry);
  }
}

void Logger::logMessage(Level level, const std::string& message) {
  LogEntry entry(level);
  entry << message;
}

Logger::Logger() {}

LogEntry::LogEntry(Level level, const std::string& filename, int lineno)
    : level(level), filename(filename), lineno(lineno) {}

LogEntry::~LogEntry() { Logger::dispatchLogEntry(*this); }

std::string LogEntry::prefix() const {
  if (filename.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << "[" << std::filesystem::path(filename).filename().string();
  if (lineno > 0) {
    ss << ":" << lineno;
  }

  ss << "] ";
  return ss.str();
}

std::string LogEntry::message() const { return ss_.str(); }

CoutSink::CoutSink(Level level) : level(level) {}

void CoutSink::dispatch(const logging::LogEntry& entry) const {
  if (entry.level < level) {
    // skip ignored entries
    return;
  }

  std::stringstream ss;
  ss << entry.prefix() << entry.message();
  switch (entry.level) {
    case logging::Level::WARNING:
      std::cerr << "[WARNING]" << ss.str() << std::endl;
      break;
    case logging::Level::ERROR:
      std::cerr << "[ERROR]" << ss.str() << std::endl;
      break;
    case logging::Level::FATAL:
      std::cerr << "[FATAL]" << ss.str() << std::endl;
      break;
    case logging::Level::INFO:
      std::cout << "[INFO]" << ss.str() << std::endl;
      break;
    default:
    case logging::Level::DEBUG:
      std::cout << "[DEBUG]" << ss.str() << std::endl;
      break;
  }
}

struct SlogLogger : config::internal::Logger {
  void logImpl(const config::internal::Severity severity,
               const std::string& message) override {
    switch (severity) {
      default:
      case config::internal::Severity::kInfo:
        logging::Logger::logMessage(Level::INFO, message);
        break;
      case config::internal::Severity::kWarning:
        logging::Logger::logMessage(Level::INFO, message);
        break;
      case config::internal::Severity::kError:
        logging::Logger::logMessage(Level::ERROR, message);
        break;
      case config::internal::Severity::kFatal:
        logging::Logger::logMessage(Level::FATAL, message);
        throw std::runtime_error(message);
        break;
    }
  }
};

void setConfigUtilitiesLogger() {
  config::internal::Logger::setLogger(std::make_shared<SlogLogger>());
}

}  // namespace logging
