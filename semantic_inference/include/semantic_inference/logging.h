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
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace logging {

enum class Level : int { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

class LogEntry;

struct LogSink {
  using Ptr = std::shared_ptr<LogSink>;
  virtual ~LogSink() = default;
  virtual void dispatch(const LogEntry& entry) const = 0;
};

class Logger {
 public:
  ~Logger() = default;

  static Logger& instance();
  static void addSink(const std::string& name, const LogSink::Ptr& sink);
  static void dispatchLogEntry(const LogEntry& entry);
  static void logMessage(Level level, const std::string& message);

 private:
  Logger();

  inline static std::unique_ptr<Logger> s_instance_;
  std::map<std::string, LogSink::Ptr> sinks_;
};

class LogEntry {
 public:
  LogEntry(Level level, const std::string& filename = "", int lineno = 0);

  ~LogEntry();

  Level level;
  std::string filename;
  int lineno;

  std::string prefix() const;
  std::string message() const;

  template <typename T>
  LogEntry& operator<<(const T& rhs) {
    ss_ << rhs;
    return *this;
  }

 private:
  std::stringstream ss_;
};

/**
 * @brief Log messages to cout/cerr as appropriate
 */
struct CoutSink : LogSink {
  CoutSink(Level level = Level::INFO);
  virtual ~CoutSink() = default;

  void dispatch(const LogEntry& entry) const override;

  Level level;
};

/**
 * @brief Forward everything to cout without log-levels or optionally prefix
 */
struct SimpleSink : LogSink {
  SimpleSink(Level level = Level::INFO, bool with_prefix = false);
  virtual ~SimpleSink() = default;
  void dispatch(const LogEntry& entry) const override;

  const Level level;
  const bool with_prefix;
};

void setConfigUtilitiesLogger();

}  // namespace logging

#define SLOG(level) logging::LogEntry(logging::Level::level, __FILE__, __LINE__)
