#pragma once
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <filesystem>

namespace logging {

enum class Level : int { DEBUG, INFO, WARNING, ERROR, FATAL };

class LogEntry;

struct LogSink {
  using Ptr = std::shared_ptr<LogSink>;
  virtual ~LogSink() = default;
  virtual void dispatch(const LogEntry& entry) const = 0;
};

class Logger {
 public:
  ~Logger() = default;

  static Logger& instance() {
    if (!s_instance_) {
      s_instance_.reset(new Logger());
    }

    return *s_instance_;
  }

  static void addSink(const std::string& name, const LogSink::Ptr& sink) {
    auto& self = instance();
    self.sinks_[name] = sink;
  }

  static void dispatchLogEntry(const LogEntry& entry) {
    auto& self = instance();
    for (const auto& name_sink_pair : self.sinks_) {
      name_sink_pair.second->dispatch(entry);
    }
  }

 private:
  Logger(){};

  inline static std::unique_ptr<Logger> s_instance_;
  std::map<std::string, LogSink::Ptr> sinks_;
};

class LogEntry {
 public:
  LogEntry(Level level, const std::string& filename, int lineno)
      : level(level), filename(filename), lineno(lineno) {}

  ~LogEntry() { Logger::dispatchLogEntry(*this); }

  Level level;
  std::string filename;
  int lineno;

  std::string prefix() const {
    std::stringstream ss;
    ss << "[" << std::filesystem::path(filename).filename().string();
    if (lineno > 0) {
      ss << ":" << lineno;
    }
    ss << "] ";
    return ss.str();
  }

  std::string message() const { return ss_.str(); }

  template <typename T>
  LogEntry& operator<<(const T& rhs) {
    ss_ << rhs;
    return *this;
  }

 private:
  std::stringstream ss_;
};

}  // namespace logging

#define SLOG(level) logging::LogEntry(logging::Level::level, __FILE__, __LINE__)
