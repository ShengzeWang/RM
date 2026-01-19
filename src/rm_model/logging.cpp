#include "rm_model/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace rm_model {

namespace {
std::mutex g_log_mutex;
LogLevel g_level = LogLevel::Info;

LogLevel parse_level(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (lower == "trace") return LogLevel::Trace;
  if (lower == "debug") return LogLevel::Debug;
  if (lower == "info") return LogLevel::Info;
  if (lower == "warn" || lower == "warning") return LogLevel::Warn;
  if (lower == "error") return LogLevel::Error;
  if (lower == "off" || lower == "none") return LogLevel::None;
  return LogLevel::Info;
}

const char* level_tag(LogLevel level) {
  switch (level) {
    case LogLevel::Trace: return "TRACE";
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info: return "INFO";
    case LogLevel::Warn: return "WARN";
    case LogLevel::Error: return "ERROR";
    case LogLevel::None: return "NONE";
  }
  return "INFO";
}
} // namespace

void init_logging() {
  const char* env = std::getenv("RUST_LOG");
  if (env != nullptr) {
    g_level = parse_level(env);
  }
}

void set_log_level(LogLevel level) {
  g_level = level;
}

LogLevel log_level() {
  return g_level;
}

bool log_enabled(LogLevel level) {
  return static_cast<int>(level) >= static_cast<int>(g_level) && g_level != LogLevel::None;
}

void log(LogLevel level, const std::string& message) {
  std::lock_guard<std::mutex> guard(g_log_mutex);
  std::cerr << "[" << level_tag(level) << "] " << message << "\n";
}

} // namespace rm_model
