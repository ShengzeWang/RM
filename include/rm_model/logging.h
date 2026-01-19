#ifndef RM_MODEL_LOGGING_H
#define RM_MODEL_LOGGING_H

#include <sstream>
#include <string>

namespace rm_model {

enum class LogLevel {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  None = 5,
};

void init_logging();
void set_log_level(LogLevel level);
LogLevel log_level();
bool log_enabled(LogLevel level);
void log(LogLevel level, const std::string& message);

} // namespace rm_model

#define RM_MODEL_LOG_TRACE(msg)                                                         \
  do {                                                                             \
    if (rm_model::log_enabled(rm_model::LogLevel::Trace)) {                                   \
      std::ostringstream _rm_model_oss;                                             \
      _rm_model_oss << msg;                                                         \
      rm_model::log(rm_model::LogLevel::Trace, _rm_model_oss.str());                \
    }                                                                              \
  } while (0)

#define RM_MODEL_LOG_DEBUG(msg)                                                         \
  do {                                                                             \
    if (rm_model::log_enabled(rm_model::LogLevel::Debug)) {                                   \
      std::ostringstream _rm_model_oss;                                             \
      _rm_model_oss << msg;                                                         \
      rm_model::log(rm_model::LogLevel::Debug, _rm_model_oss.str());                \
    }                                                                              \
  } while (0)

#define RM_MODEL_LOG_INFO(msg)                                                          \
  do {                                                                             \
    if (rm_model::log_enabled(rm_model::LogLevel::Info)) {                                    \
      std::ostringstream _rm_model_oss;                                             \
      _rm_model_oss << msg;                                                         \
      rm_model::log(rm_model::LogLevel::Info, _rm_model_oss.str());                 \
    }                                                                              \
  } while (0)

#define RM_MODEL_LOG_WARN(msg)                                                          \
  do {                                                                             \
    if (rm_model::log_enabled(rm_model::LogLevel::Warn)) {                                    \
      std::ostringstream _rm_model_oss;                                             \
      _rm_model_oss << msg;                                                         \
      rm_model::log(rm_model::LogLevel::Warn, _rm_model_oss.str());                 \
    }                                                                              \
  } while (0)

#define RM_MODEL_LOG_ERROR(msg)                                                         \
  do {                                                                             \
    if (rm_model::log_enabled(rm_model::LogLevel::Error)) {                                   \
      std::ostringstream _rm_model_oss;                                             \
      _rm_model_oss << msg;                                                         \
      rm_model::log(rm_model::LogLevel::Error, _rm_model_oss.str());                \
    }                                                                              \
  } while (0)

#endif // RM_MODEL_LOGGING_H
