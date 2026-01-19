#include "rm_model/models/model.h"
#include "rm_model/models/stdlib.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace rm_model {

std::size_t ModelParam::size() const {
  return std::visit([](const auto& value) -> std::size_t {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t>) {
      return sizeof(uint64_t);
    } else if constexpr (std::is_same_v<T, double>) {
      return sizeof(double);
    } else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
      return sizeof(uint16_t) * value.size();
    } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
      return sizeof(uint64_t) * value.size();
    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
      return sizeof(uint32_t) * value.size();
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      return sizeof(double) * value.size();
    }
  }, value_);
}

const char* ModelParam::c_type() const {
  return std::visit([](const auto& value) -> const char* {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t>) return "uint64_t";
    if constexpr (std::is_same_v<T, double>) return "double";
    if constexpr (std::is_same_v<T, std::vector<uint16_t>>) return "short";
    if constexpr (std::is_same_v<T, std::vector<uint64_t>>) return "uint64_t";
    if constexpr (std::is_same_v<T, std::vector<uint32_t>>) return "uint32_t";
    if constexpr (std::is_same_v<T, std::vector<double>>) return "double";
    return "uint64_t";
  }, value_);
}

bool ModelParam::is_array() const {
  return std::visit([](const auto& value) -> bool {
    using T = std::decay_t<decltype(value)>;
    return std::is_same_v<T, std::vector<uint16_t>> ||
           std::is_same_v<T, std::vector<uint64_t>> ||
           std::is_same_v<T, std::vector<uint32_t>> ||
           std::is_same_v<T, std::vector<double>>;
  }, value_);
}

const char* ModelParam::c_type_mod() const {
  return is_array() ? "[]" : "";
}

std::string ModelParam::c_val() const {
  return std::visit([](const auto& value) -> std::string {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t>) {
      return std::to_string(value) + "UL";
    } else if constexpr (std::is_same_v<T, double>) {
      std::ostringstream oss;
      oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
      oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
      std::string s = oss.str();
      if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
          s.find('E') == std::string::npos) {
        s += ".0";
      }
      return s;
    } else {
      std::ostringstream oss;
      oss << "{ ";
      bool first = true;
      for (const auto& v : value) {
        if (!first) {
          oss << ", ";
        }
        first = false;
        if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
          oss << v;
        } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          oss << v << "UL";
        } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
          oss << v << "UL";
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          std::ostringstream f;
          f.setf(std::ios::fmtflags(0), std::ios::floatfield);
          f << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
          std::string s = f.str();
          if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
              s.find('E') == std::string::npos) {
            s += ".0";
          }
          oss << s;
        }
      }
      oss << " }";
      return oss.str();
    }
  }, value_);
}

namespace {

template <typename T>
void write_le(std::ostream& out, T value) {
  static_assert(std::is_trivially_copyable_v<T>, "write_le requires trivial type");
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    out.put(static_cast<char>((value >> (i * 8)) & 0xFF));
  }
}

void write_le_double(std::ostream& out, double value) {
  static_assert(sizeof(double) == sizeof(uint64_t), "double size mismatch");
  uint64_t tmp;
  std::memcpy(&tmp, &value, sizeof(double));
  write_le<uint64_t>(out, tmp);
}

} // namespace

void ModelParam::write_to(std::ostream& out) const {
  std::visit([&](const auto& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t>) {
      write_le<uint64_t>(out, value);
    } else if constexpr (std::is_same_v<T, double>) {
      write_le_double(out, value);
    } else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
      for (auto v : value) {
        write_le<uint16_t>(out, v);
      }
    } else if constexpr (std::is_same_v<T, std::vector<uint64_t>>) {
      for (auto v : value) {
        write_le<uint64_t>(out, v);
      }
    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
      for (auto v : value) {
        write_le<uint32_t>(out, v);
      }
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      for (auto v : value) {
        write_le_double(out, v);
      }
    }
  }, value_);
}

double ModelParam::as_float() const {
  return std::visit([](const auto& value) -> double {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t>) {
      return static_cast<double>(value);
    } else if constexpr (std::is_same_v<T, double>) {
      return value;
    }
    throw std::runtime_error("Cannot treat array parameter as float");
  }, value_);
}

std::size_t ModelParam::len() const {
  return std::visit([](const auto& value) -> std::size_t {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, double>) {
      return 1;
    } else {
      return value.size();
    }
  }, value_);
}

std::set<StdFunctions> Model::standard_functions() const {
  return {};
}

} // namespace rm_model
