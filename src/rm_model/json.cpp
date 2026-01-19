#include "rm_model/json.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rm_model {
namespace json {

namespace {

void append_utf8(std::string& out, uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
}

uint32_t parse_hex4(std::string_view input, std::size_t& pos) {
  if (pos + 4 > input.size()) {
    throw std::runtime_error("Invalid unicode escape");
  }
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    char c = input[pos++];
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= static_cast<uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value |= static_cast<uint32_t>(10 + c - 'a');
    } else if (c >= 'A' && c <= 'F') {
      value |= static_cast<uint32_t>(10 + c - 'A');
    } else {
      throw std::runtime_error("Invalid unicode escape");
    }
  }
  return value;
}

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input), pos_(0) {}

  Value parse_value() {
    skip_ws();
    if (pos_ >= input_.size()) {
      throw std::runtime_error("Unexpected end of input");
    }
    char c = input_[pos_];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Value(parse_string());
    if (c == 't') return parse_literal("true", Value(true));
    if (c == 'f') return parse_literal("false", Value(false));
    if (c == 'n') return parse_literal("null", Value(nullptr));
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
    throw std::runtime_error("Invalid JSON value");
  }

  void ensure_end() {
    skip_ws();
    if (pos_ != input_.size()) {
      throw std::runtime_error("Trailing data after JSON value");
    }
  }

 private:
  Value parse_object() {
    expect('{');
    Value::Object obj;
    skip_ws();
    if (peek('}')) {
      expect('}');
      return Value(std::move(obj));
    }

    while (true) {
      skip_ws();
      if (!peek('"')) {
        throw std::runtime_error("Expected string key");
      }
      std::string key = parse_string();
      skip_ws();
      expect(':');
      Value val = parse_value();
      obj.emplace_back(std::move(key), std::move(val));
      skip_ws();
      if (peek('}')) {
        expect('}');
        break;
      }
      expect(',');
    }

    return Value(std::move(obj));
  }

  Value parse_array() {
    expect('[');
    Value::Array arr;
    skip_ws();
    if (peek(']')) {
      expect(']');
      return Value(std::move(arr));
    }

    while (true) {
      arr.emplace_back(parse_value());
      skip_ws();
      if (peek(']')) {
        expect(']');
        break;
      }
      expect(',');
    }

    return Value(std::move(arr));
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (pos_ < input_.size()) {
      char c = input_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= input_.size()) {
          throw std::runtime_error("Invalid escape sequence");
        }
        char esc = input_[pos_++];
        switch (esc) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            uint32_t codepoint = parse_hex4(input_, pos_);
            if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
              if (pos_ + 1 >= input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
                throw std::runtime_error("Invalid surrogate pair");
              }
              pos_ += 2;
              uint32_t low = parse_hex4(input_, pos_);
              if (low < 0xDC00 || low > 0xDFFF) {
                throw std::runtime_error("Invalid surrogate pair");
              }
              codepoint = 0x10000 + (((codepoint - 0xD800) << 10) | (low - 0xDC00));
            }
            append_utf8(out, codepoint);
            break;
          }
          default:
            throw std::runtime_error("Invalid escape sequence");
        }
      } else {
        if (static_cast<unsigned char>(c) < 0x20) {
          throw std::runtime_error("Invalid control character in string");
        }
        out.push_back(c);
      }
    }
    throw std::runtime_error("Unterminated string");
  }

  Value parse_number() {
    std::size_t start = pos_;
    bool neg = false;
    if (peek('-')) {
      neg = true;
      ++pos_;
    }
    if (peek('0')) {
      ++pos_;
    } else {
      if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        throw std::runtime_error("Invalid number");
      }
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
      }
    }
    bool is_int = true;
    if (peek('.')) {
      is_int = false;
      ++pos_;
      if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        throw std::runtime_error("Invalid number");
      }
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
      }
    }
    if (peek('e') || peek('E')) {
      is_int = false;
      ++pos_;
      if (peek('+') || peek('-')) ++pos_;
      if (pos_ >= input_.size() || !std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        throw std::runtime_error("Invalid number");
      }
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
      }
    }

    std::string_view token = input_.substr(start, pos_ - start);
    if (is_int) {
      std::size_t digit_pos = neg ? start + 1 : start;
      if (digit_pos >= pos_) {
        throw std::runtime_error("Invalid number");
      }
      uint64_t value = 0;
      uint64_t limit = neg
                           ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1ULL
                           : std::numeric_limits<uint64_t>::max();
      for (std::size_t i = digit_pos; i < pos_; ++i) {
        char c = input_[i];
        uint64_t digit = static_cast<uint64_t>(c - '0');
        if (value > (limit - digit) / 10) {
          throw std::runtime_error("Integer overflow");
        }
        value = value * 10 + digit;
      }
      if (neg) {
        if (value == static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1ULL) {
          return Value(std::numeric_limits<int64_t>::min());
        }
        return Value(-static_cast<int64_t>(value));
      }
      return Value(value);
    }

    char* end_ptr = nullptr;
    std::string tmp(token);
    double val = std::strtod(tmp.c_str(), &end_ptr);
    if (end_ptr != tmp.c_str() + tmp.size()) {
      throw std::runtime_error("Invalid number");
    }
    if (!std::isfinite(val)) {
      throw std::runtime_error("Invalid number");
    }
    return Value(val);
  }

  Value parse_literal(const std::string& literal, Value value) {
    if (input_.substr(pos_, literal.size()) != literal) {
      throw std::runtime_error("Invalid literal");
    }
    pos_ += literal.size();
    return value;
  }

  void skip_ws() {
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool peek(char c) const {
    return pos_ < input_.size() && input_[pos_] == c;
  }

  void expect(char c) {
    if (pos_ >= input_.size() || input_[pos_] != c) {
      throw std::runtime_error("Unexpected character");
    }
    ++pos_;
  }

  std::string_view input_;
  std::size_t pos_;
};

void write_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u00";
          const char* hex = "0123456789ABCDEF";
          out << hex[(c >> 4) & 0x0F] << hex[c & 0x0F];
        } else {
          out << c;
        }
    }
  }
  out << '"';
}

} // namespace

// Number

double Number::as_double() const {
  if (is_int) {
    return is_negative ? -static_cast<double>(uint_value) : static_cast<double>(uint_value);
  }
  return double_value;
}

uint64_t Number::as_uint64() const {
  if (!is_int) {
    throw std::runtime_error("Expected integer for uint64");
  }
  if (is_negative) {
    throw std::runtime_error("Negative number cannot be converted to uint64");
  }
  return uint_value;
}

// Value

Value::Value() : type_(Type::Null), data_(std::monostate{}) {}
Value::Value(std::nullptr_t) : type_(Type::Null), data_(std::monostate{}) {}
Value::Value(bool value) : type_(Type::Bool), data_(value) {}
Value::Value(uint64_t value) : type_(Type::Number), data_(Number{true, false, value, 0.0}) {}
Value::Value(int64_t value)
    : type_(Type::Number),
      data_([&]() {
        bool neg = value < 0;
        uint64_t abs_val = 0;
        if (neg) {
          if (value == std::numeric_limits<int64_t>::min()) {
            abs_val = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1ULL;
          } else {
            abs_val = static_cast<uint64_t>(-value);
          }
        } else {
          abs_val = static_cast<uint64_t>(value);
        }
        return Number{true, neg, abs_val, static_cast<double>(value)};
      }()) {}
Value::Value(double value) : type_(Type::Number), data_(Number{false, false, 0, value}) {}
Value::Value(std::string value) : type_(Type::String), data_(std::move(value)) {}
Value::Value(const char* value) : Value(std::string(value)) {}
Value::Value(Array value) : type_(Type::Array), data_(std::move(value)) {}
Value::Value(Object value) : type_(Type::Object), data_(std::move(value)) {}

bool Value::as_bool() const {
  if (!is_bool()) throw std::runtime_error("Expected bool");
  return std::get<bool>(data_);
}

const Number& Value::as_number() const {
  if (!is_number()) throw std::runtime_error("Expected number");
  return std::get<Number>(data_);
}

const std::string& Value::as_string() const {
  if (!is_string()) throw std::runtime_error("Expected string");
  return std::get<std::string>(data_);
}

const Value::Array& Value::as_array() const {
  if (!is_array()) throw std::runtime_error("Expected array");
  return std::get<Array>(data_);
}

const Value::Object& Value::as_object() const {
  if (!is_object()) throw std::runtime_error("Expected object");
  return std::get<Object>(data_);
}

const Value* Value::find(const std::string& key) const {
  if (!is_object()) return nullptr;
  const auto& obj = std::get<Object>(data_);
  for (const auto& [k, v] : obj) {
    if (k == key) return &v;
  }
  return nullptr;
}

// Parse / Write

Value parse(std::string_view input) {
  Parser parser(input);
  Value value = parser.parse_value();
  parser.ensure_end();
  return value;
}

Value parse_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Could not read JSON file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse(buffer.str());
}

void write(std::ostream& out, const Value& value) {
  switch (value.type()) {
    case Value::Type::Null:
      out << "null";
      break;
    case Value::Type::Bool:
      out << (value.as_bool() ? "true" : "false");
      break;
    case Value::Type::Number: {
      const auto& num = value.as_number();
      if (num.is_int) {
        if (num.is_negative) {
          out << '-' << num.uint_value;
        } else {
          out << num.uint_value;
        }
      } else {
        out.setf(std::ios::fmtflags(0), std::ios::floatfield);
        out << std::setprecision(std::numeric_limits<double>::max_digits10) << num.double_value;
      }
      break;
    }
    case Value::Type::String:
      write_string(out, value.as_string());
      break;
    case Value::Type::Array: {
      out << '[';
      const auto& arr = value.as_array();
      for (std::size_t i = 0; i < arr.size(); ++i) {
        if (i > 0) out << ',';
        write(out, arr[i]);
      }
      out << ']';
      break;
    }
    case Value::Type::Object: {
      out << '{';
      const auto& obj = value.as_object();
      for (std::size_t i = 0; i < obj.size(); ++i) {
        if (i > 0) out << ',';
        write_string(out, obj[i].first);
        out << ':';
        write(out, obj[i].second);
      }
      out << '}';
      break;
    }
  }
}

} // namespace json
} // namespace rm_model
