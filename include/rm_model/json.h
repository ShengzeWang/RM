#ifndef RM_MODEL_JSON_H
#define RM_MODEL_JSON_H

#include <cstdint>
#include <istream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace rm_model {
namespace json {

struct Number {
  bool is_int = false;
  bool is_negative = false;
  uint64_t uint_value = 0;
  double double_value = 0.0;

  double as_double() const;
  uint64_t as_uint64() const;
};

class Value {
 public:
  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
  };

  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;

  Value();
  Value(std::nullptr_t);
  Value(bool value);
  Value(uint64_t value);
  Value(int64_t value);
  Value(double value);
  Value(std::string value);
  Value(const char* value);
  Value(Array value);
  Value(Object value);

  Type type() const { return type_; }

  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const;
  const Number& as_number() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  const Value* find(const std::string& key) const;

 private:
  Type type_;
  std::variant<std::monostate, bool, Number, std::string, Array, Object> data_;
};

Value parse(std::string_view input);
Value parse_file(const std::string& path);
void write(std::ostream& out, const Value& value);

} // namespace json
} // namespace rm_model

#endif // RM_MODEL_JSON_H
