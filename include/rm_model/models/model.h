#ifndef RM_MODEL_MODELS_MODEL_H
#define RM_MODEL_MODELS_MODEL_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace rm_model {

enum class KeyType {
  U32,
  U64,
  F64,
  U128,
};

enum class ModelDataType {
  Int,
  Int128,
  Float,
};

inline const char* c_type(KeyType type) {
  switch (type) {
    case KeyType::U32: return "uint32_t";
    case KeyType::U64: return "uint64_t";
    case KeyType::F64: return "double";
    case KeyType::U128: return "uint128_t";
  }
  return "uint64_t";
}

inline ModelDataType to_model_data_type(KeyType type) {
  switch (type) {
    case KeyType::U32: return ModelDataType::Int;
    case KeyType::U64: return ModelDataType::Int;
    case KeyType::U128: return ModelDataType::Int128;
    case KeyType::F64: return ModelDataType::Float;
  }
  return ModelDataType::Int;
}

inline const char* c_type(ModelDataType type) {
  switch (type) {
    case ModelDataType::Int: return "uint64_t";
    case ModelDataType::Int128: return "uint128_t";
    case ModelDataType::Float: return "double";
  }
  return "uint64_t";
}

class ModelInput {
 public:
  enum class Kind { Int, Float };

  ModelInput() : kind_(Kind::Int), int_val_(0) {}
  ModelInput(uint64_t value) : kind_(Kind::Int), int_val_(value) {}
  ModelInput(uint32_t value) : kind_(Kind::Int), int_val_(value) {}
  ModelInput(int32_t value) : kind_(Kind::Int), int_val_(static_cast<uint64_t>(value)) {}
  ModelInput(double value) : kind_(Kind::Float), float_val_(value) {}

  Kind kind() const { return kind_; }

  double as_float() const {
    return kind_ == Kind::Int ? static_cast<double>(int_val_) : float_val_;
  }

  uint64_t as_int() const {
    return kind_ == Kind::Int ? int_val_ : static_cast<uint64_t>(float_val_);
  }

  ModelInput max_value() const {
    return kind_ == Kind::Int ? ModelInput(std::numeric_limits<uint64_t>::max())
                              : ModelInput(std::numeric_limits<double>::max());
  }

  ModelInput min_value() const {
    return kind_ == Kind::Int ? ModelInput(0ULL)
                              : ModelInput(std::numeric_limits<double>::lowest());
  }

  ModelInput minus_epsilon() const {
    if (kind_ == Kind::Int) {
      return ModelInput(int_val_ == 0 ? 0ULL : int_val_ - 1);
    }
    return ModelInput(float_val_ - std::numeric_limits<double>::epsilon());
  }

  ModelInput plus_epsilon() const {
    if (kind_ == Kind::Int) {
      if (int_val_ == std::numeric_limits<uint64_t>::max()) {
        return ModelInput(int_val_);
      }
      return ModelInput(int_val_ + 1);
    }
    return ModelInput(float_val_ + std::numeric_limits<double>::epsilon());
  }

  bool operator==(const ModelInput& other) const {
    if (kind_ != other.kind_) return false;
    return kind_ == Kind::Int ? int_val_ == other.int_val_ : float_val_ == other.float_val_;
  }

  bool operator!=(const ModelInput& other) const { return !(*this == other); }

  bool operator<(const ModelInput& other) const {
    if (kind_ != other.kind_) return false;
    return kind_ == Kind::Int ? int_val_ < other.int_val_ : float_val_ < other.float_val_;
  }

 private:
  Kind kind_;
  union {
    uint64_t int_val_;
    double float_val_;
  };
};

class ModelParam {
 public:
  using Value = std::variant<uint64_t, double, std::vector<uint16_t>, std::vector<uint64_t>,
                             std::vector<uint32_t>, std::vector<double>>;

  ModelParam(uint64_t value) : value_(value) {}
  ModelParam(double value) : value_(value) {}
  ModelParam(std::vector<uint16_t> value) : value_(std::move(value)) {}
  ModelParam(std::vector<uint64_t> value) : value_(std::move(value)) {}
  ModelParam(std::vector<uint32_t> value) : value_(std::move(value)) {}
  ModelParam(std::vector<double> value) : value_(std::move(value)) {}

  std::size_t size() const;
  const char* c_type() const;
  bool is_array() const;
  const char* c_type_mod() const;
  std::string c_val() const;
  bool is_same_type(const ModelParam& other) const { return value_.index() == other.value_.index(); }
  void write_to(std::ostream& out) const;
  double as_float() const;
  std::size_t len() const;

 private:
  Value value_;
};

enum class ModelRestriction {
  None,
  MustBeTop,
  MustBeBottom,
};

enum class StdFunctions;

class Model {
 public:
  virtual ~Model() = default;

  virtual double predict_to_float(const ModelInput& inp) const {
    return static_cast<double>(predict_to_int(inp));
  }

  virtual uint64_t predict_to_int(const ModelInput& inp) const {
    double val = predict_to_float(inp);
    return static_cast<uint64_t>(std::max(0.0, std::floor(val)));
  }

  virtual ModelDataType input_type() const = 0;
  virtual ModelDataType output_type() const = 0;
  virtual std::vector<ModelParam> params() const = 0;
  virtual std::string code() const = 0;
  virtual std::string function_name() const = 0;
  virtual std::set<StdFunctions> standard_functions() const;
  virtual bool needs_bounds_check() const { return true; }
  virtual ModelRestriction restriction() const { return ModelRestriction::None; }
  virtual std::optional<uint64_t> error_bound() const { return std::nullopt; }
  virtual bool set_to_constant_model(uint64_t /*constant*/) { return false; }
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_MODEL_H
