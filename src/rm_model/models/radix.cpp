#include "rm_model/models/radix.h"

#include <algorithm>

namespace rm_model {

namespace {

template <typename T>
std::pair<uint8_t, uint8_t> radix_params(const TrainingData<T>& data) {
  if (data.len() == 0) {
    return {0, 0};
  }

  uint64_t largest_value = 0;
  for (const auto& [_x, y] : data.iter()) {
    largest_value = std::max<uint64_t>(largest_value, static_cast<uint64_t>(y));
  }

  uint8_t bits = num_bits(largest_value);
  RM_MODEL_LOG_TRACE("Radix layer using " << static_cast<int>(bits) << " bits, from largest value "
                                      << largest_value << " (max layers: "
                                      << ((1ULL << (bits + 1)) - 1) << ")");

  uint8_t common_prefix = common_prefix_size(data);
  RM_MODEL_LOG_TRACE("Radix layer common prefix: " << static_cast<int>(common_prefix));

  return {common_prefix, bits};
}

template <typename T>
RadixTable build_radix_table(const TrainingData<T>& data, uint8_t bits) {
  uint8_t prefix = common_prefix_size(data);
  std::vector<uint32_t> hint_table(1ULL << bits, 0);

  uint64_t last_radix = 0;
  for (const auto& [inp, y] : data.iter_model_input()) {
    uint64_t x = inp.as_int();
    uint8_t num_bits = (prefix + bits > 64) ? 0 : static_cast<uint8_t>(64 - (prefix + bits));
    uint64_t current_radix = ((x << prefix) >> prefix) >> num_bits;
    if (current_radix == last_radix) continue;

    hint_table[current_radix] = static_cast<uint32_t>(y);
    for (uint64_t i = last_radix + 1; i < current_radix; ++i) {
      hint_table[i] = static_cast<uint32_t>(y);
    }
    last_radix = current_radix;
  }

  for (std::size_t i = static_cast<std::size_t>(last_radix) + 1; i < hint_table.size(); ++i) {
    hint_table[i] = static_cast<uint32_t>(hint_table.size());
  }

  return RadixTable(prefix, bits, std::move(hint_table));
}

} // namespace

RadixModel::RadixModel(const TrainingData<uint64_t>& data) : params_(radix_params(data)) {}
RadixModel::RadixModel(const TrainingData<uint32_t>& data) : params_(radix_params(data)) {}
RadixModel::RadixModel(const TrainingData<double>& data) : params_(radix_params(data)) {}

uint64_t RadixModel::predict_to_int(const ModelInput& inp) const {
  auto [left_shift, num_bits] = params_;
  uint64_t value = inp.as_int();
  uint64_t res = (value << left_shift) >> (64 - num_bits);
  return res;
}

std::vector<ModelParam> RadixModel::params() const {
  return {ModelParam(static_cast<uint64_t>(params_.first)), ModelParam(static_cast<uint64_t>(params_.second))};
}

std::string RadixModel::code() const {
  return R"(
inline uint64_t radix(uint64_t prefix_length, uint64_t bits, uint64_t inp) {
    return (inp << prefix_length) >> (64 - bits);
}
)";
}

RadixTable::RadixTable(const TrainingData<uint64_t>& data, uint8_t bits)
    : RadixTable(build_radix_table(data, bits)) {}
RadixTable::RadixTable(const TrainingData<uint32_t>& data, uint8_t bits)
    : RadixTable(build_radix_table(data, bits)) {}
RadixTable::RadixTable(const TrainingData<double>& data, uint8_t bits)
    : RadixTable(build_radix_table(data, bits)) {}

RadixTable::RadixTable(uint8_t prefix_bits, uint8_t table_bits, std::vector<uint32_t> hint_table)
    : prefix_bits_(prefix_bits), table_bits_(table_bits), hint_table_(std::move(hint_table)) {}

uint64_t RadixTable::predict_to_int(const ModelInput& inp) const {
  uint64_t value = inp.as_int();
  uint8_t num_bits = (prefix_bits_ + table_bits_ > 64) ? 0 : static_cast<uint8_t>(64 - (prefix_bits_ + table_bits_));
  uint64_t res = ((value << prefix_bits_) >> prefix_bits_) >> num_bits;
  return static_cast<uint64_t>(hint_table_[res]);
}

std::vector<ModelParam> RadixTable::params() const {
  return {ModelParam(hint_table_)};
}

std::string RadixTable::code() const {
  uint8_t num_bits = (prefix_bits_ + table_bits_ > 64) ? 0 : static_cast<uint8_t>(64 - (prefix_bits_ + table_bits_));
  return std::string("\ninline uint64_t radix_table(const uint32_t* table, const uint64_t inp) {\n") +
         "    return table[((inp << " + std::to_string(prefix_bits_) + ") >> " +
         std::to_string(prefix_bits_) + ") >> " + std::to_string(num_bits) + "];\n}\n";
}

} // namespace rm_model
