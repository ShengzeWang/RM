#include "rm_model/models/balanced_radix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace rm_model {

namespace {

template <typename T>
double chi2(const TrainingData<T>& data, uint64_t max_bin, const BalancedRadixModel& model) {
  std::vector<uint64_t> counts(max_bin, 0);
  for (const auto& [x, _y] : data.iter_model_input()) {
    counts[model.predict_to_int(x)] += 1;
  }

  double expected = static_cast<double>(data.len()) / static_cast<double>(max_bin);
  double score = 0.0;
  for (auto c : counts) {
    double diff = static_cast<double>(c) - expected;
    score += (diff * diff) / expected;
  }
  return score;
}

template <typename T>
BalancedRadixModel bradix(const TrainingData<T>& data, uint64_t max_output) {
  uint8_t bits = num_bits(max_output);
  uint8_t common_prefix = common_prefix_size(data);
  RM_MODEL_LOG_TRACE("Bradix layer common prefix: " << static_cast<int>(common_prefix));

  double best_score = std::numeric_limits<double>::infinity();
  std::optional<BalancedRadixModel> best_model;

  for (uint8_t test_bits = bits; test_bits < std::min<uint8_t>(bits + 2, 64); ++test_bits) {
    uint64_t bits_max = (1ULL << (test_bits + 1)) - 1;

    BalancedRadixModel high(common_prefix, test_bits, max_output - 1, true);
    double high_score = chi2(data, max_output, high);
    RM_MODEL_LOG_TRACE("Bradix high with " << static_cast<int>(test_bits) << " bits had score " << high_score);
    if (high_score < best_score) {
      best_score = high_score;
      best_model = high;
    }

    BalancedRadixModel low(common_prefix, test_bits, max_output - bits_max, false);
    double low_score = chi2(data, max_output, low);
    RM_MODEL_LOG_TRACE("Bradix low with " << static_cast<int>(test_bits) << " bits had score " << low_score);
    if (low_score < best_score) {
      best_score = low_score;
      best_model = low;
    }
  }

  RM_MODEL_LOG_TRACE("Best bradix setup chosen with score " << best_score);
  return *best_model;
}

} // namespace

BalancedRadixModel::BalancedRadixModel(uint8_t prefix, uint8_t bits, uint64_t clamp, bool high)
    : params_{prefix, bits, clamp}, high_(high) {}

BalancedRadixModel::BalancedRadixModel(const TrainingData<uint64_t>& data) {
  if (data.len() == 0) {
    params_ = {0, 0, 0};
    high_ = true;
    return;
  }

  uint64_t largest_value = 0;
  for (const auto& [_x, y] : data.iter()) {
    largest_value = std::max<uint64_t>(largest_value, static_cast<uint64_t>(y));
  }
  *this = bradix(data, largest_value);
}

BalancedRadixModel::BalancedRadixModel(const TrainingData<uint32_t>& data) {
  if (data.len() == 0) {
    params_ = {0, 0, 0};
    high_ = true;
    return;
  }

  uint64_t largest_value = 0;
  for (const auto& [_x, y] : data.iter()) {
    largest_value = std::max<uint64_t>(largest_value, static_cast<uint64_t>(y));
  }
  *this = bradix(data, largest_value);
}

BalancedRadixModel::BalancedRadixModel(const TrainingData<double>& data) {
  if (data.len() == 0) {
    params_ = {0, 0, 0};
    high_ = true;
    return;
  }

  uint64_t largest_value = 0;
  for (const auto& [_x, y] : data.iter()) {
    largest_value = std::max<uint64_t>(largest_value, static_cast<uint64_t>(y));
  }
  *this = bradix(data, largest_value);
}

uint64_t BalancedRadixModel::predict_to_int(const ModelInput& inp) const {
  auto [left_shift, num_bits, clamp] = params_;
  uint64_t value = inp.as_int();
  uint64_t res = (value << left_shift) >> (64 - num_bits);

  if (high_) {
    return std::min(res, clamp);
  }
  return res < clamp ? 0 : res - clamp;
}

std::vector<ModelParam> BalancedRadixModel::params() const {
  auto [prefix, bits, clamp] = params_;
  return {ModelParam(static_cast<uint64_t>(prefix)),
          ModelParam(static_cast<uint64_t>(bits)),
          ModelParam(clamp)};
}

std::string BalancedRadixModel::code() const {
  if (high_) {
    return R"(
inline uint64_t bradix_clamp_high(uint64_t prefix_length,
                                  uint64_t bits, uint64_t clamp, uint64_t inp) {
    uint64_t tmp = (inp << prefix_length) >> (64 - bits);
    return (tmp > clamp ? clamp : tmp);
}
)";
  }

  return R"(
inline uint64_t bradix_clamp_low(uint64_t prefix_length,
                                 uint64_t bits, uint64_t clamp, uint64_t inp) {
    uint64_t tmp = (inp << prefix_length) >> (64 - bits);
    return (tmp < clamp ? 0 : tmp - clamp);
}
)";
}

std::string BalancedRadixModel::function_name() const {
  return high_ ? "bradix_clamp_high" : "bradix_clamp_low";
}

} // namespace rm_model
