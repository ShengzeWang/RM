#ifndef RM_MODEL_MODELS_UTILS_H
#define RM_MODEL_MODELS_UTILS_H

#include "rm_model/training_data.h"
#include "rm_model/logging.h"

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rm_model {

uint8_t num_bits(uint64_t largest_target);

inline uint8_t leading_zeros_64(uint64_t value) {
  if (value == 0) return 64;
#if defined(_MSC_VER)
  return static_cast<uint8_t>(__lzcnt64(value));
#else
  return static_cast<uint8_t>(__builtin_clzll(value));
#endif
}

template <typename T>
uint8_t common_prefix_size(const TrainingData<T>& data) {
  uint64_t any_ones = 0;
  uint64_t no_ones = ~0ULL;

  for (const auto& [x, _y] : data.iter_model_input()) {
    any_ones |= x.as_int();
    no_ones &= x.as_int();
  }

  uint64_t any_zeros = ~no_ones;
  uint64_t prefix_bits = any_zeros ^ any_ones;
  return leading_zeros_64(~prefix_bits);
}

std::vector<uint64_t> radix_index(const std::vector<uint64_t>& points, uint8_t num_bits);

} // namespace rm_model

#endif // RM_MODEL_MODELS_UTILS_H
