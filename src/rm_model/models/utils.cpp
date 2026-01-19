#include "rm_model/models/utils.h"

#include <algorithm>
#include <cassert>

namespace rm_model {

uint8_t num_bits(uint64_t largest_target) {
  uint8_t nbits = 0;
  while (((1ULL << (nbits + 1)) - 1) <= largest_target) {
    nbits += 1;
  }
  assert(nbits >= 1);
  return nbits;
}

namespace {
uint8_t common_prefix_size2(const std::vector<uint64_t>& data) {
  uint64_t any_ones = 0;
  uint64_t no_ones = ~0ULL;

  for (auto x : data) {
    any_ones |= x;
    no_ones &= x;
  }

  uint64_t any_zeros = ~no_ones;
  uint64_t prefix_bits = any_zeros ^ any_ones;
  return leading_zeros_64(~prefix_bits);
}
}

std::vector<uint64_t> radix_index(const std::vector<uint64_t>& points, uint8_t num_bits) {
  uint8_t cps = common_prefix_size2(points);
  if (cps != 0) {
    RM_MODEL_LOG_WARN("Radix index assumes common prefix size 0, but got " << static_cast<int>(cps));
  }

  std::vector<uint64_t> radix_index_vec(1ULL << num_bits, 0);

  uint64_t last_radix = 0;
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    uint64_t p = points[idx];
    uint64_t radix = p >> (64 - num_bits);
    assert(radix < radix_index_vec.size());

    if (radix == last_radix) continue;

    for (uint64_t i = last_radix + 1; i < radix; ++i) {
      radix_index_vec[i] = idx;
    }
    radix_index_vec[radix] = idx;
    last_radix = radix;
  }

  for (uint64_t i = last_radix + 1; i < radix_index_vec.size(); ++i) {
    radix_index_vec[i] = points.size();
  }

  radix_index_vec.push_back(points.size());

#ifndef NDEBUG
  for (auto p : points) {
    uint64_t radix = p >> (64 - num_bits);
    uint64_t radix_lb = radix_index_vec[radix];
    uint64_t radix_ub = radix_index_vec[radix + 1];

    auto ub = std::upper_bound(points.begin(), points.end(), p);
    uint64_t correct_idx = static_cast<uint64_t>(std::distance(points.begin(), ub) - 1);
    assert(radix_lb <= correct_idx);
    assert(radix_ub > correct_idx);
  }
#endif

  return radix_index_vec;
}

} // namespace rm_model
