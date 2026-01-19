#ifndef RM_MODEL_TRAIN_LOWER_BOUND_CORRECTION_H
#define RM_MODEL_TRAIN_LOWER_BOUND_CORRECTION_H

#include "rm_model/training_data.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace rm_model {

namespace detail {

template <typename T>
std::optional<std::pair<std::size_t, std::pair<std::size_t, T>>> find_first_below(
    const std::vector<std::optional<std::pair<std::size_t, T>>>& data, std::size_t idx) {
  if (idx == 0) return std::nullopt;
  std::size_t i = idx - 1;
  while (true) {
    if (data[i].has_value()) return std::make_pair(i, *data[i]);
    if (i == 0) return std::nullopt;
    --i;
  }
}

template <typename T>
std::optional<std::pair<std::size_t, std::pair<std::size_t, T>>> find_first_above(
    const std::vector<std::optional<std::pair<std::size_t, T>>>& data, std::size_t idx) {
  if (idx + 1 >= data.size()) return std::nullopt;
  std::size_t i = idx + 1;
  while (true) {
    if (data[i].has_value()) return std::make_pair(i, *data[i]);
    if (i + 1 >= data.size()) return std::nullopt;
    ++i;
  }
}

template <typename T>
std::vector<std::pair<std::size_t, T>> compute_next_for_leaf(
    uint64_t num_leaf_models,
    std::size_t num_keys,
    const std::vector<std::optional<std::pair<std::size_t, T>>>& first_key_for_leaf) {
  std::vector<std::pair<std::size_t, T>> next_for_leaf(num_leaf_models,
                                                       {0, TrainingKeyOps<T>::zero_value()});
  std::size_t idx = 0;
  while (idx < num_leaf_models) {
    auto next = find_first_above(first_key_for_leaf, idx);
    if (next.has_value()) {
      auto [next_leaf_idx, val] = *next;
      for (std::size_t i = idx; i < next_leaf_idx; ++i) {
        next_for_leaf[i] = val;
      }
      idx = next_leaf_idx;
    } else {
      for (std::size_t i = idx; i < num_leaf_models; ++i) {
        next_for_leaf[i] = {num_keys, TrainingKeyOps<T>::max_value()};
      }
      break;
    }
  }
  return next_for_leaf;
}

template <typename T>
std::vector<std::pair<std::size_t, T>> compute_prev_for_leaf(
    uint64_t num_leaf_models,
    const std::vector<std::optional<std::pair<std::size_t, T>>>& last_key_for_leaf) {
  std::vector<std::pair<std::size_t, T>> prev_for_leaf(num_leaf_models,
                                                       {0, TrainingKeyOps<T>::zero_value()});
  std::size_t idx = num_leaf_models - 1;
  while (idx > 0) {
    auto prev = find_first_below(last_key_for_leaf, idx);
    if (prev.has_value()) {
      auto [prev_leaf_idx, val] = *prev;
      for (std::size_t i = prev_leaf_idx + 1; i < idx + 1; ++i) {
        prev_for_leaf[i] = val;
      }
      idx = prev_leaf_idx;
    } else {
      break;
    }
  }
  return prev_for_leaf;
}

} // namespace detail

template <typename T>
class LowerBoundCorrection {
 public:
  template <typename Pred>
  LowerBoundCorrection(Pred pred_func, uint64_t num_leaf_models, const TrainingData<T>& data)
      : first_(num_leaf_models), last_(num_leaf_models),
        next_(num_leaf_models), prev_(num_leaf_models),
        run_lengths_(num_leaf_models, 0) {
    std::vector<std::optional<std::pair<std::size_t, T>>> first_key_for_leaf(num_leaf_models);
    std::vector<std::optional<std::pair<std::size_t, T>>> last_key_for_leaf(num_leaf_models);
    std::vector<uint64_t> max_run_length(num_leaf_models, 0);

    std::size_t last_target = 0;
    uint64_t current_run_length = 0;
    T current_run_key = data.get_key(0);
    for (const auto& [x, y] : data.iter()) {
      uint64_t leaf_idx = pred_func(x);
      std::size_t target = static_cast<std::size_t>(std::min<uint64_t>(num_leaf_models - 1, leaf_idx));

      if (target == last_target && x == current_run_key) {
        current_run_length += 1;
      } else {
        max_run_length[last_target] = std::max(max_run_length[last_target], current_run_length);
        current_run_length = 1;
        current_run_key = x;
        last_target = target;
      }

      if (!first_key_for_leaf[target].has_value()) {
        first_key_for_leaf[target] = std::make_pair(y, x);
      }
      last_key_for_leaf[target] = std::make_pair(y, x);
    }

    next_ = detail::compute_next_for_leaf<T>(num_leaf_models, data.len(), first_key_for_leaf);
    prev_ = detail::compute_prev_for_leaf<T>(num_leaf_models, last_key_for_leaf);

    first_ = std::move(first_key_for_leaf);
    last_ = std::move(last_key_for_leaf);
    run_lengths_ = std::move(max_run_length);
  }

  std::optional<T> first_key(std::size_t leaf_idx) const {
    if (!first_[leaf_idx].has_value()) return std::nullopt;
    return first_[leaf_idx]->second;
  }

  std::optional<T> last_key(std::size_t leaf_idx) const {
    if (!last_[leaf_idx].has_value()) return std::nullopt;
    return last_[leaf_idx]->second;
  }

  std::pair<std::size_t, T> next(std::size_t leaf_idx) const { return next_[leaf_idx]; }

  std::size_t next_index(std::size_t leaf_idx) const { return next_[leaf_idx].first; }

  T prev_key(std::size_t leaf_idx) const { return prev_[leaf_idx].second; }

  uint64_t longest_run(std::size_t leaf_idx) const { return run_lengths_[leaf_idx]; }

 private:
  std::vector<std::optional<std::pair<std::size_t, T>>> first_;
  std::vector<std::optional<std::pair<std::size_t, T>>> last_;
  std::vector<std::pair<std::size_t, T>> next_;
  std::vector<std::pair<std::size_t, T>> prev_;
  std::vector<uint64_t> run_lengths_;
};

} // namespace rm_model

#endif // RM_MODEL_TRAIN_LOWER_BOUND_CORRECTION_H
