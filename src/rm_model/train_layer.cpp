#include "rm_model/train.h"

#include "rm_model/logging.h"
#include "rm_model/models/balanced_radix.h"
#include "rm_model/models/cubic_spline.h"
#include "rm_model/models/histogram.h"
#include "rm_model/models/linear.h"
#include "rm_model/models/linear_spline.h"
#include "rm_model/models/normal.h"
#include "rm_model/models/radix.h"
#include "rm_model/train_lower_bound_correction.h"
#include "rm_model/parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace rm_model {

namespace {

uint64_t error_between(uint64_t v1, uint64_t v2, uint64_t max_pred) {
  uint64_t pred1 = std::min(v1, max_pred);
  uint64_t pred2 = std::min(v2, max_pred);
  return std::max(pred1, pred2) - std::min(pred1, pred2);
}

std::size_t max_leaf_models_limit(std::size_t num_rows) {
  const char* env = std::getenv("RM_MODEL_SELECTOR_MAX_LEAF_MODELS");
  std::size_t limit = 0;
  if (!env) {
    limit = 1000000;
  } else {
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (!end || *end != '\0' || value < 1) {
      throw std::runtime_error("Invalid RM_MODEL_SELECTOR_MAX_LEAF_MODELS (must be >= 1)");
    }
    limit = static_cast<std::size_t>(value);
  }

  if (num_rows > 0 && limit > num_rows) {
    limit = num_rows;
  }
  return limit;
}

template <typename T>
std::unique_ptr<Model> train_model(const std::string& model_type, const TrainingData<T>& data) {
  if (model_type == "linear") return std::make_unique<LinearModel>(data);
  if (model_type == "robust_linear") return std::make_unique<RobustLinearModel>(data);
  if (model_type == "linear_spline") return std::make_unique<LinearSplineModel>(data);
  if (model_type == "cubic") return std::make_unique<CubicSplineModel>(data);
  if (model_type == "loglinear") return std::make_unique<LogLinearModel>(data);
  if (model_type == "normal") return std::make_unique<NormalModel>(data);
  if (model_type == "lognormal") return std::make_unique<LogNormalModel>(data);
  if (model_type == "radix") return std::make_unique<RadixModel>(data);
  if (model_type == "radix8") return std::make_unique<RadixTable>(data, 8);
  if (model_type == "radix18") return std::make_unique<RadixTable>(data, 18);
  if (model_type == "radix22") return std::make_unique<RadixTable>(data, 22);
  if (model_type == "radix26") return std::make_unique<RadixTable>(data, 26);
  if (model_type == "radix28") return std::make_unique<RadixTable>(data, 28);
  if (model_type == "bradix") return std::make_unique<BalancedRadixModel>(data);
  if (model_type == "histogram") return std::make_unique<EquidepthHistogramModel>(data);
  throw std::runtime_error("Unknown model type: " + model_type);
}

template <typename T>
std::vector<std::unique_ptr<Model>> build_models_from(
    const TrainingData<T>& data,
    const Model& top_model,
    const std::string& model_type,
    std::size_t start_idx,
    std::size_t end_idx,
    std::size_t first_model_idx,
    std::size_t num_models) {
  if (end_idx <= start_idx || end_idx > data.len()) {
    throw std::runtime_error("Invalid range for build_models_from");
  }
  if (num_models == 0) {
    throw std::runtime_error("Cannot build models with zero leaf models");
  }

  auto dummy_md = TrainingData<T>::empty();
  std::vector<std::unique_ptr<Model>> leaf_models(num_models);
  using value_type = typename TrainingData<T>::value_type;

  std::vector<value_type> second_layer_data;
  const std::size_t reserve_hint = (end_idx - start_idx) / num_models + 2;
  second_layer_data.reserve(reserve_hint);

  auto train_leaf = [&](std::size_t leaf_idx, std::vector<value_type> data_vec) {
    TrainingData<T> container(
        std::make_shared<typename TrainingData<T>::VectorProvider>(std::move(data_vec)));
    leaf_models[leaf_idx - first_model_idx] = train_model(model_type, container);
  };

  auto fill_empty = [&](std::size_t from, std::size_t to) {
    for (std::size_t skip = from; skip < to; ++skip) {
      leaf_models[skip - first_model_idx] = train_model(model_type, dummy_md);
    }
  };

  const bool use_parallel = thread_count() > 1 && !in_parallel_region();
  std::optional<TaskGroup> tasks;
  if (use_parallel) {
    tasks.emplace(thread_count() * 2);
  }

  auto schedule_leaf = [&](std::size_t leaf_idx, std::vector<value_type> data_vec) {
    if (!use_parallel) {
      train_leaf(leaf_idx, std::move(data_vec));
      return;
    }
    tasks->schedule([&, leaf_idx, data = std::move(data_vec)]() mutable {
      train_leaf(leaf_idx, std::move(data));
    });
  };

  std::size_t last_target = first_model_idx;
  std::size_t idx = 0;
  for (const auto& [x, y] : data.iter()) {
    if (idx < start_idx) {
      ++idx;
      continue;
    }
    if (idx >= end_idx) break;

    std::size_t model_pred = static_cast<std::size_t>(top_model.predict_to_int(TrainingKeyOps<T>::to_model_input(x)));
    if (!top_model.needs_bounds_check() && model_pred >= first_model_idx + num_models) {
      throw std::runtime_error("Top model prediction out of bounds");
    }

    std::size_t target = std::min(first_model_idx + num_models - 1, model_pred);
    if (target < last_target) {
      throw std::runtime_error("Non-monotonic target index in build_models_from");
    }

    if (target > last_target) {
      auto last_item = second_layer_data.empty() ? std::optional<std::pair<T, std::size_t>>()
                                                 : std::optional<std::pair<T, std::size_t>>(second_layer_data.back());
      second_layer_data.emplace_back(x, y);

      schedule_leaf(last_target, std::move(second_layer_data));
      fill_empty(last_target + 1, target);

      second_layer_data = std::vector<value_type>();
      second_layer_data.reserve(reserve_hint);
      if (last_item.has_value()) {
        second_layer_data.push_back(*last_item);
      }
    }

    second_layer_data.emplace_back(x, y);
    last_target = target;
    ++idx;
  }

  if (second_layer_data.empty()) {
    throw std::runtime_error("No data for final leaf model");
  }

  schedule_leaf(last_target, std::move(second_layer_data));
  fill_empty(last_target + 1, first_model_idx + num_models);
  if (use_parallel) {
    tasks->wait();
  }
  for (const auto& model : leaf_models) {
    if (!model) {
      throw std::runtime_error("Unexpected number of leaf models built");
    }
  }

  return leaf_models;
}

template <typename T>
void validate_models(const std::vector<std::string>& model_spec) {
  std::size_t num_layers = model_spec.size();
  auto empty = TrainingData<T>::empty();

  for (std::size_t idx = 0; idx < model_spec.size(); ++idx) {
    auto model = train_model(model_spec[idx], empty);
    switch (model->restriction()) {
      case ModelRestriction::None:
        break;
      case ModelRestriction::MustBeTop:
        if (idx != 0) {
          throw std::runtime_error("Model type must be the root model: " + model_spec[idx]);
        }
        break;
      case ModelRestriction::MustBeBottom:
        if (idx != num_layers - 1) {
          throw std::runtime_error("Model type must be the bottom model: " + model_spec[idx]);
        }
        break;
    }
  }
}

std::string join_models(const std::vector<std::string>& layers, const std::string& last_model) {
  std::string out;
  for (const auto& layer : layers) {
    if (!out.empty()) out += ",";
    out += layer;
  }
  if (!last_model.empty()) {
    if (!out.empty()) out += ",";
    out += last_model;
  }
  return out;
}

inline std::size_t clamp_index(uint64_t pred, std::size_t bound, bool needs_check) {
  if (bound == 0) return 0;
  if (!needs_check) {
    return static_cast<std::size_t>(pred);
  }
  uint64_t max_val = static_cast<uint64_t>(bound - 1);
  return static_cast<std::size_t>(pred > max_val ? max_val : pred);
}

inline std::size_t clamp_index(double pred, std::size_t bound, bool needs_check) {
  if (bound == 0) return 0;
  if (!needs_check) {
    return static_cast<std::size_t>(pred);
  }
  if (pred < 0.0) return 0;
  double max_val = static_cast<double>(bound - 1);
  return static_cast<std::size_t>(pred > max_val ? max_val : pred);
}

template <typename T>
std::size_t predict_leaf_index(const std::vector<std::vector<std::unique_ptr<Model>>>& layers,
                               T key,
                               std::size_t leaf_count) {
  if (leaf_count == 0) return 0;

  ModelInput input = TrainingKeyOps<T>::to_model_input(key);
  ModelDataType last_output = std::is_same_v<T, double> ? ModelDataType::Float : ModelDataType::Int;
  bool needs_check = true;
  uint64_t ipred = 0;
  double fpred = 0.0;
  std::size_t model_index = 0;

  for (const auto& layer : layers) {
    if (layer.empty()) {
      throw std::runtime_error("Encountered empty model layer");
    }
    if (layer.size() > 1) {
      if (last_output == ModelDataType::Float) {
        model_index = clamp_index(fpred, layer.size(), needs_check);
      } else {
        model_index = clamp_index(ipred, layer.size(), needs_check);
      }
    } else {
      model_index = 0;
    }

    const auto& model = layer[model_index];
    if (model->output_type() == ModelDataType::Float) {
      fpred = model->predict_to_float(input);
    } else {
      ipred = model->predict_to_int(input);
    }
    last_output = model->output_type();
    needs_check = model->needs_bounds_check();
  }

  if (last_output == ModelDataType::Float) {
    return clamp_index(fpred, leaf_count, needs_check);
  }
  return clamp_index(ipred, leaf_count, needs_check);
}

} // namespace

template <typename T>
TrainedModel train_two_layer(TrainingData<T>& data,
                           const std::string& layer1_model,
                           const std::string& layer2_model,
                           uint64_t num_leaf_models) {
  validate_models<T>({layer1_model, layer2_model});
  if (num_leaf_models <= 1) {
    throw std::runtime_error("Branching factor must be >= 2");
  }

  std::size_t num_rows = data.len();
  if (num_rows == 0) {
    throw std::runtime_error("Cannot train on empty data");
  }

  RM_MODEL_LOG_TRACE("Training top-level " << layer1_model << " model layer");
  data.set_scale(static_cast<double>(num_leaf_models) / static_cast<double>(num_rows));
  auto top_model = train_model(layer1_model, data);

#ifndef NDEBUG
  uint64_t last_pred = 0;
  for (const auto& [x, _y] : data.iter_model_input()) {
    uint64_t prediction = top_model->predict_to_int(x);
    if (prediction < last_pred) {
      throw std::runtime_error("Top model was non-monotonic");
    }
    last_pred = prediction;
  }
  RM_MODEL_LOG_TRACE("Top model was monotonic");
#endif

  RM_MODEL_LOG_TRACE("Training second-level " << layer2_model << " model layer (num models = "
                                        << num_leaf_models << ")");
  data.set_scale(1.0);

  uint64_t midpoint_model = num_leaf_models / 2;
  std::size_t split_idx = data.lower_bound_by([&](const typename TrainingData<T>::value_type& item) {
    uint64_t model_idx = top_model->predict_to_int(TrainingKeyOps<T>::to_model_input(item.first));
    uint64_t model_target = std::min<uint64_t>(num_leaf_models - 1, model_idx);
    if (model_target < midpoint_model) return -1;
    if (model_target > midpoint_model) return 1;
    return 0;
  });

  if (split_idx > 0 && split_idx < data.len()) {
    uint64_t key_at = top_model->predict_to_int(TrainingKeyOps<T>::to_model_input(data.get_key(split_idx)));
    uint64_t key_prev = top_model->predict_to_int(TrainingKeyOps<T>::to_model_input(data.get_key(split_idx - 1)));
    if (key_at <= key_prev) {
      throw std::runtime_error("Split point not strictly increasing");
    }
  }

  std::vector<std::unique_ptr<Model>> leaf_models;
  if (split_idx >= data.len()) {
    leaf_models = build_models_from(data, *top_model, layer2_model, 0, data.len(), 0,
                                    static_cast<std::size_t>(num_leaf_models));
  } else {
    std::size_t split_target = static_cast<std::size_t>(std::min<uint64_t>(
        num_leaf_models - 1,
        top_model->predict_to_int(TrainingKeyOps<T>::to_model_input(data.get_key(split_idx)))));

    std::size_t first_half = split_target;
    std::size_t second_half = static_cast<std::size_t>(num_leaf_models) - split_target;

    std::vector<std::unique_ptr<Model>> hf1;
    std::vector<std::unique_ptr<Model>> hf2;

    join(
        [&]() {
          hf1 = build_models_from(data, *top_model, layer2_model, 0, split_idx, 0, first_half);
        },
        [&]() {
          hf2 = build_models_from(data, *top_model, layer2_model, split_idx + 1, data.len(),
                                  split_target, second_half);
        });

    leaf_models.reserve(hf1.size() + hf2.size());
    for (auto& model : hf1) {
      leaf_models.push_back(std::move(model));
    }
    for (auto& model : hf2) {
      leaf_models.push_back(std::move(model));
    }
  }

  RM_MODEL_LOG_TRACE("Computing lower bound stats...");
  LowerBoundCorrection<T> lb_corrections(
      [&](T key) { return top_model->predict_to_int(TrainingKeyOps<T>::to_model_input(key)); },
      num_leaf_models, data);

  RM_MODEL_LOG_TRACE("Fixing empty models...");
  bool could_not_replace = false;
  for (std::size_t idx = 0; idx + 1 < static_cast<std::size_t>(num_leaf_models); ++idx) {
    bool has_first = lb_corrections.first_key(idx).has_value();
    bool has_last = lb_corrections.last_key(idx).has_value();
    if (has_first != has_last) {
      throw std::runtime_error("Inconsistent lower bound correction data");
    }

    if (!has_last) {
      std::size_t upper_bound = lb_corrections.next_index(idx);
      if (!leaf_models[idx]->set_to_constant_model(static_cast<uint64_t>(upper_bound))) {
        could_not_replace = true;
      }
    }
  }

  if (could_not_replace) {
    RM_MODEL_LOG_WARN("Some empty models could not be replaced with constants; performance may suffer");
  }

  RM_MODEL_LOG_TRACE("Computing last level errors...");
  std::vector<std::pair<uint64_t, uint64_t>> last_layer_max_l1s(num_leaf_models, {0, 0});
  double point_abs_sum = 0.0;
  double point_sq_sum = 0.0;
  for (const auto& [x, y] : data.iter_model_input()) {
    uint64_t leaf_idx = top_model->predict_to_int(x);
    std::size_t target = static_cast<std::size_t>(std::min<uint64_t>(num_leaf_models - 1, leaf_idx));

    uint64_t pred = leaf_models[target]->predict_to_int(x);
    uint64_t err = error_between(pred, static_cast<uint64_t>(y), static_cast<uint64_t>(data.len()));
    double err_d = static_cast<double>(err);
    point_abs_sum += err_d;
    point_sq_sum += err_d * err_d;

    auto cur_val = last_layer_max_l1s[target];
    last_layer_max_l1s[target] = {cur_val.first + 1, std::max(err, cur_val.second)};
  }

  std::size_t large_corrections = 0;
  for (std::size_t leaf_idx = 0; leaf_idx < static_cast<std::size_t>(num_leaf_models); ++leaf_idx) {
    uint64_t curr_err = last_layer_max_l1s[leaf_idx].second;

    auto [idx_of_next, key_of_next] = lb_corrections.next(leaf_idx);
    uint64_t pred_upper = leaf_models[leaf_idx]->predict_to_int(
        TrainingKeyOps<T>::to_model_input(TrainingKeyOps<T>::minus_epsilon(key_of_next)));
    uint64_t upper_error = error_between(pred_upper, static_cast<uint64_t>(idx_of_next + 1),
                                         static_cast<uint64_t>(data.len()));

    T first_key_before = lb_corrections.prev_key(leaf_idx);
    std::size_t prev_idx = (leaf_idx == 0) ? 0 : leaf_idx - 1;
    std::size_t first_idx = lb_corrections.next_index(prev_idx);
    uint64_t pred_lower = leaf_models[leaf_idx]->predict_to_int(
        TrainingKeyOps<T>::to_model_input(TrainingKeyOps<T>::plus_epsilon(first_key_before)));
    uint64_t lower_error = error_between(pred_lower, static_cast<uint64_t>(first_idx),
                                         static_cast<uint64_t>(data.len()));

    uint64_t new_err = std::max({curr_err, upper_error, lower_error}) +
                       lb_corrections.longest_run(leaf_idx);

    uint64_t num_items = last_layer_max_l1s[leaf_idx].first;
    last_layer_max_l1s[leaf_idx] = {num_items, new_err};

    if (new_err > curr_err + 512 && num_items > 100) {
      large_corrections += 1;
    }
  }

  if (large_corrections > 1) {
    RM_MODEL_LOG_TRACE("Of " << num_leaf_models << " models, " << large_corrections
                        << " needed large lower bound corrections.");
  }

  RM_MODEL_LOG_TRACE("Evaluating two-layer model...");
  auto max_it = std::max_element(last_layer_max_l1s.begin(), last_layer_max_l1s.end(),
                                 [](const auto& a, const auto& b) { return a.second < b.second; });
  std::size_t model_max_error_idx = static_cast<std::size_t>(std::distance(last_layer_max_l1s.begin(), max_it));
  uint64_t model_max_error = max_it->second;

  double model_avg_error = 0.0;
  double model_avg_l2_error = 0.0;
  double model_avg_log2_error = 0.0;

  for (const auto& [n, err] : last_layer_max_l1s) {
    model_avg_error += static_cast<double>(n) * static_cast<double>(err);
    double err_d = static_cast<double>(err);
    model_avg_l2_error += static_cast<double>(n) * err_d * err_d;
    model_avg_log2_error += static_cast<double>(n) * std::log2(static_cast<double>(2 * err + 2));
  }

  model_avg_error /= static_cast<double>(num_rows);
  model_avg_l2_error /= static_cast<double>(num_rows);
  model_avg_log2_error /= static_cast<double>(num_rows);
  double model_max_log2_error =
      std::log2(static_cast<double>(2 * model_max_error + 2));
  double model_point_mae = point_abs_sum / static_cast<double>(num_rows);
  double model_point_rmse = std::sqrt(point_sq_sum / static_cast<double>(num_rows));

  std::vector<uint64_t> final_errors;
  final_errors.reserve(last_layer_max_l1s.size());
  for (const auto& [_n, err] : last_layer_max_l1s) {
    final_errors.push_back(err);
  }

  TrainedModel result;
  result.num_model_rows = data.len();
  result.num_data_rows = data.len();
  result.model_avg_error = model_avg_error;
  result.model_avg_l2_error = model_avg_l2_error;
  result.model_avg_log2_error = model_avg_log2_error;
  result.model_point_mae = model_point_mae;
  result.model_point_rmse = model_point_rmse;
  result.model_max_error = model_max_error;
  result.model_max_error_idx = model_max_error_idx;
  result.model_max_log2_error = model_max_log2_error;
  result.last_layer_max_l1s = std::move(final_errors);
  result.model_layers.push_back({});
  result.model_layers.back().push_back(std::move(top_model));
  result.model_layers.push_back(std::move(leaf_models));
  result.model_spec = layer1_model + "," + layer2_model;
  result.branching_factor = num_leaf_models;
  result.cache_fix = std::nullopt;
  return result;
}

template <typename T>
TrainedModel train_multi_layer(TrainingData<T>& data,
                             const std::vector<std::string>& model_list,
                             const std::string& last_model,
                             uint64_t branch_factor) {
  if (model_list.empty()) {
    throw std::runtime_error("Multi-layer training requires at least one intermediate layer");
  }
  if (branch_factor <= 1) {
    throw std::runtime_error("Branching factor must be >= 2");
  }

  std::size_t num_rows = data.len();
  if (num_rows == 0) {
    throw std::runtime_error("Cannot train on empty data");
  }
  std::size_t max_leaf_models = max_leaf_models_limit(num_rows);

  using value_type = typename TrainingData<T>::value_type;

  std::vector<std::vector<value_type>> data_partitions;
  data_partitions.emplace_back();
  data_partitions.back().reserve(num_rows);
  for (const auto& item : data.iter()) {
    data_partitions.back().push_back(item);
  }

  std::vector<std::vector<std::unique_ptr<Model>>> model_layers;
  uint64_t current_model_count = 1;

  for (const auto& model_type : model_list) {
    RM_MODEL_LOG_INFO("Training " << model_type << " model layer");
    if (current_model_count > std::numeric_limits<uint64_t>::max() / branch_factor) {
      throw std::runtime_error("Branching factor overflow for multi-layer training");
    }
    uint64_t next_layer_size_u64 = current_model_count * branch_factor;
    if (next_layer_size_u64 == 0) {
      throw std::runtime_error("Invalid next layer size");
    }
    if (next_layer_size_u64 > static_cast<uint64_t>(max_leaf_models)) {
      throw std::runtime_error("Next layer would create " + std::to_string(next_layer_size_u64) +
                               " models, exceeding limit " + std::to_string(max_leaf_models) +
                               " (set RM_MODEL_SELECTOR_MAX_LEAF_MODELS to override)");
    }
    if (next_layer_size_u64 > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("Next layer size exceeds addressable memory");
    }

    std::size_t next_layer_size = static_cast<std::size_t>(next_layer_size_u64);
    std::size_t per_bucket = num_rows / std::max<std::size_t>(1, next_layer_size);
    if (per_bucket == 0) {
      per_bucket = 1;
    }

    std::vector<std::vector<value_type>> next_layer_data;
    try {
      next_layer_data.resize(next_layer_size);
      if (per_bucket > 1) {
        for (auto& bucket : next_layer_data) {
          bucket.reserve(per_bucket);
        }
      }
    } catch (const std::bad_alloc&) {
      throw std::runtime_error("Not enough memory for layer buckets ("
                               + std::to_string(next_layer_size)
                               + " buckets). Reduce branching factor or lower "
                               "RM_MODEL_SELECTOR_MAX_LEAF_MODELS.");
    }

    std::vector<std::unique_ptr<Model>> models;
    models.reserve(data_partitions.size());

    for (auto& partition : data_partitions) {
      TrainingData<T> md_container(
          std::make_shared<typename TrainingData<T>::VectorProvider>(std::move(partition)));

      md_container.set_scale(static_cast<double>(next_layer_size_u64) /
                             static_cast<double>(num_rows));
      auto model = train_model(model_type, md_container);
      md_container.set_scale(1.0);

      for (const auto& [x, y] : md_container.iter()) {
        uint64_t model_pred = model->predict_to_int(TrainingKeyOps<T>::to_model_input(x));
        if (!model->needs_bounds_check() && model_pred >= next_layer_size_u64) {
          throw std::runtime_error("Model prediction out of bounds");
        }
        std::size_t target = static_cast<std::size_t>(
            std::min<uint64_t>(next_layer_size_u64 - 1, model_pred));
        next_layer_data[target].emplace_back(x, y);
      }

      models.push_back(std::move(model));
    }

    data_partitions = std::move(next_layer_data);
    current_model_count = next_layer_size_u64;
    model_layers.push_back(std::move(models));
  }

  RM_MODEL_LOG_INFO("Training last level " << last_model << " model");
  std::vector<std::unique_ptr<Model>> last_layer;
  last_layer.reserve(data_partitions.size());

  for (std::size_t midx = 0; midx < data_partitions.size(); ++midx) {
    TrainingData<T> md_container(
        std::make_shared<typename TrainingData<T>::VectorProvider>(
            std::move(data_partitions[midx])));
    auto model = train_model(last_model, md_container);

    last_layer.push_back(std::move(model));
  }

  std::size_t num_leaf_models = last_layer.size();
  auto leaf_index = [&](T key) -> uint64_t {
    return static_cast<uint64_t>(predict_leaf_index(model_layers, key, num_leaf_models));
  };

  RM_MODEL_LOG_TRACE("Computing lower bound stats...");
  LowerBoundCorrection<T> lb_corrections(leaf_index, num_leaf_models, data);

  RM_MODEL_LOG_TRACE("Computing last level errors...");
  std::vector<std::pair<uint64_t, uint64_t>> last_layer_max_l1s(num_leaf_models, {0, 0});
  double point_abs_sum = 0.0;
  double point_sq_sum = 0.0;
  for (const auto& [x, y] : data.iter()) {
    std::size_t target = static_cast<std::size_t>(
        std::min<uint64_t>(num_leaf_models - 1, leaf_index(x)));

    uint64_t pred = last_layer[target]->predict_to_int(TrainingKeyOps<T>::to_model_input(x));
    uint64_t err = error_between(pred, static_cast<uint64_t>(y), static_cast<uint64_t>(data.len()));
    double err_d = static_cast<double>(err);
    point_abs_sum += err_d;
    point_sq_sum += err_d * err_d;

    if (auto bound = last_layer[target]->error_bound()) {
      if (err > *bound) {
        RM_MODEL_LOG_WARN("Model error bound exceeded: bound=" << *bound
                                                         << " err=" << err
                                                         << " key=" << TrainingKeyOps<T>::as_uint(x));
      }
    }

    auto cur_val = last_layer_max_l1s[target];
    last_layer_max_l1s[target] = {cur_val.first + 1, std::max(err, cur_val.second)};
  }

  std::size_t large_corrections = 0;
  for (std::size_t leaf_idx = 0; leaf_idx < num_leaf_models; ++leaf_idx) {
    uint64_t curr_err = last_layer_max_l1s[leaf_idx].second;

    auto [idx_of_next, key_of_next] = lb_corrections.next(leaf_idx);
    uint64_t pred_upper = last_layer[leaf_idx]->predict_to_int(
        TrainingKeyOps<T>::to_model_input(TrainingKeyOps<T>::minus_epsilon(key_of_next)));
    uint64_t upper_error = error_between(pred_upper, static_cast<uint64_t>(idx_of_next + 1),
                                         static_cast<uint64_t>(data.len()));

    T first_key_before = lb_corrections.prev_key(leaf_idx);
    std::size_t prev_idx = (leaf_idx == 0) ? 0 : leaf_idx - 1;
    std::size_t first_idx = lb_corrections.next_index(prev_idx);
    uint64_t pred_lower = last_layer[leaf_idx]->predict_to_int(
        TrainingKeyOps<T>::to_model_input(TrainingKeyOps<T>::plus_epsilon(first_key_before)));
    uint64_t lower_error = error_between(pred_lower, static_cast<uint64_t>(first_idx),
                                         static_cast<uint64_t>(data.len()));

    uint64_t new_err = std::max({curr_err, upper_error, lower_error}) +
                       lb_corrections.longest_run(leaf_idx);

    uint64_t num_items = last_layer_max_l1s[leaf_idx].first;
    last_layer_max_l1s[leaf_idx] = {num_items, new_err};

    if (new_err > curr_err + 512 && num_items > 100) {
      large_corrections += 1;
    }
  }

  if (large_corrections > 1) {
    RM_MODEL_LOG_TRACE("Of " << num_leaf_models << " models, " << large_corrections
                        << " needed large lower bound corrections.");
  }

  model_layers.push_back(std::move(last_layer));

  auto max_it = std::max_element(last_layer_max_l1s.begin(), last_layer_max_l1s.end(),
                                 [](const auto& a, const auto& b) { return a.second < b.second; });
  std::size_t model_max_error_idx =
      static_cast<std::size_t>(std::distance(last_layer_max_l1s.begin(), max_it));
  uint64_t model_max_error = max_it->second;

  double model_avg_error = 0.0;
  double model_avg_l2_error = 0.0;
  double model_avg_log2_error = 0.0;

  for (const auto& [n, err] : last_layer_max_l1s) {
    model_avg_error += static_cast<double>(n) * static_cast<double>(err);
    double err_d = static_cast<double>(err);
    model_avg_l2_error += static_cast<double>(n) * err_d * err_d;
    model_avg_log2_error += static_cast<double>(n) * std::log2(static_cast<double>(2 * err + 2));
  }

  model_avg_error /= static_cast<double>(num_rows);
  model_avg_l2_error /= static_cast<double>(num_rows);
  model_avg_log2_error /= static_cast<double>(num_rows);
  double model_max_log2_error =
      std::log2(static_cast<double>(2 * model_max_error + 2));
  double model_point_mae = point_abs_sum / static_cast<double>(num_rows);
  double model_point_rmse = std::sqrt(point_sq_sum / static_cast<double>(num_rows));

  std::vector<uint64_t> final_errors;
  final_errors.reserve(last_layer_max_l1s.size());
  for (const auto& [_n, err] : last_layer_max_l1s) {
    final_errors.push_back(err);
  }

  TrainedModel result;
  result.num_model_rows = data.len();
  result.num_data_rows = data.len();
  result.model_avg_error = model_avg_error;
  result.model_avg_l2_error = model_avg_l2_error;
  result.model_avg_log2_error = model_avg_log2_error;
  result.model_point_mae = model_point_mae;
  result.model_point_rmse = model_point_rmse;
  result.model_max_error = model_max_error;
  result.model_max_error_idx = model_max_error_idx;
  result.model_max_log2_error = model_max_log2_error;
  result.last_layer_max_l1s = std::move(final_errors);
  result.model_layers = std::move(model_layers);
  result.model_spec = join_models(model_list, last_model);
  result.branching_factor = branch_factor;
  result.cache_fix = std::nullopt;
  return result;
}

template TrainedModel train_two_layer<uint64_t>(TrainingData<uint64_t>&, const std::string&, const std::string&, uint64_t);
template TrainedModel train_two_layer<uint32_t>(TrainingData<uint32_t>&, const std::string&, const std::string&, uint64_t);
template TrainedModel train_two_layer<double>(TrainingData<double>&, const std::string&, const std::string&, uint64_t);
template TrainedModel train_multi_layer<uint64_t>(TrainingData<uint64_t>&, const std::vector<std::string>&, const std::string&, uint64_t);
template TrainedModel train_multi_layer<uint32_t>(TrainingData<uint32_t>&, const std::vector<std::string>&, const std::string&, uint64_t);
template TrainedModel train_multi_layer<double>(TrainingData<double>&, const std::vector<std::string>&, const std::string&, uint64_t);

} // namespace rm_model
