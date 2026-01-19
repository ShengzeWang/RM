#include "rm_model/learned_model_selector.h"

#include "rm_model/codegen.h"
#include "rm_model/logging.h"
#include "rm_model/parallel.h"
#include "rm_model/progress.h"
#include "rm_model/train.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

namespace rm_model {

namespace {

std::string selector_profile() {
  const char* env = std::getenv("RM_MODEL_SELECTOR_PROFILE");
  return env ? std::string(env) : std::string();
}

std::size_t selector_layers() {
  const char* env = std::getenv("RM_MODEL_SELECTOR_LAYERS");
  if (!env) return 2;
  char* end = nullptr;
  long value = std::strtol(env, &end, 10);
  if (!end || *end != '\0' || value < 2) {
    throw std::runtime_error("Invalid RM_MODEL_SELECTOR_LAYERS (must be >= 2)");
  }
  return static_cast<std::size_t>(value);
}

std::size_t selector_jobs(std::size_t config_count) {
  const char* env = std::getenv("RM_MODEL_SELECTOR_JOBS");
  std::size_t jobs = 0;
  if (env) {
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (!end || *end != '\0' || value < 1) {
      throw std::runtime_error("Invalid RM_MODEL_SELECTOR_JOBS (must be >= 1)");
    }
    jobs = static_cast<std::size_t>(value);
  } else {
    jobs = thread_count();
    if (selector_layers() > 2) {
      jobs = 1;
    }
  }
  if (config_count == 0) return 0;
  return std::min<std::size_t>(jobs, config_count);
}

std::size_t selector_max_leaf_models() {
  const char* env = std::getenv("RM_MODEL_SELECTOR_MAX_LEAF_MODELS");
  if (!env) {
    return selector_layers() > 2 ? 1000000 : 10000000;
  }
  char* end = nullptr;
  long value = std::strtol(env, &end, 10);
  if (!end || *end != '\0' || value < 1) {
    throw std::runtime_error("Invalid RM_MODEL_SELECTOR_MAX_LEAF_MODELS (must be >= 1)");
  }
  return static_cast<std::size_t>(value);
}

bool selector_light_layers() {
  const char* env = std::getenv("RM_MODEL_SELECTOR_LIGHT_LAYERS");
  if (!env) return selector_layers() > 2;
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
  if (value == "0" || value == "false" || value == "no" || value == "off") return false;
  throw std::runtime_error("Invalid RM_MODEL_SELECTOR_LIGHT_LAYERS (use 0/1)");
}

std::vector<std::string> top_only_layers() {
  std::string profile = selector_profile();
  if (profile.empty()) {
    return {"radix", "radix18", "radix22", "robust_linear"};
  }

  if (profile == "fast") return {"robust_linear"};
  if (profile == "memory") return {"radix", "radix18", "radix22", "robust_linear"};
  if (profile == "disk") {
    return {"radix", "radix18", "radix22", "robust_linear", "normal", "lognormal", "loglinear"};
  }
  throw std::runtime_error("Invalid selector profile " + profile);
}

std::vector<std::string> anywhere_layers() {
  std::string profile = selector_profile();
  if (profile.empty()) {
    return {"linear", "cubic", "linear_spline"};
  }

  if (profile == "fast") return {"linear", "cubic"};
  if (profile == "memory" || profile == "disk") return {"linear", "cubic", "linear_spline"};
  throw std::runtime_error("Invalid selector profile " + profile);
}

std::vector<std::string> light_layers() {
  std::string profile = selector_profile();
  if (profile.empty() || profile == "memory" || profile == "disk") {
    return {"linear"};
  }
  if (profile == "fast") return {"linear"};
  throw std::runtime_error("Invalid selector profile " + profile);
}

std::vector<std::string> non_top_layers() {
  return selector_light_layers() ? light_layers() : anywhere_layers();
}

std::vector<uint64_t> get_branching_factors() {
  std::string profile = selector_profile();
  int start = selector_layers() > 2 ? 4 : 6;
  int end = 25;
  int step = 1;

  if (profile == "fast") step = 2;
  if (profile == "disk") end = 28;

  std::vector<uint64_t> results;
  for (int i = start; i < end; i += step) {
    results.push_back(1ULL << i);
  }
  return results;
}

std::size_t count_layers(const std::string& model_spec) {
  if (model_spec.empty()) return 0;
  return static_cast<std::size_t>(1 + std::count(model_spec.begin(), model_spec.end(), ','));
}

bool exceeds_leaf_limit(std::size_t layers, uint64_t branch_factor, std::size_t max_leaf_models) {
  if (layers <= 1) return false;
  uint64_t leaf_models = 1;
  for (std::size_t idx = 1; idx < layers; ++idx) {
    if (branch_factor == 0) return true;
    if (leaf_models > max_leaf_models / branch_factor) return true;
    leaf_models *= branch_factor;
  }
  return leaf_models > max_leaf_models;
}

std::vector<std::pair<std::string, uint64_t>> filter_configs(
    const std::vector<std::pair<std::string, uint64_t>>& configs,
    std::size_t num_rows) {
  std::size_t max_leaf = std::min<std::size_t>(selector_max_leaf_models(), num_rows);
  std::vector<std::pair<std::string, uint64_t>> filtered;
  filtered.reserve(configs.size());
  std::size_t skipped = 0;
  for (const auto& cfg : configs) {
    std::size_t layers = count_layers(cfg.first);
    if (exceeds_leaf_limit(layers, cfg.second, max_leaf)) {
      skipped += 1;
      continue;
    }
    filtered.push_back(cfg);
  }
  if (skipped > 0) {
    RM_MODEL_LOG_INFO("Optimizer skipped " << skipped << " configs due to leaf model limit "
                                      << max_leaf);
  }
  return filtered;
}

std::vector<ModelSelectionStats> pareto_front(const std::vector<ModelSelectionStats>& results) {
  std::vector<ModelSelectionStats> front;
  for (const auto& res : results) {
    bool dominated = false;
    for (const auto& other : results) {
      if (res.dominated_by(other)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) front.push_back(res);
  }
  return front;
}

std::vector<ModelSelectionStats> narrow_front(const std::vector<ModelSelectionStats>& results,
                                        std::size_t desired_size) {
  if (desired_size < 2) {
    throw std::runtime_error("Optimizer restrict must be >= 2");
  }
  if (results.size() <= desired_size) return results;

  std::vector<ModelSelectionStats> tmp = results;
  std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.size < b.size; });

  auto best = tmp.front();
  tmp.erase(tmp.begin());

  while (tmp.size() > desired_size - 1) {
    double smallest_gap = std::numeric_limits<double>::infinity();
    std::size_t idx1 = 0;
    std::size_t idx2 = 1;

    for (std::size_t i = 0; i + 1 < tmp.size(); ++i) {
      double gap = static_cast<double>(tmp[i + 1].size) / static_cast<double>(tmp[i].size);
      if (gap < smallest_gap) {
        smallest_gap = gap;
        idx1 = i;
        idx2 = i + 1;
      }
    }

    double err1 = tmp[idx1].average_log2_error;
    double err2 = tmp[idx2].average_log2_error;
    if (err1 > err2) {
      tmp.erase(tmp.begin() + idx1);
    } else {
      tmp.erase(tmp.begin() + idx2);
    }
  }

  tmp.insert(tmp.begin(), best);
  return tmp;
}

std::vector<std::pair<std::string, uint64_t>> first_phase_configs() {
  std::vector<std::pair<std::string, uint64_t>> results;
  std::size_t layers = selector_layers();
  auto top_models = top_only_layers();
  auto any_models = anywhere_layers();
  auto non_top = non_top_layers();

  top_models.insert(top_models.end(), any_models.begin(), any_models.end());

  auto branch_factors = get_branching_factors();
  std::vector<uint64_t> sampled;
  if (layers <= 2) {
    for (std::size_t idx = 0; idx < branch_factors.size(); idx += 5) {
      sampled.push_back(branch_factors[idx]);
    }
  } else {
    std::size_t target = 6;
    std::size_t step = std::max<std::size_t>(1, branch_factors.size() / target);
    for (std::size_t idx = 0; idx < branch_factors.size(); idx += step) {
      sampled.push_back(branch_factors[idx]);
    }
    if (!branch_factors.empty() && sampled.back() != branch_factors.back()) {
      sampled.push_back(branch_factors.back());
    }
  }

  if (layers == 2) {
    for (const auto& top : top_models) {
      for (const auto& bottom : non_top) {
        for (const auto& bf : sampled) {
          results.emplace_back(top + "," + bottom, bf);
        }
      }
    }
    return results;
  }

  std::vector<std::string> middle(layers - 2);
  std::function<void(std::size_t, const std::string&)> emit_configs =
      [&](std::size_t depth, const std::string& top) {
        if (depth == middle.size()) {
          for (const auto& bottom : non_top) {
            std::string model_spec = top;
            for (const auto& mid : middle) {
              model_spec += "," + mid;
            }
            model_spec += "," + bottom;
            for (const auto& bf : sampled) {
              results.emplace_back(model_spec, bf);
            }
          }
          return;
        }
        for (const auto& mid : non_top) {
          middle[depth] = mid;
          emit_configs(depth + 1, top);
        }
      };

  for (const auto& top : top_models) {
    emit_configs(0, top);
  }

  return results;
}

std::vector<std::pair<std::string, uint64_t>> second_phase_configs(const std::vector<ModelSelectionStats>& first_phase) {
  auto front = pareto_front(first_phase);
  std::set<std::string> qualifying;
  for (const auto& result : front) {
    qualifying.insert(result.model_spec);
  }

  RM_MODEL_LOG_INFO("Qualifying model types for phase 2: " << qualifying.size());
  std::vector<std::pair<std::string, uint64_t>> results;
  auto branch_factors = get_branching_factors();

  if (selector_layers() <= 2) {
    for (const auto& model : qualifying) {
      for (auto bf : branch_factors) {
        bool exists = std::any_of(first_phase.begin(), first_phase.end(), [&](const auto& v) {
          return v.has_config(model, bf);
        });
        if (exists) continue;
        results.emplace_back(model, bf);
      }
    }
    return results;
  }

  std::set<std::pair<std::string, uint64_t>> seen;
  for (const auto& item : first_phase) {
    seen.insert({item.model_spec, item.branching_factor});
  }

  for (const auto& model : qualifying) {
    const ModelSelectionStats* best = nullptr;
    for (const auto& item : first_phase) {
      if (item.model_spec != model) continue;
      if (!best || item.average_log2_error < best->average_log2_error) {
        best = &item;
      }
    }
    if (!best || branch_factors.empty()) continue;

    auto it = std::lower_bound(branch_factors.begin(), branch_factors.end(),
                               best->branching_factor);
    std::size_t idx = 0;
    if (it == branch_factors.end()) {
      idx = branch_factors.size() - 1;
    } else {
      idx = static_cast<std::size_t>(std::distance(branch_factors.begin(), it));
      if (it != branch_factors.begin()) {
        auto prev = *(it - 1);
        auto next = *it;
        if (best->branching_factor - prev < next - best->branching_factor) {
          idx -= 1;
        }
      }
    }

    std::size_t start = (idx > 2) ? idx - 2 : 0;
    std::size_t end = std::min<std::size_t>(branch_factors.size(), idx + 3);
    for (std::size_t i = start; i < end; ++i) {
      auto key = std::make_pair(model, branch_factors[i]);
      if (seen.insert(key).second) {
        results.emplace_back(key);
      }
    }
  }

  return results;
}

template <typename T>
std::vector<ModelSelectionStats> measure_models(TrainingData<T>& data,
                                               const std::vector<std::pair<std::string, uint64_t>>& configs,
                                               const std::string& phase_label) {
  auto filtered = filter_configs(configs, data.len());
  ProgressBar pbar(filtered.size());
  pbar.set_message(phase_label);

  std::vector<ModelSelectionStats> results(filtered.size());
  std::size_t jobs = selector_jobs(filtered.size());
  if (jobs <= 1) {
    for (std::size_t idx = 0; idx < filtered.size(); ++idx) {
      const auto& cfg = filtered[idx];
      std::ostringstream msg;
      msg << phase_label << " " << cfg.first << " bf=" << cfg.second;
      pbar.update_message(msg.str());
      auto local_data = data.soft_copy();
      auto trained = train<T>(local_data, cfg.first, cfg.second);
      results[idx] = ModelSelectionStats{trained.model_spec,
                                   trained.branching_factor,
                                   trained.model_avg_log2_error,
                                   trained.model_max_log2_error,
                                   trained.model_point_mae,
                                   trained.model_point_rmse,
                                   model_size_bytes(trained)};
      pbar.inc_with_message(msg.str());
    }
    pbar.finish();
    return results;
  }

  TaskGroup group(jobs);
  for (std::size_t idx = 0; idx < filtered.size(); ++idx) {
    group.schedule([&, idx]() {
      ScopedParallelRegion guard;
      const auto& cfg = filtered[idx];
      {
        std::ostringstream msg;
        msg << phase_label << " " << cfg.first << " bf=" << cfg.second;
        pbar.update_message(msg.str());
      }
      auto local_data = data.soft_copy();
      auto trained = train<T>(local_data, cfg.first, cfg.second);
      results[idx] = ModelSelectionStats{trained.model_spec,
                                   trained.branching_factor,
                                   trained.model_avg_log2_error,
                                   trained.model_max_log2_error,
                                   trained.model_point_mae,
                                   trained.model_point_rmse,
                                   model_size_bytes(trained)};
      std::ostringstream msg;
      msg << phase_label << " " << cfg.first << " bf=" << cfg.second;
      pbar.inc_with_message(msg.str());
    });
  }

  group.wait();
  pbar.finish();
  return results;
}

} // namespace

bool ModelSelectionStats::dominated_by(const ModelSelectionStats& other) const {
  if (size < other.size) return false;
  if (average_log2_error < other.average_log2_error) return false;

  if (size == other.size && average_log2_error <= other.average_log2_error) return false;

  double log2_diff = std::abs(average_log2_error - other.average_log2_error);
  if (size <= other.size && log2_diff < std::numeric_limits<double>::epsilon()) return false;

  return true;
}

bool ModelSelectionStats::has_config(const std::string& model, uint64_t bf) const {
  return model_spec == model && branching_factor == bf;
}

json::Value ModelSelectionStats::to_grid_spec(const std::string& namespace_name) const {
  json::Value::Object obj;
  obj.emplace_back("layers", json::Value(model_spec));
  obj.emplace_back("branching factor", json::Value(branching_factor));
  obj.emplace_back("namespace", json::Value(namespace_name));
  obj.emplace_back("size", json::Value(size));
  obj.emplace_back("average log2 error", json::Value(average_log2_error));
  obj.emplace_back("point mae", json::Value(point_mae));
  obj.emplace_back("point rmse", json::Value(point_rmse));
  obj.emplace_back("binary", json::Value(true));
  return json::Value(std::move(obj));
}

void print_selection_table(const std::vector<ModelSelectionStats>& stats) {
  std::ostringstream oss;
  oss << std::left << std::setw(20) << "Models" << std::right
      << std::setw(12) << "Branch" << std::setw(12) << "AvgLg2"
      << std::setw(12) << "MaxLg2" << std::setw(12) << "MAE"
      << std::setw(12) << "RMSE" << std::setw(12) << "Size (b)" << "\n";

  for (const auto& item : stats) {
    oss << std::left << std::setw(20) << item.model_spec << std::right
        << std::setw(12) << item.branching_factor
        << std::setw(12) << std::fixed << std::setprecision(5) << item.average_log2_error
        << std::setw(12) << std::fixed << std::setprecision(5) << item.max_log2_error
        << std::setw(12) << std::fixed << std::setprecision(5) << item.point_mae
        << std::setw(12) << std::fixed << std::setprecision(5) << item.point_rmse
        << std::setw(12) << item.size << "\n";
  }

  std::cout << oss.str();
}

template <typename T>
std::vector<ModelSelectionStats> select_pareto_configs(TrainingData<T>& data,
                                                         std::size_t restrict) {
  if (selector_light_layers()) {
    RM_MODEL_LOG_INFO("Selector light-layer mode: restricting non-top layers to light models");
  }
  auto initial_configs = first_phase_configs();
  RM_MODEL_LOG_INFO("Selector phase 1 configs: " << initial_configs.size());
  auto first_phase_results = measure_models(data, initial_configs, "selector phase 1");

  auto next_configs = second_phase_configs(first_phase_results);
  RM_MODEL_LOG_INFO("Selector phase 2 configs: " << next_configs.size());
  auto second_phase_results = measure_models(data, next_configs, "selector phase 2");

  auto final_front = pareto_front(second_phase_results);
  final_front = narrow_front(final_front, restrict);
  std::sort(final_front.begin(), final_front.end(), [](const auto& a, const auto& b) {
    return a.average_log2_error < b.average_log2_error;
  });

  return final_front;
}

template std::vector<ModelSelectionStats> select_pareto_configs<uint64_t>(TrainingData<uint64_t>&, std::size_t);
template std::vector<ModelSelectionStats> select_pareto_configs<uint32_t>(TrainingData<uint32_t>&, std::size_t);
template std::vector<ModelSelectionStats> select_pareto_configs<double>(TrainingData<double>&, std::size_t);

} // namespace rm_model
