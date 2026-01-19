#include "rm_model/train.h"

#include "rm_model/cache_fix.h"
#include "rm_model/logging.h"
#include "rm_model/models/balanced_radix.h"
#include "rm_model/models/cubic_spline.h"
#include "rm_model/models/histogram.h"
#include "rm_model/models/linear.h"
#include "rm_model/models/linear_spline.h"
#include "rm_model/models/normal.h"
#include "rm_model/models/radix.h"
#include "rm_model/learned_model_selector.h"

#include <algorithm>
#include <chrono>
#include <sstream>

namespace rm_model {

namespace {

std::string trim_copy(const std::string& value) {
  const auto start = value.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  const auto end = value.find_last_not_of(" \t\n\r");
  return value.substr(start, end - start + 1);
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

} // namespace

template <typename T>
TrainedModel train_two_layer(TrainingData<T>& data,
                           const std::string& layer1_model,
                           const std::string& layer2_model,
                           uint64_t num_leaf_models);

template <typename T>
TrainedModel train_multi_layer(TrainingData<T>& data,
                             const std::vector<std::string>& model_list,
                             const std::string& last_model,
                             uint64_t branch_factor);

template <typename T>
TrainedModel train(TrainingData<T>& data, const std::string& model_spec, uint64_t branch_factor) {
  auto start = std::chrono::steady_clock::now();
  if (branch_factor <= 1) {
    throw std::runtime_error("Branching factor must be >= 2");
  }

  std::vector<std::string> models;
  std::stringstream ss(model_spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed = trim_copy(item);
    if (trimmed.empty()) {
      throw std::runtime_error("Model spec contains an empty layer");
    }
    models.push_back(trimmed);
  }
  if (models.empty()) {
    throw std::runtime_error("Model spec must not be empty");
  }

  validate_models<T>(models);
  std::string last_model = models.back();
  models.pop_back();

  if (models.size() == 1) {
    auto result = train_two_layer<T>(data, models[0], last_model, branch_factor);
    auto end = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
    return result;
  }
  if (models.size() > 1) {
    auto result = train_multi_layer<T>(data, models, last_model, branch_factor);
    auto end = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
    return result;
  }

  throw std::runtime_error("Invalid model specification");
}

template <typename T>
TrainedModel train_for_size(TrainingData<T>& data, std::size_t max_size) {
  auto start = std::chrono::steady_clock::now();
  auto pareto = select_pareto_configs(data, 1000);

  auto it = std::find_if(pareto.begin(), pareto.end(), [&](const ModelSelectionStats& stats) {
    return stats.size < max_size;
  });
  if (it == pareto.end()) {
    throw std::runtime_error("Could not find any configurations smaller than " +
                             std::to_string(max_size));
  }

  RM_MODEL_LOG_INFO("Found model config " << it->model_spec << " " << it->branching_factor
                                          << " with size " << it->size << " and average log2 "
                                          << it->average_log2_error);

  auto result = train<T>(data, it->model_spec, it->branching_factor);
  auto end = std::chrono::steady_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
  return result;
}

TrainedModel train_bounded(TrainingData<uint64_t>& data,
                         const std::string& model_spec,
                         uint64_t branch_factor,
                         std::size_t line_size) {
  auto start = std::chrono::steady_clock::now();
  if (line_size == 0) {
    throw std::runtime_error("Line size must be >= 1");
  }

  auto spline = cache_fix(data, line_size);

  std::vector<std::pair<uint64_t, std::size_t>> reindexed;
  reindexed.reserve(spline.size());
  for (std::size_t idx = 0; idx < spline.size(); ++idx) {
    reindexed.emplace_back(spline[idx].first, idx);
  }

  TrainingData<uint64_t> new_data(
      std::make_shared<TrainingData<uint64_t>::VectorProvider>(reindexed));

  auto result = train<uint64_t>(new_data, model_spec, branch_factor);
  result.cache_fix = std::make_pair(line_size, spline);
  result.num_data_rows = data.len();

  auto end = std::chrono::steady_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
  return result;
}

template TrainedModel train<uint64_t>(TrainingData<uint64_t>&, const std::string&, uint64_t);
template TrainedModel train<uint32_t>(TrainingData<uint32_t>&, const std::string&, uint64_t);
template TrainedModel train<double>(TrainingData<double>&, const std::string&, uint64_t);

template TrainedModel train_for_size<uint64_t>(TrainingData<uint64_t>&, std::size_t);
template TrainedModel train_for_size<uint32_t>(TrainingData<uint32_t>&, std::size_t);
template TrainedModel train_for_size<double>(TrainingData<double>&, std::size_t);

} // namespace rm_model
