#ifndef RM_MODEL_LEARNED_MODEL_SELECTOR_H
#define RM_MODEL_LEARNED_MODEL_SELECTOR_H

#include "rm_model/training_data.h"

#include <cstdint>
#include <string>
#include <vector>

#include "rm_model/json.h"

namespace rm_model {

struct ModelSelectionStats {
  std::string model_spec;
  uint64_t branching_factor = 0;
  double average_log2_error = 0.0;
  double max_log2_error = 0.0;
  double point_mae = 0.0;
  double point_rmse = 0.0;
  uint64_t size = 0;

  json::Value to_grid_spec(const std::string& namespace_name) const;
  bool dominated_by(const ModelSelectionStats& other) const;
  bool has_config(const std::string& model, uint64_t bf) const;
};

void print_selection_table(const std::vector<ModelSelectionStats>& stats);

template <typename T>
std::vector<ModelSelectionStats> select_pareto_configs(TrainingData<T>& data, std::size_t restrict);

} // namespace rm_model

#endif // RM_MODEL_LEARNED_MODEL_SELECTOR_H
