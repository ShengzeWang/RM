#ifndef RM_MODEL_TRAIN_H
#define RM_MODEL_TRAIN_H

#include "rm_model/models/model.h"
#include "rm_model/training_data.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rm_model {

struct TrainedModel {
  std::size_t num_model_rows = 0;
  std::size_t num_data_rows = 0;
  double model_avg_error = 0.0;
  double model_avg_l2_error = 0.0;
  double model_avg_log2_error = 0.0;
  double model_point_mae = 0.0;
  double model_point_rmse = 0.0;
  uint64_t model_max_error = 0;
  std::size_t model_max_error_idx = 0;
  double model_max_log2_error = 0.0;
  std::vector<uint64_t> last_layer_max_l1s;
  std::vector<std::vector<std::unique_ptr<Model>>> model_layers;
  std::string model_spec;
  uint64_t branching_factor = 0;
  std::optional<std::pair<std::size_t, std::vector<std::pair<uint64_t, std::size_t>>>> cache_fix;
  uint64_t build_time_ns = 0;
};

template <typename T>
TrainedModel train(TrainingData<T>& data, const std::string& model_spec, uint64_t branch_factor);

template <typename T>
TrainedModel train_for_size(TrainingData<T>& data, std::size_t max_size);

TrainedModel train_bounded(TrainingData<uint64_t>& data,
                         const std::string& model_spec,
                         uint64_t branch_factor,
                         std::size_t line_size);

} // namespace rm_model

#endif // RM_MODEL_TRAIN_H
