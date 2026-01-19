#include "rm_model/models/histogram.h"

#include <algorithm>
#include <stdexcept>

namespace rm_model {

namespace {

template <typename T>
std::vector<uint64_t> equidepth_histogram(const TrainingData<T>& data) {
  if (data.len() == 0) return {};

  std::vector<uint64_t> splits;
  std::size_t num_bins = data.get(data.len() - 1).second;
  if (num_bins == 0) return {};
  std::size_t items_per_bin = data.len() / num_bins;

  if (items_per_bin < 1) {
    throw std::runtime_error("not enough items for equidepth histogram");
  }

  RM_MODEL_LOG_INFO("Equidepth histogram using " << num_bins << " bins");

  for (std::size_t bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
    std::size_t start_idx = bin_idx * items_per_bin;
    uint64_t start_val = TrainingKeyOps<T>::as_uint(data.get_key(start_idx));
    splits.push_back(start_val);
  }

  return splits;
}

} // namespace

EquidepthHistogramModel::EquidepthHistogramModel(const TrainingData<uint64_t>& data) {
  if (data.len() == 0) return;
  params_ = equidepth_histogram(data);
  radix_ = radix_index(params_, 20);
}

EquidepthHistogramModel::EquidepthHistogramModel(const TrainingData<uint32_t>& data) {
  if (data.len() == 0) return;
  params_ = equidepth_histogram(data);
  radix_ = radix_index(params_, 20);
}

EquidepthHistogramModel::EquidepthHistogramModel(const TrainingData<double>& data) {
  if (data.len() == 0) return;
  params_ = equidepth_histogram(data);
  radix_ = radix_index(params_, 20);
}

uint64_t EquidepthHistogramModel::predict_to_int(const ModelInput& inp) const {
  uint64_t val = inp.as_int();
  auto it = std::upper_bound(params_.begin(), params_.end(), val);
  std::size_t idx = static_cast<std::size_t>(std::distance(params_.begin(), it));
  if (idx == 0) return 0;
  return static_cast<uint64_t>(idx - 1);
}

std::vector<ModelParam> EquidepthHistogramModel::params() const {
  return {ModelParam(static_cast<uint64_t>(params_.size())),
          ModelParam(radix_),
          ModelParam(params_)};
}

std::string EquidepthHistogramModel::code() const {
  return R"(
inline uint64_t ed_histogram(const uint64_t length,
                             const uint64_t radix[],
                             const uint64_t pivots[],
                             uint64_t key) {
    uint64_t key_radix = key >> (64 - 20);
    unsigned int radix_lb = radix[key_radix];
    unsigned int radix_ub = radix[key_radix + 1];
    uint64_t li = bs_upper_bound(pivots + radix_lb, radix_ub - radix_lb, key) + radix_lb - 1;
    return li;
}
)";
}

std::set<StdFunctions> EquidepthHistogramModel::standard_functions() const {
  return {StdFunctions::BinarySearch};
}

} // namespace rm_model
