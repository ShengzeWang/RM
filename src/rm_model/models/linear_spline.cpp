#include "rm_model/models/linear_spline.h"

#include <cmath>

namespace rm_model {

namespace {

template <typename T>
std::pair<double, double> linear_splines(const TrainingData<T>& data) {
  if (data.len() == 0) return {0.0, 0.0};
  if (data.len() == 1) return {static_cast<double>(data.get(0).second), 0.0};

  auto first_pt = data.get(0);
  auto last_pt = data.get(data.len() - 1);

  if (first_pt.first == last_pt.first) {
    return {static_cast<double>(data.get(0).second), 0.0};
  }

  double slope = (static_cast<double>(first_pt.second) - static_cast<double>(last_pt.second)) /
                 (TrainingKeyOps<T>::as_float(first_pt.first) -
                  TrainingKeyOps<T>::as_float(last_pt.first));
  double intercept = static_cast<double>(first_pt.second) -
                     slope * TrainingKeyOps<T>::as_float(first_pt.first);
  return {intercept, slope};
}

} // namespace

LinearSplineModel::LinearSplineModel(const TrainingData<uint64_t>& data)
    : params_(linear_splines(data)) {}
LinearSplineModel::LinearSplineModel(const TrainingData<uint32_t>& data)
    : params_(linear_splines(data)) {}
LinearSplineModel::LinearSplineModel(const TrainingData<double>& data)
    : params_(linear_splines(data)) {}

double LinearSplineModel::predict_to_float(const ModelInput& inp) const {
  auto [alpha, beta] = params_;
  return std::fma(beta, inp.as_float(), alpha);
}

std::vector<ModelParam> LinearSplineModel::params() const {
  return {ModelParam(params_.first), ModelParam(params_.second)};
}

std::string LinearSplineModel::code() const {
  return R"(
inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}
)";
}

bool LinearSplineModel::set_to_constant_model(uint64_t constant) {
  params_ = {static_cast<double>(constant), 0.0};
  return true;
}

} // namespace rm_model
