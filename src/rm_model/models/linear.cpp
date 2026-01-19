#include "rm_model/models/linear.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rm_model {

namespace {

struct SlrAccumulator {
  double mean_x = 0.0;
  double mean_y = 0.0;
  double c = 0.0;
  uint64_t n = 0;
  double m2 = 0.0;

  void add(double x, double y) {
    n += 1;
    double dx = x - mean_x;
    mean_x += dx / static_cast<double>(n);
    mean_y += (y - mean_y) / static_cast<double>(n);
    c += dx * (y - mean_y);

    double dx2 = x - mean_x;
    m2 += dx * dx2;
  }

  std::pair<double, double> finish() const {
    if (n == 0) {
      return {0.0, 0.0};
    }
    if (n == 1) {
      return {mean_y, 0.0};
    }

    double cov = c / static_cast<double>(n - 1);
    double var = m2 / static_cast<double>(n - 1);
    if (var == 0.0) {
      return {mean_y, 0.0};
    }

    double beta = cov / var;
    double alpha = mean_y - beta * mean_x;
    return {alpha, beta};
  }
};

template <typename T>
std::pair<double, double> loglinear_slr(const TrainingData<T>& data) {
  SlrAccumulator acc;
  for (const auto& [x, y] : data.iter()) {
    double ln = std::log(static_cast<double>(y));
    if (std::isfinite(ln)) {
      acc.add(TrainingKeyOps<T>::as_float(x), ln);
    }
  }
  return acc.finish();
}

template <typename T>
std::pair<double, double> linear_params(const TrainingData<T>& data) {
  SlrAccumulator acc;
  for (const auto& [x, y] : data.iter()) {
    acc.add(TrainingKeyOps<T>::as_float(x), static_cast<double>(y));
  }
  return acc.finish();
}

inline double exp1(double inp) {
  double x = inp;
  x = 1.0 + x / 64.0;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}

template <typename T>
std::pair<double, double> robust_params(const TrainingData<T>& data) {
  std::size_t total_items = data.len();
  if (total_items == 0) {
    return {0.0, 0.0};
  }

  std::size_t bnd = std::max<std::size_t>(1, static_cast<std::size_t>(total_items * 0.0001));
  if (bnd * 2 + 1 >= data.len()) {
    throw std::runtime_error("robust_linear requires at least 3 samples after trimming");
  }

  SlrAccumulator acc;
  std::size_t idx = 0;
  for (const auto& [x, y] : data.iter()) {
    if (idx >= bnd && idx < data.len() - bnd) {
      acc.add(TrainingKeyOps<T>::as_float(x), static_cast<double>(y));
    }
    ++idx;
  }

  return acc.finish();
}

} // namespace

LinearModel::LinearModel(const TrainingData<uint64_t>& data) : params_(linear_params(data)) {}
LinearModel::LinearModel(const TrainingData<uint32_t>& data) : params_(linear_params(data)) {}
LinearModel::LinearModel(const TrainingData<double>& data) : params_(linear_params(data)) {}

LogLinearModel::LogLinearModel(const TrainingData<uint64_t>& data) : params_(loglinear_slr(data)) {}
LogLinearModel::LogLinearModel(const TrainingData<uint32_t>& data) : params_(loglinear_slr(data)) {}
LogLinearModel::LogLinearModel(const TrainingData<double>& data) : params_(loglinear_slr(data)) {}

RobustLinearModel::RobustLinearModel(const TrainingData<uint64_t>& data) : params_(robust_params(data)) {}
RobustLinearModel::RobustLinearModel(const TrainingData<uint32_t>& data) : params_(robust_params(data)) {}
RobustLinearModel::RobustLinearModel(const TrainingData<double>& data) : params_(robust_params(data)) {}


double LinearModel::predict_to_float(const ModelInput& inp) const {
  auto [alpha, beta] = params_;
  return std::fma(beta, inp.as_float(), alpha);
}

std::vector<ModelParam> LinearModel::params() const {
  return {ModelParam(params_.first), ModelParam(params_.second)};
}

std::string LinearModel::code() const {
  return R"(
inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}
)";
}

bool LinearModel::set_to_constant_model(uint64_t constant) {
  params_ = {static_cast<double>(constant), 0.0};
  return true;
}

double LogLinearModel::predict_to_float(const ModelInput& inp) const {
  auto [alpha, beta] = params_;
  return exp1(std::fma(beta, inp.as_float(), alpha));
}

std::vector<ModelParam> LogLinearModel::params() const {
  return {ModelParam(params_.first), ModelParam(params_.second)};
}

std::string LogLinearModel::code() const {
  return R"(
inline double loglinear(double alpha, double beta, double inp) {
    return exp1(std::fma(beta, inp, alpha));
}
)";
}

std::set<StdFunctions> LogLinearModel::standard_functions() const {
  return {StdFunctions::EXP1};
}

double RobustLinearModel::predict_to_float(const ModelInput& inp) const {
  auto [alpha, beta] = params_;
  return std::fma(beta, inp.as_float(), alpha);
}

std::vector<ModelParam> RobustLinearModel::params() const {
  return {ModelParam(params_.first), ModelParam(params_.second)};
}

std::string RobustLinearModel::code() const {
  return R"(
inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}
)";
}

bool RobustLinearModel::set_to_constant_model(uint64_t constant) {
  params_ = {static_cast<double>(constant), 0.0};
  return true;
}

} // namespace rm_model
