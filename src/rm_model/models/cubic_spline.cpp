#include "rm_model/models/cubic_spline.h"
#include "rm_model/models/linear_spline.h"

#include <cmath>

namespace rm_model {

namespace {

inline double scale(double value, double min, double max) {
  return (value - min) / (max - min);
}

template <typename T>
std::tuple<double, double, double, double> cubic(const TrainingData<T>& data) {
  if (data.len() == 0) {
    return {0.0, 0.0, 1.0, 0.0};
  }

  if (data.len() == 1) {
    return {0.0, 0.0, 0.0, static_cast<double>(data.get(0).second)};
  }

  T candidate = data.get(0).first;
  bool unique = false;
  for (const auto& [x, _y] : data.iter()) {
    if (x != candidate) {
      unique = true;
      break;
    }
  }

  if (!unique) {
    return {0.0, 0.0, 0.0, static_cast<double>(data.get(0).second)};
  }

  auto first_pt = data.get(0);
  auto last_pt = data.get(data.len() - 1);
  double xmin = TrainingKeyOps<T>::as_float(first_pt.first);
  double ymin = static_cast<double>(first_pt.second);
  double xmax = TrainingKeyOps<T>::as_float(last_pt.first);
  double ymax = static_cast<double>(last_pt.second);

  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 1.0;
  double y2 = 1.0;

  double m1 = 0.0;
  for (const auto& [x, y] : data.iter()) {
    double sx = scale(TrainingKeyOps<T>::as_float(x), xmin, xmax);
    if (sx > 0.0) {
      double sy = scale(static_cast<double>(y), ymin, ymax);
      m1 = (sy - y1) / (sx - x1);
      break;
    }
  }

  double m2 = 0.0;
  for (std::size_t idx = data.len(); idx-- > 0;) {
    auto [x, y] = data.get(idx);
    double sx = scale(TrainingKeyOps<T>::as_float(x), xmin, xmax);
    if (sx < 1.0) {
      double sy = scale(static_cast<double>(y), ymin, ymax);
      m2 = (y2 - sy) / (x2 - sx);
      break;
    }
  }

  double m1_sq = m1 * m1;
  double m2_sq = m2 * m2;
  double slope_norm2 = m1_sq + m2_sq;
  if (slope_norm2 > 9.0) {
    double tau = 3.0 / std::sqrt(slope_norm2);
    m1 *= tau;
    m2 *= tau;
  }

  double denom = xmax - xmin;
  double inv_denom3 = 1.0 / (denom * denom * denom);
  double xmax2 = xmax * xmax;
  double xmin2 = xmin * xmin;
  double a = (m1 + m2 - 2.0) * inv_denom3;
  double b = -(xmax * (2.0 * m1 + m2 - 3.0) + xmin * (m1 + 2.0 * m2 - 3.0)) * inv_denom3;
  double c = (m1 * xmax2 + m2 * xmin2 + xmax * xmin * (2.0 * m1 + 2.0 * m2 - 6.0)) * inv_denom3;
  double d = -xmin * (m1 * xmax2 + xmax * xmin * (m2 - 3.0) + xmin2) * inv_denom3;

  a *= (ymax - ymin);
  b *= (ymax - ymin);
  c *= (ymax - ymin);
  d *= (ymax - ymin);
  d += ymin;

  return {a, b, c, d};
}

} // namespace

CubicSplineModel::CubicSplineModel(const TrainingData<uint64_t>& data) : params_(cubic(data)) {
  LinearSplineModel linear(data);
  double our_error = 0.0;
  double lin_error = 0.0;

  for (const auto& [x, y] : data.iter_model_input()) {
    double c_pred = predict_to_float(x);
    double l_pred = linear.predict_to_float(x);
    our_error += std::abs(c_pred - static_cast<double>(y));
    lin_error += std::abs(l_pred - static_cast<double>(y));
  }

  if (lin_error < our_error) {
    auto lp = linear.params();
    params_ = {0.0, 0.0, lp[1].as_float(), lp[0].as_float()};
  }
}

CubicSplineModel::CubicSplineModel(const TrainingData<uint32_t>& data) : params_(cubic(data)) {
  LinearSplineModel linear(data);
  double our_error = 0.0;
  double lin_error = 0.0;

  for (const auto& [x, y] : data.iter_model_input()) {
    double c_pred = predict_to_float(x);
    double l_pred = linear.predict_to_float(x);
    our_error += std::abs(c_pred - static_cast<double>(y));
    lin_error += std::abs(l_pred - static_cast<double>(y));
  }

  if (lin_error < our_error) {
    auto lp = linear.params();
    params_ = {0.0, 0.0, lp[1].as_float(), lp[0].as_float()};
  }
}

CubicSplineModel::CubicSplineModel(const TrainingData<double>& data) : params_(cubic(data)) {
  LinearSplineModel linear(data);
  double our_error = 0.0;
  double lin_error = 0.0;

  for (const auto& [x, y] : data.iter_model_input()) {
    double c_pred = predict_to_float(x);
    double l_pred = linear.predict_to_float(x);
    our_error += std::abs(c_pred - static_cast<double>(y));
    lin_error += std::abs(l_pred - static_cast<double>(y));
  }

  if (lin_error < our_error) {
    auto lp = linear.params();
    params_ = {0.0, 0.0, lp[1].as_float(), lp[0].as_float()};
  }
}

double CubicSplineModel::predict_to_float(const ModelInput& inp) const {
  auto [a, b, c, d] = params_;
  double val = inp.as_float();
  double v1 = std::fma(a, val, b);
  double v2 = std::fma(v1, val, c);
  double v3 = std::fma(v2, val, d);
  return v3;
}

std::vector<ModelParam> CubicSplineModel::params() const {
  auto [a, b, c, d] = params_;
  return {ModelParam(a), ModelParam(b), ModelParam(c), ModelParam(d)};
}

std::string CubicSplineModel::code() const {
  return R"(
inline double cubic(double a, double b, double c, double d, double x) {
    auto v1 = std::fma(a, x, b);
    auto v2 = std::fma(v1, x, c);
    auto v3 = std::fma(v2, x, d);
    return v3;
}
)";
}

bool CubicSplineModel::set_to_constant_model(uint64_t constant) {
  params_ = {0.0, 0.0, 0.0, static_cast<double>(constant)};
  return true;
}

} // namespace rm_model
