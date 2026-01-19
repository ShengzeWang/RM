#include "rm_model/models/normal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rm_model {

namespace {

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

inline double phi(double x) {
  return 1.0 / (1.0 + exp1(-1.65451 * x));
}

template <typename T>
std::tuple<double, double, double> ncdf(const TrainingData<T>& data) {
  double scale = -std::numeric_limits<double>::infinity();
  double mean = 0.0;
  double stdev = 0.0;

  double n = static_cast<double>(data.len());
  for (const auto& [inp, y] : data.iter()) {
    double x = TrainingKeyOps<T>::as_float(inp);
    mean += x / n;
    scale = std::max(scale, static_cast<double>(y));
  }

  for (const auto& [inp, _y] : data.iter()) {
    double x = TrainingKeyOps<T>::as_float(inp);
    double diff = x - mean;
    stdev += diff * diff;
  }

  stdev /= n;
  stdev = std::sqrt(stdev);

  return {mean, stdev, scale};
}

template <typename T>
std::tuple<double, double, double> lncdf(const TrainingData<T>& data) {
  double scale = -std::numeric_limits<double>::infinity();
  double mean = 0.0;
  double stdev = 0.0;

  double n = static_cast<double>(data.len());
  for (const auto& [inp, y] : data.iter()) {
    double x = TrainingKeyOps<T>::as_float(inp);
    double lnx = std::isfinite(std::log(x)) ? std::log(x) : 0.0;
    mean += lnx / n;
    scale = std::max(scale, static_cast<double>(y));
  }

  for (const auto& [inp, _y] : data.iter()) {
    double x = TrainingKeyOps<T>::as_float(inp);
    double lnx = std::isfinite(std::log(x)) ? std::log(x) : 0.0;
    double diff = lnx - mean;
    stdev += diff * diff;
  }

  stdev /= n;
  stdev = std::sqrt(stdev);

  return {mean, stdev, scale};
}

} // namespace

NormalModel::NormalModel(const TrainingData<uint64_t>& data) : params_(ncdf(data)) {}
NormalModel::NormalModel(const TrainingData<uint32_t>& data) : params_(ncdf(data)) {}
NormalModel::NormalModel(const TrainingData<double>& data) : params_(ncdf(data)) {}

LogNormalModel::LogNormalModel(const TrainingData<uint64_t>& data) : params_(lncdf(data)) {}
LogNormalModel::LogNormalModel(const TrainingData<uint32_t>& data) : params_(lncdf(data)) {}
LogNormalModel::LogNormalModel(const TrainingData<double>& data) : params_(lncdf(data)) {}


double NormalModel::predict_to_float(const ModelInput& inp) const {
  auto [mean, stdev, scale] = params_;
  return phi((inp.as_float() - mean) / stdev) * scale;
}

std::vector<ModelParam> NormalModel::params() const {
  auto [mean, stdev, scale] = params_;
  return {ModelParam(mean), ModelParam(stdev), ModelParam(scale)};
}

std::string NormalModel::code() const {
  return R"(
inline double ncdf(double mean, double stdev, double scale, double inp) {
    return phi((inp - mean) / stdev) * scale;
}
)";
}

std::set<StdFunctions> NormalModel::standard_functions() const {
  return {StdFunctions::EXP1, StdFunctions::PHI};
}

double LogNormalModel::predict_to_float(const ModelInput& inp) const {
  auto [mean, stdev, scale] = params_;
  double data = inp.as_float();
  return phi((std::max(std::log(data), 0.0) - mean) / stdev) * scale;
}

std::vector<ModelParam> LogNormalModel::params() const {
  auto [mean, stdev, scale] = params_;
  return {ModelParam(mean), ModelParam(stdev), ModelParam(scale)};
}

std::string LogNormalModel::code() const {
  return R"(
inline double lncdf(double mean, double stdev, double scale, double inp) {
    return phi((fmax(0.0, log(inp)) - mean) / stdev) * scale;
}
)";
}

std::set<StdFunctions> LogNormalModel::standard_functions() const {
  return {StdFunctions::EXP1, StdFunctions::PHI};
}

} // namespace rm_model
