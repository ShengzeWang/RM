#include "rm_model/cache_fix.h"

#include "rm_model/logging.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rm_model {

namespace {

struct Spline {
  uint64_t from_x;
  std::size_t from_y;
  uint64_t to_x;
  std::size_t to_y;

  static Spline from(std::pair<uint64_t, std::size_t> pt1,
                     std::pair<uint64_t, std::size_t> pt2) {
    assert(pt1.first <= pt2.first);
    assert(pt1.second <= pt2.second);
    return {pt1.first, pt1.second, pt2.first, pt2.second};
  }

  Spline with_new_dest(std::pair<uint64_t, std::size_t> dest) const {
    assert(dest.first >= from_x);
    assert(dest.second >= from_y);
    return {from_x, from_y, dest.first, dest.second};
  }

  std::pair<uint64_t, std::size_t> end() const { return {to_x, to_y}; }

  std::size_t predict(uint64_t inp) const {
    double v0 = static_cast<double>(from_y);
    double v1 = static_cast<double>(to_y);
    double t = static_cast<double>(inp - from_x) / static_cast<double>(to_x - from_x);
    return static_cast<std::size_t>(std::fma(1.0 - t, v0, t * v1));
  }
};

struct SplineFit {
  std::optional<Spline> spline;
  std::vector<std::pair<uint64_t, std::size_t>> curr_pts;
  std::size_t line_size;

  explicit SplineFit(std::size_t line) : spline(std::nullopt), line_size(line) {}

  std::optional<std::pair<uint64_t, std::size_t>> add_point(std::pair<uint64_t, std::size_t> point) {
    if (!spline.has_value()) {
      spline = Spline::from(point, point);
      return point;
    }

    auto last_spline = *spline;
    auto proposed = last_spline.with_new_dest(point);

    curr_pts.push_back(last_spline.end());
    if (check_spline(proposed)) {
      spline = proposed;
      return std::nullopt;
    }

    auto prev_pt = last_spline.end();
    assert(point.first > prev_pt.first);

    spline = Spline::from(prev_pt, point);
    curr_pts.clear();
    curr_pts.push_back(point);
    return prev_pt;
  }

  std::optional<std::pair<uint64_t, std::size_t>> finish() {
    if (!spline.has_value()) return std::nullopt;
    return spline->end();
  }

  bool check_spline(const Spline& candidate) const {
    return std::all_of(curr_pts.begin(), curr_pts.end(), [&](const auto& pt) {
      std::size_t predicted_line = candidate.predict(pt.first) / line_size;
      std::size_t correct_line = pt.second / line_size;
      return predicted_line == correct_line;
    });
  }
};

} // namespace

std::vector<std::pair<uint64_t, std::size_t>> cache_fix(const TrainingData<uint64_t>& data,
                                                        std::size_t line_size) {
  if (data.len() <= line_size) {
    throw std::runtime_error("Cannot apply cachefix with fewer items than line size");
  }

  RM_MODEL_LOG_INFO("Fitting cachefix spline to " << data.len() << " datapoints");

  SplineFit fit(line_size);
  std::vector<std::pair<uint64_t, std::size_t>> spline;

  uint64_t last_key = 0;
  for (const auto& [key, offset] : data.iter_unique()) {
    uint64_t minus = TrainingKeyOps<uint64_t>::minus_epsilon(key);
    assert(minus >= last_key);

    if (minus != last_key) {
      auto added = fit.add_point({minus, offset});
      if (added.has_value()) spline.push_back(*added);
    }

    auto added = fit.add_point({key, offset});
    if (added.has_value()) spline.push_back(*added);

    last_key = key;
  }

  auto last = fit.finish();
  if (last.has_value()) spline.push_back(*last);

  double pct = (static_cast<double>(spline.size()) / static_cast<double>(data.len())) * 100.0;
  RM_MODEL_LOG_INFO("Bounded spline compressed data to " << std::round(pct) << "% of original ("
                                                   << spline.size() << " points, constructed from "
                                                   << data.len() << " points)."
  );

  return spline;
}

} // namespace rm_model
