#ifndef RM_MODEL_MODELS_STDLIB_H
#define RM_MODEL_MODELS_STDLIB_H

#include <cstdint>
#include <string>

namespace rm_model {

enum class StdFunctions {
  EXP1,
  PHI,
  BinarySearch,
};

inline bool operator<(StdFunctions lhs, StdFunctions rhs) {
  return static_cast<int>(lhs) < static_cast<int>(rhs);
}

inline const char* decl(StdFunctions fn) {
  switch (fn) {
    case StdFunctions::EXP1:
      return "inline double exp1(double x);";
    case StdFunctions::PHI:
      return "inline double phi(double x);";
    case StdFunctions::BinarySearch:
      return "uint64_t bs_upper_bound(const uint64_t a[], uint64_t n, uint64_t x);";
  }
  return "";
}

inline const char* code(StdFunctions fn) {
  switch (fn) {
    case StdFunctions::EXP1:
      return R"(
inline double exp1(double x) {
  x = 1.0 + x / 64.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x;
  return x;
}
)";
    case StdFunctions::PHI:
      return R"(
inline double phi(double x) {
  return 1.0 / (1.0 + exp1(- 1.65451 * x));
}
)";
    case StdFunctions::BinarySearch:
      return R"(
uint64_t bs_upper_bound(const uint64_t a[], uint64_t n, uint64_t x) {
    int l = 0;
    int h = static_cast<int>(n);
    while (l < h) {
        int mid = (l + h) / 2;
        if (x >= a[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return static_cast<uint64_t>(l);
}

)";
  }
  return "";
}

} // namespace rm_model

#endif // RM_MODEL_MODELS_STDLIB_H
