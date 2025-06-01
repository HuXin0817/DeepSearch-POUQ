#pragma once

#include <string>
#include <unordered_map>

namespace deepsearch {

enum class Metric {
  L2,
  IP,
};

inline std::unordered_map<std::string, Metric> metric_map = {
    {"L2", Metric::L2}, {"IP", Metric::IP}};

inline constexpr size_t upper_div(size_t x, size_t y) {
  return (x + y - 1) / y;
}

#if defined(__clang__)

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define FAST_BEGIN                     \
  _Pragma("GCC push_options") _Pragma( \
      "GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAST_END _Pragma("GCC pop_options")
#define GLASS_INLINE [[gnu::always_inline]]

#endif

}  // namespace deepsearch
