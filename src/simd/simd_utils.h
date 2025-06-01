#pragma once

#include <cstddef>
#include <cstdint>

// Platform detection
#if defined(_WIN32)
#include <intrin.h>
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#elif defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#endif

namespace deepsearch {
namespace simd {

/**
 * SIMD Capability Detection
 */
class SIMDCapabilities {
 public:
  static bool hasAVX512();
  static bool hasAVX2();
  static bool hasSSE();
  static bool hasNEON();

  // Get the best available SIMD level
  enum class Level { NONE, SSE, AVX2, AVX512, NEON };

  static Level getOptimalSIMD();
};

/**
 * Memory Prefetch Functions
 */
// void prefetch(const void* ptr);
//
// template <typename T>
// void mem_prefetch(const T* ptr, size_t count) {
//   const char* byte_ptr = reinterpret_cast<const char*>(ptr);
//   switch (count) {
//     case 8:
//       prefetch(byte_ptr + 7 * 64);
//       [[fallthrough]];
//     case 7:
//       prefetch(byte_ptr + 6 * 64);
//       [[fallthrough]];
//     case 6:
//       prefetch(byte_ptr + 5 * 64);
//       [[fallthrough]];
//     case 5:
//       prefetch(byte_ptr + 4 * 64);
//       [[fallthrough]];
//     case 4:
//       prefetch(byte_ptr + 3 * 64);
//       [[fallthrough]];
//     case 3:
//       prefetch(byte_ptr + 2 * 64);
//       [[fallthrough]];
//     case 2:
//       prefetch(byte_ptr + 1 * 64);
//       [[fallthrough]];
//     case 1:
//       prefetch(byte_ptr);
//       break;
//     default:
//       break;
//   }
// }

/**
 * SIMD Reduce Functions
 */
#ifdef __AVX512F__
float reduce_add_f32x16(__m512 x);
#endif

#ifdef __AVX2__
float reduce_add_f32x8(__m256 x);
#endif

#ifdef __SSE__
float reduce_add_f32x4(__m128 x);
#endif

#ifdef __ARM_NEON
float reduce_add_f32x4(float32x4_t x);
#endif

/**
 * Generic reduce function that automatically selects the best implementation
 */
template <typename VectorType>
float reduce_add(VectorType x);

}  // namespace simd
}  // namespace deepsearch
