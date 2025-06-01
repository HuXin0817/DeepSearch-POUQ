#include "simd_utils.h"

namespace deepsearch {
namespace simd {

// SIMD Capability Detection Implementation
bool SIMDCapabilities::hasAVX512() {
#ifdef __AVX512F__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  return (cpuInfo[1] & (1 << 16)) != 0;  // AVX512F bit
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 16)) != 0;  // AVX512F bit
  }
#endif
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasAVX2() {
#ifdef __AVX2__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  return (cpuInfo[1] & (1 << 5)) != 0;  // AVX2 bit
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 5)) != 0;  // AVX2 bit
  }
#endif
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasSSE() {
#ifdef __SSE__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);
  return (cpuInfo[3] & (1 << 25)) != 0;  // SSE bit
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return (edx & (1 << 25)) != 0;  // SSE bit
  }
#endif
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasNEON() {
#ifdef __ARM_NEON
  return true;  // If compiled with NEON support, assume it's available
#else
  return false;
#endif
}

SIMDCapabilities::Level SIMDCapabilities::getOptimalSIMD() {
  if (hasAVX512()) return Level::AVX512;
  if (hasAVX2()) return Level::AVX2;
  if (hasSSE()) return Level::SSE;
  if (hasNEON()) return Level::NEON;
  return Level::NONE;
}

// Memory Prefetch Implementation
// void prefetch(const void* ptr) {
// #if defined(__x86_64__) || defined(__i386__)
//   _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
// #elif defined(__aarch64__) || defined(__arm__)
//   __builtin_prefetch(ptr, 0, 3);
// #else
//   (void)ptr;  // Suppress unused parameter warning
// #endif
// }

// SIMD Reduce Functions Implementation
#ifdef __AVX512F__
float reduce_add_f32x16(__m512 x) {
  __m256 low = _mm512_castps512_ps256(x);
  __m256 high = _mm512_extractf32x8_ps(x, 1);
  __m256 sum256 = _mm256_add_ps(low, high);
  return reduce_add_f32x8(sum256);
}
#endif

#ifdef __AVX2__
float reduce_add_f32x8(__m256 x) {
  __m128 low = _mm256_castps256_ps128(x);
  __m128 high = _mm256_extractf128_ps(x, 1);
  __m128 sum128 = _mm_add_ps(low, high);
  return reduce_add_f32x4(sum128);
}
#endif

#ifdef __SSE__
float reduce_add_f32x4(__m128 x) {
  __m128 shuf = _mm_movehdup_ps(x);
  __m128 sums = _mm_add_ps(x, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
#endif

#ifdef __ARM_NEON
float reduce_add_f32x4(float32x4_t x) {
  float32x2_t sum = vadd_f32(vget_low_f32(x), vget_high_f32(x));
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}
#endif

// Generic reduce function specializations
#ifdef __AVX512F__
template <>
float reduce_add<__m512>(__m512 x) {
  return reduce_add_f32x16(x);
}
#endif

#ifdef __AVX2__
template <>
float reduce_add<__m256>(__m256 x) {
  return reduce_add_f32x8(x);
}
#endif

#ifdef __SSE__
template <>
float reduce_add<__m128>(__m128 x) {
  return reduce_add_f32x4(x);
}
#endif

#ifdef __ARM_NEON
template <>
float reduce_add<float32x4_t>(float32x4_t x) {
  return reduce_add_f32x4(x);
}
#endif

}  // namespace simd
}  // namespace deepsearch
