#include "computers.h"

// 平台检测和头文件包含
#ifdef _WIN32
#include <intrin.h>
#ifdef __AVX512F__
#include <immintrin.h>
#elif __AVX2__
#include <immintrin.h>
#elif __SSE__
#include <emmintrin.h>
#include <xmmintrin.h>
#endif
#elif defined(__APPLE__) || defined(__linux__)
#if defined(__x86_64__) || defined(__i386__)  // 修复：使用 #if 而不是 #ifdef
#include <cpuid.h>
#ifdef __AVX512F__
#include <immintrin.h>
#elif __AVX2__
#include <immintrin.h>
#elif __SSE__
#include <emmintrin.h>
#include <nmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#endif
#elif defined(__aarch64__) || defined(__arm__)
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
// ARM 平台不需要 cpuid
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif
#endif

#include <algorithm>
#include <cmath>

// 如果需要使用 simd 命名空间的函数，需要包含相应头文件
// #include "simd/avx2.h"
// #include "simd/avx512.h"

namespace deepsearch {
namespace distance {

// SIMD能力检测实现
bool SIMDCapabilities::hasAVX512() {
#ifdef __AVX512F__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 16)) != 0;  // AVX512F
#elif defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1 << 16)) != 0;  // AVX512F
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasAVX2() {
#ifdef __AVX2__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 5)) != 0;  // AVX2
#elif defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1 << 5)) != 0;  // AVX2
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasSSE() {
#ifdef __SSE__
#if defined(_WIN32)
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);
  return (cpuInfo[3] & (1 << 25)) != 0;  // SSE
#elif defined(__x86_64__) || defined(__i386__)
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  return (edx & (1 << 25)) != 0;  // SSE
#endif
#endif
  return false;
}

bool SIMDCapabilities::hasNEON() {
#ifdef __ARM_NEON
#ifdef __APPLE__
  // 在 Apple Silicon 上，NEON 总是可用的
  return true;
#elif defined(__linux__)
  // 在 Linux ARM 上检查 /proc/cpuinfo
  // 简化实现：如果编译时启用了 NEON，就认为运行时也支持
  return true;
#endif
#endif
  return false;
}

std::string SIMDCapabilities::getOptimalSIMD() {
  if (hasAVX512()) return "AVX512";
  if (hasAVX2()) return "AVX2";
  if (hasSSE()) return "SSE";
  if (hasNEON()) return "NEON";
  return "Generic";
}

// L2距离计算器实现
template <typename T>
float L2DistanceComputer<T>::compute(const T* a, const T* b) const {
  if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
    if (SIMDCapabilities::hasAVX512()) {
      return computeAVX512(a, b);
    }
#endif
#ifdef __AVX2__
    if (SIMDCapabilities::hasAVX2()) {
      return computeAVX2(a, b);
    }
#endif
#ifdef __SSE__
    if (SIMDCapabilities::hasSSE()) {
      return computeSSE(a, b);
    }
#endif
  }
  return computeGeneric(a, b);
}

template <typename T>
void L2DistanceComputer<T>::prefetch(const char* data) const {
  // prefetch_L1(data);
  // prefetch_L2(data + 64 / sizeof(T));
}

template <typename T>
float L2DistanceComputer<T>::computeAVX512(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "AVX512 only supports float");

#ifdef __AVX512F__
  __m512 sum = _mm512_setzero_ps();
  size_t simd_end = (dim_ / 16) * 16;

  for (size_t i = 0; i < simd_end; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    __m512 diff = _mm512_sub_ps(va, vb);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  float result = _mm512_reduce_add_ps(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    float diff = a[i] - b[i];
    result += diff * diff;
  }

  return result;
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float L2DistanceComputer<T>::computeAVX2(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "AVX2 only supports float");

#ifdef __AVX2__
  __m256 sum = _mm256_setzero_ps();
  size_t simd_end = (dim_ / 8) * 8;

  for (size_t i = 0; i < simd_end; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  float result = simd::reduce_add(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    float diff = a[i] - b[i];
    result += diff * diff;
  }

  return result;
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float L2DistanceComputer<T>::computeSSE(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "SSE only supports float");

#ifdef __SSE__
  __m128 sum = _mm_setzero_ps();
  size_t simd_end = (dim_ / 4) * 4;

  for (size_t i = 0; i < simd_end; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 diff = _mm_sub_ps(va, vb);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  // 水平求和
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float result = _mm_cvtss_f32(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    float diff = a[i] - b[i];
    result += diff * diff;
  }

  return result;
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float L2DistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
  float result = 0;
  for (size_t i = 0; i < dim_; ++i) {
    float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    result += diff * diff;
  }
  return result;
}

// 内积距离计算器实现
template <typename T>
float IPDistanceComputer<T>::compute(const T* a, const T* b) const {
  if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
    if (SIMDCapabilities::hasAVX512()) {
      return computeAVX512(a, b);
    }
#endif
#ifdef __AVX2__
    if (SIMDCapabilities::hasAVX2()) {
      return computeAVX2(a, b);
    }
#endif
#ifdef __SSE__
    if (SIMDCapabilities::hasSSE()) {
      return computeSSE(a, b);
    }
#endif
  }
  return computeGeneric(a, b);
}

template <typename T>
void IPDistanceComputer<T>::prefetch(const char* data) const {
  // prefetch_L1(data);
  // prefetch_L2(data + 64 / sizeof(T));
}

template <typename T>
float IPDistanceComputer<T>::computeAVX512(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "AVX512 only supports float");

#ifdef __AVX512F__
  __m512 sum = _mm512_setzero_ps();
  size_t simd_end = (dim_ / 16) * 16;

  for (size_t i = 0; i < simd_end; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    sum = _mm512_fmadd_ps(va, vb, sum);
  }

  float result = _mm512_reduce_add_ps(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    result += a[i] * b[i];
  }

  return 1.0f - result;  // 转换为距离
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float IPDistanceComputer<T>::computeAVX2(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "AVX2 only supports float");

#ifdef __AVX2__
  __m256 sum = _mm256_setzero_ps();
  size_t simd_end = (dim_ / 8) * 8;

  for (size_t i = 0; i < simd_end; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }

  float result = simd::reduce_add(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    result += a[i] * b[i];
  }

  return 1.0f - result;  // 转换为距离
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float IPDistanceComputer<T>::computeSSE(const T* a, const T* b) const {
  static_assert(std::is_same_v<T, float>, "SSE only supports float");

#ifdef __SSE__
  __m128 sum = _mm_setzero_ps();
  size_t simd_end = (dim_ / 4) * 4;

  for (size_t i = 0; i < simd_end; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }

  // 水平求和
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float result = _mm_cvtss_f32(sum);

  // 处理剩余元素
  for (size_t i = simd_end; i < dim_; ++i) {
    result += a[i] * b[i];
  }

  return 1.0f - result;  // 转换为距离
#else
  return computeGeneric(a, b);
#endif
}

template <typename T>
float IPDistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
  float result = 0;
  for (size_t i = 0; i < dim_; ++i) {
    result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return 1.0 - result;  // 转换为距离
}

// 余弦距离计算器实现
template <typename T>
float CosineDistanceComputer<T>::compute(const T* a, const T* b) const {
  return computeGeneric(a, b);
}

template <typename T>
void CosineDistanceComputer<T>::prefetch(const char* data) const {
  // prefetch_L1(data);
  // prefetch_L2(data + 64 / sizeof(T));
}

template <typename T>
float CosineDistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
  float dot_product = 0;
  float norm_a = 0;
  float norm_b = 0;

  for (size_t i = 0; i < dim_; ++i) {
    float val_a = static_cast<float>(a[i]);
    float val_b = static_cast<float>(b[i]);
    dot_product += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  float norm_product = std::sqrt(norm_a * norm_b);
  if (norm_product == 0) return 1.0;  // 避免除零

  return 1.0 - (dot_product / norm_product);
}

// 工厂实现
template <typename T>
std::unique_ptr<core::DistanceComputerTemplate<T>>
DistanceComputerFactory::create(core::DistanceType type, size_t dim) {
  switch (type) {
    case core::DistanceType::L2:
      return std::make_unique<L2DistanceComputer<T>>(dim);
    case core::DistanceType::IP:
      return std::make_unique<IPDistanceComputer<T>>(dim);
    case core::DistanceType::COSINE:
      return std::make_unique<CosineDistanceComputer<T>>(dim);
    default:
      throw std::invalid_argument("Unsupported distance type");
  }
}

std::vector<std::string> DistanceComputerFactory::getSupportedTypes() {
  return {"L2", "IP", "COSINE"};
}

bool DistanceComputerFactory::isTypeSupported(core::DistanceType type) {
  return type == core::DistanceType::L2 || type == core::DistanceType::IP ||
         type == core::DistanceType::COSINE;
}

// 显式实例化
template class L2DistanceComputer<float>;
template class IPDistanceComputer<float>;
template class CosineDistanceComputer<float>;

template std::unique_ptr<core::DistanceComputerTemplate<float>>
DistanceComputerFactory::create<float>(core::DistanceType, size_t);

}  // namespace distance
}  // namespace deepsearch
