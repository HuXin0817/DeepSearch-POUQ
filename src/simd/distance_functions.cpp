#include "distance_functions.h"

#include <cmath>

namespace deepsearch {
namespace simd {

// Global function pointers definition
L2SqrFunc L2Sqr = nullptr;
IPFunc IP = nullptr;
CosineFunc CosineDistance = nullptr;
L2SqrSQ8Func L2SqrSQ8_ext = nullptr;
L2SqrSQ4Func L2SqrSQ4 = nullptr;
IPSQ8Func IPSQ8_ext = nullptr;

// Initialize SIMD function pointers
void initializeSIMDFunctions() {
  auto level = SIMDCapabilities::getOptimalSIMD();

  switch (level) {
    case SIMDCapabilities::Level::AVX512:
      L2Sqr = detail::L2Sqr_avx512;
      IP = detail::IP_avx512;
      L2SqrSQ8_ext = detail::L2SqrSQ8_avx512;
      IPSQ8_ext = detail::IPSQ8_avx512;
      L2SqrSQ4 = detail::L2SqrSQ4_avx2;  // Fallback to AVX2 for SQ4
      break;

    case SIMDCapabilities::Level::AVX2:
      L2Sqr = detail::L2Sqr_avx2;
      IP = detail::IP_avx2;
      L2SqrSQ4 = detail::L2SqrSQ4_avx2;
      L2SqrSQ8_ext = L2SqrSQ8_ref;  // Fallback to reference
      IPSQ8_ext = IPSQ8_ref;
      break;

    case SIMDCapabilities::Level::SSE:
      L2Sqr = detail::L2Sqr_sse;
      IP = detail::IP_sse;
      L2SqrSQ8_ext = L2SqrSQ8_ref;
      L2SqrSQ4 = L2SqrSQ4_ref;
      IPSQ8_ext = IPSQ8_ref;
      break;

    case SIMDCapabilities::Level::NEON:
      L2Sqr = detail::L2Sqr_neon;
      IP = detail::IP_neon;
      L2SqrSQ8_ext = L2SqrSQ8_ref;
      L2SqrSQ4 = L2SqrSQ4_ref;
      IPSQ8_ext = IPSQ8_ref;
      break;

    default:
      L2Sqr = L2SqrRef;
      IP = IPRef;
      L2SqrSQ8_ext = L2SqrSQ8_ref;
      L2SqrSQ4 = L2SqrSQ4_ref;
      IPSQ8_ext = IPSQ8_ref;
      break;
  }

  // Cosine distance always uses optimized L2 and IP
  CosineDistance = CosineDistanceRef;
}

// Reference implementations
float L2SqrRef(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    res += t * t;
    pVect1++;
    pVect2++;
  }
  return res;
}

float IPRef(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += (*pVect1) * (*pVect2);
    pVect1++;
    pVect2++;
  }
  return res;
}

float CosineDistanceRef(const float* pVect1, const float* pVect2, size_t qty) {
  // float dot_product = IP(pVect1, pVect2, qty);
  // 确保向量已经归一化
  // float norm1 = std::sqrt(L2Sqr(pVect1, pVect1, qty));
  // float norm2 = std::sqrt(L2Sqr(pVect2, pVect2, qty));

  // if (norm1 == 0.0f || norm2 == 0.0f) {
  //   return 1.0f;  // Maximum distance for zero vectors
  // }

  float cosine_similarity = IP(pVect1, pVect2, qty);
  return 1.0f - cosine_similarity;
}

float L2SqrSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float diff = static_cast<float>(pVect1[i]) - static_cast<float>(pVect2[i]);
    res += diff * diff;
  }
  return res;
}

float L2SqrSQ4_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  size_t qty_bytes = (qty + 1) / 2;

  for (size_t i = 0; i < qty_bytes; i++) {
    uint8_t byte1 = pVect1[i];
    uint8_t byte2 = pVect2[i];

    float val1_low = static_cast<float>(byte1 & 0x0F);
    float val1_high = static_cast<float>((byte1 >> 4) & 0x0F);
    float val2_low = static_cast<float>(byte2 & 0x0F);
    float val2_high = static_cast<float>((byte2 >> 4) & 0x0F);

    float diff_low = val1_low - val2_low;
    float diff_high = val1_high - val2_high;

    res += diff_low * diff_low + diff_high * diff_high;
  }

  return res;
}

float IPSQ8_ref(const void* pVect1v, const void* pVect2v, size_t qty) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    res += static_cast<float>(pVect1[i]) * static_cast<float>(pVect2[i]);
  }
  return res;
}

// SIMD implementations
namespace detail {

#ifdef __SSE__
float L2Sqr_sse(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  const float* pEnd1 = pVect1 + (qty4 << 2);
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    __m128 diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    float t = pVect1[i - (qty4 << 2)] - pVect2[i - (qty4 << 2)];
    res += t * t;
  }

  return res;
}

float IP_sse(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  const float* pEnd1 = pVect1 + (qty4 << 2);
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m128 v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    __m128 v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    res += pVect1[i - (qty4 << 2)] * pVect2[i - (qty4 << 2)];
  }

  return res;
}
#else
float L2Sqr_sse(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_sse(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}
#endif

#ifdef __AVX2__
float L2Sqr_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty8 = qty >> 3;

  const float* pEnd1 = pVect1 + (qty8 << 3);
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    __m256 diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  for (size_t i = qty8 << 3; i < qty; i++) {
    float t = pVect1[i - (qty8 << 3)] - pVect2[i - (qty8 << 3)];
    res += t * t;
  }

  return res;
}

float IP_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty8 = qty >> 3;

  const float* pEnd1 = pVect1 + (qty8 << 3);
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum = _mm256_fmadd_ps(v1, v2, sum);
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  for (size_t i = qty8 << 3; i < qty; i++) {
    res += pVect1[i - (qty8 << 3)] * pVect2[i - (qty8 << 3)];
  }

  return res;
}

float L2SqrSQ4_avx2(const void* pVect1v, const void* pVect2v,
                    const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty_bytes = (qty + 1) / 2;
  size_t qty16 = qty_bytes >> 4;

  __m256 sum = _mm256_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    // Load 16 bytes (32 4-bit values)
    __m128i v1_packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    // Unpack 4-bit values to 8-bit
    __m128i mask = _mm_set1_epi8(0x0F);
    __m128i v1_low = _mm_and_si128(v1_packed, mask);
    __m128i v1_high = _mm_and_si128(_mm_srli_epi16(v1_packed, 4), mask);
    __m128i v2_low = _mm_and_si128(v2_packed, mask);
    __m128i v2_high = _mm_and_si128(_mm_srli_epi16(v2_packed, 4), mask);

    // Convert to float and compute differences
    __m256i v1_16_low = _mm256_cvtepu8_epi16(v1_low);
    __m256i v2_16_low = _mm256_cvtepu8_epi16(v2_low);

    __m256 v1_f_low = _mm256_cvtepi32_ps(
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_16_low)));
    __m256 v2_f_low = _mm256_cvtepi32_ps(
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_16_low)));

    __m256 diff_low = _mm256_sub_ps(v1_f_low, v2_f_low);
    sum = _mm256_fmadd_ps(diff_low, diff_low, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x8(sum);

  // Handle remaining elements
  size_t remaining_bytes = qty_bytes - (qty16 << 4);
  for (size_t i = 0; i < remaining_bytes; i++) {
    uint8_t byte1 = pVect1[i];
    uint8_t byte2 = pVect2[i];

    float val1_low = static_cast<float>(byte1 & 0x0F);
    float val1_high = static_cast<float>((byte1 >> 4) & 0x0F);
    float val2_low = static_cast<float>(byte2 & 0x0F);
    float val2_high = static_cast<float>((byte2 >> 4) & 0x0F);

    float diff_low = val1_low - val2_low;
    float diff_high = val1_high - val2_high;

    res += diff_low * diff_low + diff_high * diff_high;
  }

  return res;
}
#else
float L2Sqr_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_avx2(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}

float L2SqrSQ4_avx2(const void* pVect1v, const void* pVect2v, size_t qty) {
  return L2SqrSQ4_ref(pVect1v, pVect2v, qty);
}
#endif

#ifdef __AVX512F__
float L2Sqr_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    __m512 diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    float t = pVect1[i - (qty16 << 4)] - pVect2[i - (qty16 << 4)];
    res += t * t;
  }

  return res;
}

float IP_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    sum = _mm512_fmadd_ps(v1, v2, sum);
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    res += pVect1[i - (qty16 << 4)] * pVect2[i - (qty16 << 4)];
  }

  return res;
}

float L2SqrSQ8_avx512(const void* pVect1v, const void* pVect2v,
                      const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty16 = qty >> 4;

  __m512 sum = _mm512_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    __m128i v1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    __m512i v1_32 = _mm512_cvtepu8_epi32(v1_8);
    __m512i v2_32 = _mm512_cvtepu8_epi32(v2_8);

    __m512 v1_f = _mm512_cvtepi32_ps(v1_32);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_32);

    __m512 diff = _mm512_sub_ps(v1_f, v2_f);
    sum = _mm512_fmadd_ps(diff, diff, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    float diff = static_cast<float>(pVect1[i - (qty16 << 4)]) -
                 static_cast<float>(pVect2[i - (qty16 << 4)]);
    res += diff * diff;
  }

  return res;
}

float IPSQ8_avx512(const void* pVect1v, const void* pVect2v,
                   const void* qty_ptr) {
  const uint8_t* pVect1 = static_cast<const uint8_t*>(pVect1v);
  const uint8_t* pVect2 = static_cast<const uint8_t*>(pVect2v);
  size_t qty = *static_cast<const size_t*>(qty_ptr);

  float res = 0;
  size_t qty16 = qty >> 4;

  __m512 sum = _mm512_set1_ps(0);

  for (size_t i = 0; i < qty16; i++) {
    __m128i v1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect1));
    __m128i v2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pVect2));

    __m512i v1_32 = _mm512_cvtepu8_epi32(v1_8);
    __m512i v2_32 = _mm512_cvtepu8_epi32(v2_8);

    __m512 v1_f = _mm512_cvtepi32_ps(v1_32);
    __m512 v2_f = _mm512_cvtepi32_ps(v2_32);

    sum = _mm512_fmadd_ps(v1_f, v2_f, sum);

    pVect1 += 16;
    pVect2 += 16;
  }

  res += reduce_add_f32x16(sum);

  // Handle remaining elements
  for (size_t i = qty16 << 4; i < qty; i++) {
    res += static_cast<float>(pVect1[i - (qty16 << 4)]) *
           static_cast<float>(pVect2[i - (qty16 << 4)]);
  }

  return res;
}
#else
float L2Sqr_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_avx512(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}

float L2SqrSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty) {
  return L2SqrSQ8_ref(pVect1v, pVect2v, qty);
}

float IPSQ8_avx512(const void* pVect1v, const void* pVect2v, size_t qty) {
  return IPSQ8_ref(pVect1v, pVect2v, qty);
}
#endif

#ifdef __ARM_NEON
float L2Sqr_neon(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  float32x4_t sum = vdupq_n_f32(0);

  for (size_t i = 0; i < qty4; i++) {
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    float32x4_t diff = vsubq_f32(v1, v2);
    sum = vfmaq_f32(sum, diff, diff);
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    float t = pVect1[i - (qty4 << 2)] - pVect2[i - (qty4 << 2)];
    res += t * t;
  }

  return res;
}

float IP_neon(const float* pVect1, const float* pVect2, size_t qty) {
  float res = 0;
  size_t qty4 = qty >> 2;

  float32x4_t sum = vdupq_n_f32(0);

  for (size_t i = 0; i < qty4; i++) {
    float32x4_t v1 = vld1q_f32(pVect1);
    pVect1 += 4;
    float32x4_t v2 = vld1q_f32(pVect2);
    pVect2 += 4;
    sum = vfmaq_f32(sum, v1, v2);
  }

  res += reduce_add_f32x4(sum);

  // Handle remaining elements
  for (size_t i = qty4 << 2; i < qty; i++) {
    res += pVect1[i - (qty4 << 2)] * pVect2[i - (qty4 << 2)];
  }

  return res;
}
#else
float L2Sqr_neon(const float* pVect1, const float* pVect2, size_t qty) {
  return L2SqrRef(pVect1, pVect2, qty);
}

float IP_neon(const float* pVect1, const float* pVect2, size_t qty) {
  return IPRef(pVect1, pVect2, qty);
}
#endif

}  // namespace detail

}  // namespace simd
}  // namespace deepsearch

// 在文件末尾添加静态初始化器
namespace {
struct SIMDInitializer {
  SIMDInitializer() { deepsearch::simd::initializeSIMDFunctions(); }
};

// 在程序启动时自动初始化
[[maybe_unused]] SIMDInitializer simd_init;
}  // namespace
