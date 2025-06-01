#include <distance.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <random>
#include <vector>

#include "simd/simd_utils.h"

using namespace deepsearch::simd;
using Catch::Approx;

class SIMDUtilsTestFixture {
 public:
  SIMDUtilsTestFixture() { generateTestData(); }

 private:
  void generateTestData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    test_data_.resize(64);  // 64个float，支持AVX512测试
    for (size_t i = 0; i < test_data_.size(); ++i) {
      test_data_[i] = dis(gen);
    }
  }

 protected:
  std::vector<float> test_data_;
};

TEST_CASE("SIMD Capabilities Detection", "[simd][capabilities]") {
  SECTION("Basic capability detection") {
    // 测试基本的SIMD能力检测
    bool has_sse = SIMDCapabilities::hasSSE();
    bool has_avx2 = SIMDCapabilities::hasAVX2();
    bool has_avx512 = SIMDCapabilities::hasAVX512();
    bool has_neon = SIMDCapabilities::hasNEON();

    // 至少应该有一种SIMD支持或者都不支持（在某些平台上）
    INFO("SSE: " << has_sse << ", AVX2: " << has_avx2
                 << ", AVX512: " << has_avx512 << ", NEON: " << has_neon);

    // 检查层次关系：如果支持AVX512，通常也支持AVX2和SSE
    if (has_avx512) {
      REQUIRE(has_avx2);  // AVX512通常意味着也支持AVX2
    }
    if (has_avx2) {
      REQUIRE(has_sse);  // AVX2通常意味着也支持SSE
    }
  }

  SECTION("Optimal SIMD level detection") {
    auto level = SIMDCapabilities::getOptimalSIMD();

    // 检查返回的级别是否有效
    bool valid_level = (level == SIMDCapabilities::Level::AVX512 ||
                        level == SIMDCapabilities::Level::AVX2 ||
                        level == SIMDCapabilities::Level::SSE ||
                        level == SIMDCapabilities::Level::NEON ||
                        level == SIMDCapabilities::Level::NONE);
    REQUIRE(valid_level);
  }
}

TEST_CASE("Memory Prefetch Functions", "[simd][prefetch]") {
  std::vector<float> data(1024);

  SECTION("Template prefetch") {
    // 测试模板预取功能
    REQUIRE_NOTHROW(deepsearch::mem_prefetch<deepsearch::prefetch_L1>(
        reinterpret_cast<char*>(data.data()), 1));
    REQUIRE_NOTHROW(deepsearch::mem_prefetch<deepsearch::prefetch_L1>(
        reinterpret_cast<char*>(data.data()), 4));
    REQUIRE_NOTHROW(deepsearch::mem_prefetch<deepsearch::prefetch_L1>(
        reinterpret_cast<char*>(data.data()), 8));
  }
}

TEST_CASE_METHOD(SIMDUtilsTestFixture, "SIMD Reduce Functions",
                 "[simd][reduce]") {
  SECTION("SSE reduce") {
#ifdef __SSE__
    if (SIMDCapabilities::hasSSE()) {
      __m128 vec = _mm_loadu_ps(test_data_.data());
      float result = reduce_add_f32x4(vec);

      // 计算期望结果
      float expected =
          test_data_[0] + test_data_[1] + test_data_[2] + test_data_[3];
      REQUIRE(result == Approx(expected).margin(1e-5f));
    }
#endif
  }

  SECTION("AVX2 reduce") {
#ifdef __AVX2__
    if (SIMDCapabilities::hasAVX2()) {
      __m256 vec = _mm256_loadu_ps(test_data_.data());
      float result = reduce_add_f32x8(vec);

      // 计算期望结果
      float expected = 0.0f;
      for (int i = 0; i < 8; ++i) {
        expected += test_data_[i];
      }
      REQUIRE(result == Approx(expected).margin(1e-5f));
    }
#endif
  }

  SECTION("AVX512 reduce") {
#ifdef __AVX512F__
    if (SIMDCapabilities::hasAVX512()) {
      __m512 vec = _mm512_loadu_ps(test_data_.data());
      float result = reduce_add_f32x16(vec);

      // 计算期望结果
      float expected = 0.0f;
      for (int i = 0; i < 16; ++i) {
        expected += test_data_[i];
      }
      REQUIRE(result == Approx(expected).margin(1e-5f));
    }
#endif
  }

  SECTION("NEON reduce") {
#ifdef __ARM_NEON
    if (SIMDCapabilities::hasNEON()) {
      float32x4_t vec = vld1q_f32(test_data_.data());
      float result = reduce_add_f32x4(vec);

      // 计算期望结果
      float expected =
          test_data_[0] + test_data_[1] + test_data_[2] + test_data_[3];
      REQUIRE(result == Approx(expected).margin(1e-5f));
    }
#endif
  }
}
