#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <map>
#include <random>
#include <vector>

#include "simd/distance_functions.h"

using namespace deepsearch::simd;

class BenchmarkFixture {
 public:
  BenchmarkFixture() {
    initializeSIMDFunctions();
    generateBenchmarkData();
  }

 private:
  void generateBenchmarkData() {
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子以确保可重现性
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // 生成不同大小的测试数据
    for (size_t dim : {64, 128, 256, 512, 1024}) {
      std::vector<float> a(dim), b(dim);
      for (size_t i = 0; i < dim; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
      }
      test_data_[dim] = {std::move(a), std::move(b)};
    }
  }

 protected:
  std::map<size_t, std::pair<std::vector<float>, std::vector<float>>>
      test_data_;
};

TEST_CASE_METHOD(BenchmarkFixture, "L2 Distance Benchmark",
                 "[simd][benchmark][l2]") {
  SECTION("L2Sqr vs L2SqrRef performance") {
    for (auto& [dim, data] : test_data_) {
      const auto& [a, b] = data;

      BENCHMARK("L2Sqr SIMD dim=" + std::to_string(dim)) {
        return L2Sqr(a.data(), b.data(), dim);
      };

      BENCHMARK("L2SqrRef dim=" + std::to_string(dim)) {
        return L2SqrRef(a.data(), b.data(), dim);
      };
    }
  }
}

TEST_CASE_METHOD(BenchmarkFixture, "IP Distance Benchmark",
                 "[simd][benchmark][ip]") {
  SECTION("IP vs IPRef performance") {
    for (auto& [dim, data] : test_data_) {
      const auto& [a, b] = data;

      BENCHMARK("IP SIMD dim=" + std::to_string(dim)) {
        return IP(a.data(), b.data(), dim);
      };

      BENCHMARK("IPRef dim=" + std::to_string(dim)) {
        return IPRef(a.data(), b.data(), dim);
      };
    }
  }
}

TEST_CASE_METHOD(BenchmarkFixture, "Cosine Distance Benchmark",
                 "[simd][benchmark][cosine]") {
  SECTION("CosineDistance vs CosineDistanceRef performance") {
    for (auto& [dim, data] : test_data_) {
      const auto& [a, b] = data;

      BENCHMARK("CosineDistance SIMD dim=" + std::to_string(dim)) {
        return CosineDistance(a.data(), b.data(), dim);
      };

      BENCHMARK("CosineDistanceRef dim=" + std::to_string(dim)) {
        return CosineDistanceRef(a.data(), b.data(), dim);
      };
    }
  }
}
