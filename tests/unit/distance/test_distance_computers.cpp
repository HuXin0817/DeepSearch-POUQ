#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "distance/computers.h"
#include "simd/simd_utils.h"

using namespace deepsearch::core;
using namespace deepsearch::distance;
using namespace deepsearch::simd;

using Catch::Approx;

class DistanceComputerTestFixture {
 public:
  DistanceComputerTestFixture() : dim_(128) { generateTestData(); }

 private:
  void generateTestData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data_a_.resize(dim_);
    data_b_.resize(dim_);

    for (size_t i = 0; i < dim_; ++i) {
      data_a_[i] = dis(gen);
      data_b_[i] = dis(gen);
    }

    normalizeVector(data_a_);
    normalizeVector(data_b_);
  }

  void normalizeVector(std::vector<float>& vec) {
    // 计算L2范数
    float norm = 0.0f;
    for (float val : vec) {
      norm += val * val;
    }
    norm = std::sqrt(norm);

    // 避免除零错误
    if (norm < 1e-8f) {
      // 如果范数太小，生成新的随机向量
      std::uniform_real_distribution<float> dis(0.1f, 1.0f);
      for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen_);
      }
      normalizeVector(vec);  // 递归归一化
      return;
    }

    // 执行归一化
    for (float& val : vec) {
      val /= norm;
    }
  }

 protected:
  size_t dim_;
  std::vector<float> data_a_;
  std::vector<float> data_b_;
  std::mt19937 gen_{std::random_device{}()};
};

// 辅助函数：将SIMD Level枚举转换为字符串
std::string simdLevelToString(SIMDCapabilities::Level level) {
  switch (level) {
    case SIMDCapabilities::Level::AVX512:
      return "AVX512";
    case SIMDCapabilities::Level::AVX2:
      return "AVX2";
    case SIMDCapabilities::Level::SSE:
      return "SSE";
    case SIMDCapabilities::Level::NEON:
      return "NEON";
    case SIMDCapabilities::Level::NONE:
      return "Generic";
    default:
      return "Unknown";
  }
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "SIMD Capabilities Detection",
                 "[distance][simd]") {
  // 测试SIMD能力检测
  auto optimal = SIMDCapabilities::getOptimalSIMD();

  // 检查返回的是有效的枚举值
  bool valid_simd = (optimal == SIMDCapabilities::Level::AVX512 ||
                     optimal == SIMDCapabilities::Level::AVX2 ||
                     optimal == SIMDCapabilities::Level::SSE ||
                     optimal == SIMDCapabilities::Level::NEON ||
                     optimal == SIMDCapabilities::Level::NONE);
  REQUIRE(valid_simd);

  // 输出检测到的SIMD能力
  std::cout << "Detected SIMD capability: " << simdLevelToString(optimal)
            << std::endl;
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "L2 Distance Computer",
                 "[distance][l2]") {
  auto computer = std::make_unique<L2DistanceComputer<float>>(dim_);

  SECTION("Basic functionality") {
    // 测试基本功能
    float distance = computer->compute(data_a_.data(), data_b_.data());
    REQUIRE(distance >= 0.0f);
  }

  SECTION("Self distance should be zero") {
    // 测试相同向量的距离应该为0
    float self_distance = computer->compute(data_a_.data(), data_a_.data());
    REQUIRE(self_distance == Approx(0.0f).margin(1e-6f));
  }

  SECTION("Computer name") {
    // 测试名称
    REQUIRE(computer->name() == "L2Distance_FP32");
  }
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "IP Distance Computer",
                 "[distance][ip]") {
  auto computer = std::make_unique<IPDistanceComputer<float>>(dim_);

  SECTION("Basic functionality") {
    // 测试基本功能
    float distance = computer->compute(data_a_.data(), data_b_.data());
    REQUIRE(distance >= 0.0f);
  }

  SECTION("Computer name") {
    // 测试名称
    REQUIRE(computer->name() == "IPDistance_FP32");
  }
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "Cosine Distance Computer",
                 "[distance][cosine]") {
  auto computer = std::make_unique<CosineDistanceComputer<float>>(dim_);

  SECTION("Basic functionality") {
    // 测试基本功能
    float distance = computer->compute(data_a_.data(), data_b_.data());
    REQUIRE(distance >= 0.0f);
    REQUIRE(distance <= 2.0f);  // 余弦距离范围[0,2]
  }

  SECTION("Computer name") {
    // 测试名称
    REQUIRE(computer->name() == "CosineDistance");
  }
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "Distance Computer Factory",
                 "[distance][factory]") {
  SECTION("Create L2 distance computer") {
    // 测试工厂创建L2距离计算器
    auto l2_computer =
        DistanceComputerFactory::create<float>(DistanceType::L2, dim_);
    REQUIRE(l2_computer != nullptr);
    REQUIRE(l2_computer->name() == "L2Distance_FP32");
  }

  SECTION("Create IP distance computer") {
    // 测试工厂创建IP距离计算器
    auto ip_computer =
        DistanceComputerFactory::create<float>(DistanceType::IP, dim_);
    REQUIRE(ip_computer != nullptr);
    REQUIRE(ip_computer->name() == "IPDistance_FP32");
  }

  SECTION("Create Cosine distance computer") {
    // 测试工厂创建Cosine距离计算器
    auto cosine_computer =
        DistanceComputerFactory::create<float>(DistanceType::COSINE, dim_);
    REQUIRE(cosine_computer != nullptr);
    REQUIRE(cosine_computer->name() == "CosineDistance");
  }

  SECTION("Supported types") {
    // 测试支持的类型
    auto supported_types = DistanceComputerFactory::getSupportedTypes();
    REQUIRE(supported_types.size() >= 3);

    REQUIRE(DistanceComputerFactory::isTypeSupported(DistanceType::L2));
    REQUIRE(DistanceComputerFactory::isTypeSupported(DistanceType::IP));
    REQUIRE(DistanceComputerFactory::isTypeSupported(DistanceType::COSINE));
  }
}

TEST_CASE_METHOD(DistanceComputerTestFixture, "Performance Comparison",
                 "[distance][performance]") {
  const size_t num_iterations = 10000;

  auto l2_computer =
      DistanceComputerFactory::create<float>(DistanceType::L2, dim_);

  // 预热
  for (int i = 0; i < 100; ++i) {
    l2_computer->compute(data_a_.data(), data_b_.data());
  }

  // 性能测试
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_iterations; ++i) {
    volatile float result =
        l2_computer->compute(data_a_.data(), data_b_.data());
    (void)result;  // 避免编译器优化
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double avg_time = static_cast<double>(duration.count()) / num_iterations;

  std::cout << "Average L2 distance computation time: " << avg_time
            << " microseconds" << std::endl;

  // 使用辅助函数转换枚举为字符串
  auto simd_level = SIMDCapabilities::getOptimalSIMD();
  std::cout << "SIMD capability: " << simdLevelToString(simd_level)
            << std::endl;

  // 基本性能要求：每次计算应该在合理时间内完成
  REQUIRE(avg_time < 100.0);  // 小于100微秒
}
