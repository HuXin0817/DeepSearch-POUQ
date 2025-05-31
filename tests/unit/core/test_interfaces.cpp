#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "core/factory.h"
#include "core/interfaces.h"

using namespace deepsearch::core;

// 测试用的L2距离计算器实现
class TestL2Computer : public DistanceComputerTemplate<float> {
 public:
  float compute(const float* a, const float* b, size_t dim) const override {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

  std::string name() const override { return "TestL2Computer"; }
};

// 测试用的搜索引擎实现
class TestSearchEngine : public SearchEngineTemplate<float> {
 private:
  std::vector<std::vector<float>> data_;
  size_t dim_;
  std::unique_ptr<DistanceComputerTemplate<float>> distance_computer_;

 public:
  TestSearchEngine() : dim_(0) {
    distance_computer_ = std::make_unique<TestL2Computer>();
  }

  void build(const float* data, size_t n, size_t dim) override {
    dim_ = dim;
    data_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      data_[i].resize(dim);
      for (size_t j = 0; j < dim; ++j) {
        data_[i][j] = data[i * dim + j];
      }
    }
  }

  std::vector<int> search(const float* query, size_t k) const override {
    std::vector<std::pair<float, int>> distances;

    for (size_t i = 0; i < data_.size(); ++i) {
      float dist = distance_computer_->compute(query, data_[i].data(), dim_);
      distances.emplace_back(dist, static_cast<int>(i));
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> result;
    for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
      result.push_back(distances[i].second);
    }

    return result;
  }

  void save(const std::string& path) const override {}
  void load(const std::string& path) override {}

  size_t size() const override { return data_.size(); }

  size_t dimension() const override { return dim_; }
};

TEST_CASE("DistanceComputerTemplate Tests", "[interfaces][distance]") {
  SECTION("L2 distance computation") {
    TestL2Computer computer;

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    float distance = computer.compute(a.data(), b.data(), 3);
    float expected = std::sqrt(27.0f);  // sqrt(9+9+9)

    REQUIRE(std::abs(distance - expected) < 1e-6);
    REQUIRE(computer.name() == "TestL2Computer");
  }

  SECTION("Zero distance") {
    TestL2Computer computer;
    std::vector<float> a = {1.0f, 2.0f, 3.0f};

    float distance = computer.compute(a.data(), a.data(), 3);
    REQUIRE(distance == 0.0f);
  }
}

TEST_CASE("SearchEngineTemplate Tests", "[interfaces][search]") {
  SECTION("Basic search functionality") {
    TestSearchEngine engine;

    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,  // 向量0
        0.0f, 1.0f, 0.0f,  // 向量1
        0.0f, 0.0f, 1.0f,  // 向量2
        1.0f, 1.0f, 0.0f   // 向量3
    };

    engine.build(data.data(), 4, 3);

    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    auto results = engine.search(query.data(), 2);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == 0);  // 最近的应该是向量0
    REQUIRE(engine.size() == 4);
    REQUIRE(engine.dimension() == 3);
  }

  SECTION("Empty search") {
    TestSearchEngine engine;
    std::vector<float> query = {1.0f, 0.0f, 0.0f};

    auto results = engine.search(query.data(), 5);
    REQUIRE(results.empty());
    REQUIRE(engine.size() == 0);
    REQUIRE(engine.dimension() == 0);
  }

  SECTION("Single vector search") {
    TestSearchEngine engine;
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    engine.build(data.data(), 1, 3);

    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    auto results = engine.search(query.data(), 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0] == 0);
  }
}

TEST_CASE("Batch Search Tests", "[interfaces][batch]") {
  SECTION("Multiple queries") {
    TestSearchEngine engine;

    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,  // 向量0
        0.0f, 1.0f, 0.0f,  // 向量1
        0.0f, 0.0f, 1.0f,  // 向量2
    };

    engine.build(data.data(), 3, 3);

    std::vector<float> queries = {
        1.0f, 0.0f, 0.0f,  // 查询0
        0.0f, 1.0f, 0.0f   // 查询1
    };

    auto results = engine.batch_search(queries.data(), 2, 3, 1);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0].size() == 1);
    REQUIRE(results[1].size() == 1);
    REQUIRE(results[0][0] == 0);
    REQUIRE(results[1][0] == 1);
  }

  SECTION("Batch search with k > data size") {
    TestSearchEngine engine;
    std::vector<float> data = {1.0f, 0.0f, 0.0f};
    engine.build(data.data(), 1, 3);

    std::vector<float> queries = {1.0f, 0.0f, 0.0f};
    auto results = engine.batch_search(queries.data(), 1, 3, 5);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].size() == 1);
  }
}

TEST_CASE("Factory Enums Tests", "[interfaces][factory]") {
  SECTION("SearchEngine algorithm types") {
    auto hnsw_type = SearchEngineFactory<float>::AlgorithmType::HNSW;
    auto bruteforce_type =
        SearchEngineFactory<float>::AlgorithmType::BRUTEFORCE;

    REQUIRE(static_cast<int>(hnsw_type) != static_cast<int>(bruteforce_type));
  }

  SECTION("DistanceComputer metric types") {
    auto l2_type = DistanceComputerFactory<float>::MetricType::L2;
    auto ip_type = DistanceComputerFactory<float>::MetricType::IP;

    REQUIRE(static_cast<int>(l2_type) != static_cast<int>(ip_type));
  }
}

TEST_CASE("Interface Inheritance Tests", "[interfaces][inheritance]") {
  SECTION("DistanceComputer polymorphism") {
    std::unique_ptr<DistanceComputerTemplate<float>> computer =
        std::make_unique<TestL2Computer>();

    REQUIRE(computer != nullptr);
    REQUIRE(computer->name() == "TestL2Computer");
  }

  SECTION("SearchEngine polymorphism") {
    std::unique_ptr<SearchEngineTemplate<float>> engine =
        std::make_unique<TestSearchEngine>();

    REQUIRE(engine != nullptr);
    REQUIRE(engine->dimension() == 0);
    REQUIRE(engine->size() == 0);
  }

  SECTION("Interface method calls") {
    TestSearchEngine engine;
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    engine.build(data.data(), 1, 3);

    // 测试保存和加载方法（虽然是空实现）
    engine.save("test.bin");
    engine.load("test.bin");

    REQUIRE(engine.size() == 1);
    REQUIRE(engine.dimension() == 3);
  }
}
