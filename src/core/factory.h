#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "interfaces.h"

namespace deepsearch {
namespace core {

// 模板化的搜索引擎工厂
template <typename T>
class SearchEngineFactory {
 public:
  enum class AlgorithmType { HNSW, BRUTEFORCE, GRAPH_SEARCH };

  enum class MetricType { L2, IP, COSINE };

  static std::unique_ptr<SearchEngineTemplate<T>> create(
      AlgorithmType algo_type, MetricType metric_type,
      const std::unordered_map<std::string, std::any>& params = {});

  static std::string algorithm_name(AlgorithmType type);
  static std::string metric_name(MetricType type);
};

// 模板化的距离计算器工厂
template <typename T>
class DistanceComputerFactory {
 public:
  enum class MetricType { L2, IP, COSINE, HAMMING };

  static std::unique_ptr<DistanceComputerTemplate<T>> create(
      MetricType metric_type,
      const std::unordered_map<std::string, std::any>& params = {});

  static std::string metric_name(MetricType type);
};

// 量化器工厂
template <typename InputType, typename CodeType>
class QuantizerFactory {
 public:
  enum class QuantizerType { SQ4, SQ8, PQ, NONE };

  static std::unique_ptr<Quantizer<InputType, CodeType>> create(
      QuantizerType type,
      const std::unordered_map<std::string, std::any>& params = {});

  static std::string quantizer_name(QuantizerType type);
};

}  // namespace core
}  // namespace deepsearch
