#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "builder.h"
#include "core/interfaces.h"

namespace deepsearch {
namespace graph {

enum class BuilderType { HNSW, BRUTEFORCE, RANDOM };

template <typename T>
class BuilderFactory {
 public:
  static std::unique_ptr<GraphBuilder<T>> create(
      BuilderType type, core::DistanceType distance_type, size_t dimension,
      const BuilderConfig& config = BuilderConfig{});

  static std::vector<BuilderType> supported_types();
  static std::string type_name(BuilderType type);
  static BuilderType parse_type(const std::string& name);

 private:
  static std::unordered_map<std::string, BuilderType> type_map_;
};

}  // namespace graph
}  // namespace deepsearch
