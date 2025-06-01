#pragma once

#include <memory>
#include <string>

#include "core/interfaces.h"
#include "graph.h"
#include "hnswlib/hnswalg.h"

namespace deepsearch {
namespace graph {

// 图构建器配置
struct BuilderConfig {
  size_t M = 16;                  // 连接数
  size_t ef_construction = 200;   // 构建时的搜索参数
  size_t max_elements = 1000000;  // 最大元素数
  size_t random_seed = 100;       // 随机种子
  bool allow_replace_deleted = false;
};

// 图构建器基类
template <typename T>
class GraphBuilder {
 public:
  virtual ~GraphBuilder() = default;

  // 核心接口
  virtual void configure(const BuilderConfig& config) = 0;
  virtual Graph build(const T* data, size_t n, size_t dim) = 0;
  virtual void add_points(const T* data, const size_t* labels, size_t n) = 0;
  virtual void remove_points(const size_t* labels, size_t n) = 0;

  // 信息接口
  virtual size_t dimension() const = 0;
  virtual size_t size() const = 0;
  virtual std::string name() const = 0;
  virtual BuilderConfig get_config() const = 0;
};

class HNSWBuilder : public GraphBuilder<float> {
 public:
  explicit HNSWBuilder(core::DistanceType distance_type, size_t dim);
  ~HNSWBuilder() override = default;

  // 实现基类接口
  void configure(const BuilderConfig& config) override;
  Graph build(const float* data, size_t n, size_t dim) override;
  void add_points(const float* data, const size_t* labels, size_t n) override;
  void remove_points(const size_t* labels, size_t n) override;

  // 信息接口
  size_t dimension() const override { return dim_; }
  size_t size() const override { return current_size_; }
  std::string name() const override { return "HNSWBuilder"; }
  BuilderConfig get_config() const override { return config_; }

  // HNSW 特有接口
  void set_ef_construction(size_t ef) { config_.ef_construction = ef; }
  void set_M(size_t M) { config_.M = M; }

 private:
  void initialize_hnsw();
  Graph extract_graph();

  core::DistanceType distance_type_;
  size_t dim_;
  size_t current_size_ = 0;
  BuilderConfig config_;

  // HNSW 内部实现，固定为 float 类型
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

}  // namespace graph
}  // namespace deepsearch
