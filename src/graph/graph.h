#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "allocator.h"
#include "graph/HnswInitializer.h"
#include "simd/distance.h"

namespace deepsearch {
namespace graph {
constexpr int EMPTY_ID = -1;

// 图元数据
struct GraphMetadata {
  size_t num_nodes = 0;
  size_t max_degree = 0;
  size_t total_edges = 0;
  std::string builder_name;
  std::string distance_type;
  std::vector<int> entry_points;
};

// 图接口
template <typename node_t>
class GraphInterface {
 public:
  virtual ~GraphInterface() = default;

  // 基本访问
  virtual size_t num_nodes() const = 0;
  virtual size_t max_degree() const = 0;
  virtual const node_t* neighbors(size_t node_id) const = 0;
  virtual size_t degree(size_t node_id) const = 0;

  // 搜索支持
  virtual void prefetch_neighbors(size_t node_id, int lines = 1) const = 0;
  virtual const std::vector<size_t>& entry_points() const = 0;

  // 序列化
  virtual void save(const std::string& filename) const = 0;
  virtual void load(const std::string& filename) = 0;

  // 元数据
  virtual GraphMetadata metadata() const = 0;
};

// 密集图实现（当前实现的优化版本）
template <typename node_t>
class DenseGraph : public GraphInterface<node_t> {
 public:
  DenseGraph() = default;
  explicit DenseGraph(size_t num_nodes, size_t max_degree);
  DenseGraph(const DenseGraph& other);
  DenseGraph& operator=(const DenseGraph& other);
  DenseGraph(DenseGraph&& other) noexcept;
  DenseGraph& operator=(DenseGraph&& other) noexcept;
  ~DenseGraph() override;

  // 构建接口
  void initialize(size_t num_nodes, size_t max_degree);
  void set_neighbors(size_t node_id, const node_t* neighbors, size_t count);
  void add_edge(size_t from, size_t to);
  void remove_edge(size_t from, size_t to);

  // 实现基类接口
  size_t num_nodes() const override { return num_nodes_; }
  size_t max_degree() const override { return max_degree_; }
  const node_t* neighbors(size_t node_id) const override;
  size_t degree(size_t node_id) const override;
  void prefetch_neighbors(size_t node_id, int lines = 1) const override;
  const std::vector<size_t>& entry_points() const override {
    return entry_points_;
  }
  void save(const std::string& filename) const override;
  void load(const std::string& filename) override;
  GraphMetadata metadata() const override;

  // 兼容性接口（保持向后兼容）
  node_t at(size_t i, size_t j) const { return data_[i * max_degree_ + j]; }
  node_t& at(size_t i, size_t j) { return data_[i * max_degree_ + j]; }
  const node_t* edges(size_t u) const { return neighbors(u); }
  node_t* edges(size_t u) { return data_ + max_degree_ * u; }

  // 搜索初始化
  template <typename Pool, typename Quant>
  void initialize_search(Pool& pool, const Quant& quant) const {
    if (initializer_) {
      initializer_->initialize(pool, quant);
    } else {
      for (auto ep : entry_points_) {
        pool.insert(ep, quant->compute_query_distance(ep));
      }
    }
  }

  // 设置初始化器和入口点
  void set_initializer(std::unique_ptr<HnswInitializer> init) {
    initializer_ = std::move(init);
  }
  void set_entry_points(const std::vector<size_t>& eps) { entry_points_ = eps; }

 private:
  void cleanup();
  void copy_from(const DenseGraph& other);

  size_t num_nodes_ = 0;
  size_t max_degree_ = 0;
  node_t* data_ = nullptr;
  std::vector<size_t> degrees_;  // 每个节点的实际度数
  std::vector<size_t> entry_points_;
  std::unique_ptr<HnswInitializer> initializer_;
  GraphMetadata metadata_;
};

// 类型别名保持兼容性
using Graph = DenseGraph<int>;

}  // namespace graph
}  // namespace deepsearch
