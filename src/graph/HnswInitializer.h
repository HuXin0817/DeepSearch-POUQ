#pragma once

#include <fstream>
#include <vector>

namespace deepsearch {

struct HnswInitializer {
  int N, K;
  int ep;
  std::vector<int> levels;
  std::vector<std::vector<int, align_alloc<int>>> lists;
  HnswInitializer() = default;

  explicit HnswInitializer(int n, int k)
      : N(n), K(k), ep(0), levels(n), lists(n) {}

  HnswInitializer(const HnswInitializer& rhs) = default;

  int at(int level, int u, int i) const {
    return lists[u][(level - 1) * K + i];
  }

  int& at(int level, int u, int i) { return lists[u][(level - 1) * K + i]; }

  const int* edges(int level, int u) const {
    return lists[u].data() + (level - 1) * K;
  }

  int* edges(int level, int u) { return lists[u].data() + (level - 1) * K; }

  template <typename Pool, typename Quant>
  void initialize(Pool& pool, const Quant& quant) const {
    int u = ep;
    auto cur_dist = quant->compute_query_distance(u);
    for (int level = levels[u]; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        const int* list = edges(level, u);
        for (int i = 0; i < K && list[i] != -1; ++i) {
          int v = list[i];
          auto dist = quant->compute_query_distance(v);
          if (dist < cur_dist) {
            cur_dist = dist;
            u = v;
            changed = true;
          }
        }
      }
    }
    pool.insert(u, cur_dist);
    pool.vis.set(u);
  }

  void load(std::ifstream& reader) {
    reader.read((char*)&N, 4);
    reader.read((char*)&K, 4);
    reader.read((char*)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur;
      reader.read((char*)&cur, 4);
      levels[i] = cur / K;
      lists[i].assign(cur, -1);
      reader.read((char*)lists[i].data(), cur * 4);
    }
  }

  void save(std::ofstream& writer) const {
    writer.write((char*)&N, 4);
    writer.write((char*)&K, 4);
    writer.write((char*)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur = levels[i] * K;
      writer.write((char*)&cur, 4);
      writer.write((char*)lists[i].data(), cur * 4);
    }
  }
};

}  // namespace deepsearch
