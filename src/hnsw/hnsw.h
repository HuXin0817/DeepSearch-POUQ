#pragma once

#include "builder.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"

namespace deepsearch {

class Hnsw : public Builder {
 public:
  int nb, dim;
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float> > hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float> > space = nullptr;

  Graph<int> final_graph;

  Hnsw(int dim, const std::string& metric, int R = 32, int L = 200);

  int Dim() override;

  void Build(float* data, int N) override;

  Graph<int> GetGraph() override { return final_graph; }
};

}  // namespace deepsearch
