#pragma once

#include "graph.h"

namespace deepsearch {
class Builder {
 public:
  virtual int Dim() = 0;
  virtual void Build(float* data, int nb) = 0;

  virtual Graph<int> GetGraph() = 0;

  virtual ~Builder() = default;
};
}  // namespace deepsearch
