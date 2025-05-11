#pragma once

#include <unordered_map>

#include "common.h"

namespace deepsearch {

enum class QuantizerType { FP32, SQ8, SQ4 };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ8;
  return 42;
}();

template <Metric metric, int DIM>
class Quantizer {
 public:
  virtual ~Quantizer() = 0;

  virtual void train(const float* data, int64_t n) = 0;

  virtual void encode(const float* from, char* to) = 0;

  virtual char* get_data(int u) const = 0;
};

}  // namespace deepsearch
