#pragma once

#include <string>
#include <vector>

#include "core/interfaces.h"

namespace deepsearch {
namespace quantization {

// 量化器类型枚举
enum class QuantizerType { FP32, SQ8, SQ4 };

// 量化器基类接口
template <typename InputType, typename CodeType>
class QuantizerBase {
 public:
  virtual ~QuantizerBase() = default;

  // 核心量化接口
  virtual void train(const InputType* data, size_t n, size_t dim) = 0;
  virtual void encode(const InputType* input, CodeType* output) const = 0;
  virtual void decode(const CodeType* input, InputType* output) const = 0;

  // 查询编码接口 - 新增
  virtual void encode_query(const InputType* query) = 0;
  virtual float compute_query_distance(size_t index) const = 0;
  virtual float compute_query_distance(const CodeType* code) const = 0;

  // 信息接口
  virtual size_t code_size() const = 0;
  virtual size_t dimension() const = 0;
  virtual std::string name() const = 0;

  // 数据访问
  virtual const char* get_data(size_t index) const = 0;
  virtual char* get_data(size_t index) = 0;

 protected:
  // 存储编码后的查询向量
  mutable CodeType* query_ = nullptr;
};

// 距离计算器基类
template <typename T>
class DistanceComputerBase {
 public:
  virtual ~DistanceComputerBase() = default;
  virtual float compute(const T* a, const T* b, size_t dim) const = 0;
  virtual void prefetch(const char* ptr, int lines = 1) const = 0;
  virtual std::string name() const = 0;
};

// 量化器工厂
template <typename InputType, typename CodeType>
class QuantizerFactory {
 public:
  static std::unique_ptr<QuantizerBase<InputType, CodeType>> create(
      QuantizerType type, core::DistanceType distanceType, size_t dim);

  static std::vector<QuantizerType> get_supported_types();
  static bool is_type_supported(QuantizerType type);
  static std::string type_name(QuantizerType type);
};

// // 量化器计算机接口（用于搜索）
// template <typename QuantizerType>
// class QuantizerComputer {
//  public:
//   using dist_type = float;
//
//   QuantizerComputer(const QuantizerType& quantizer, const float* query);
//   ~QuantizerComputer();
//
//   dist_type operator()(int u) const;
//   void prefetch(int u, int lines) const;
//
//  private:
//   const QuantizerType& quantizer_;
//   float* aligned_query_;
// };

}  // namespace quantization
}  // namespace deepsearch
