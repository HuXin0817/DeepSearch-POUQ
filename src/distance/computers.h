#pragma once

#include <memory>
#include <string>

#include "core/interfaces.h"

namespace deepsearch {
namespace distance {

// SIMD能力检测
class SIMDCapabilities {
 public:
  static bool hasAVX512();
  static bool hasAVX2();
  static bool hasSSE();
  static bool hasNEON();
  static std::string getOptimalSIMD();
};

// L2距离计算器
template <typename T>
class L2DistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit L2DistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override;
  void prefetch(const char* data) const override;
  std::string name() const override { return "L2Distance"; }

 private:
  size_t dim_;

  // SIMD优化版本
  float computeAVX512(const T* a, const T* b) const;
  float computeAVX2(const T* a, const T* b) const;
  float computeSSE(const T* a, const T* b) const;
  float computeGeneric(const T* a, const T* b) const;
};

// 内积距离计算器
template <typename T>
class IPDistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit IPDistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override;
  void prefetch(const char* data) const override;
  std::string name() const override { return "IPDistance"; }

 private:
  size_t dim_;

  // SIMD优化版本
  float computeAVX512(const T* a, const T* b) const;
  float computeAVX2(const T* a, const T* b) const;
  float computeSSE(const T* a, const T* b) const;
  float computeGeneric(const T* a, const T* b) const;
};

// 余弦距离计算器
template <typename T>
class CosineDistanceComputer : public core::DistanceComputerTemplate<T> {
 public:
  explicit CosineDistanceComputer(size_t dim) : dim_(dim) {}

  float compute(const T* a, const T* b) const override;
  void prefetch(const char* data) const override;
  std::string name() const override { return "CosineDistance"; }

 private:
  size_t dim_;

  float computeGeneric(const T* a, const T* b) const;
};

// 距离计算器工厂
class DistanceComputerFactory {
 public:
  template <typename T>
  static std::unique_ptr<core::DistanceComputerTemplate<T>> create(
      core::DistanceType type, size_t dim);

  static std::vector<std::string> getSupportedTypes();
  static bool isTypeSupported(core::DistanceType type);
};

}  // namespace distance
}  // namespace deepsearch
