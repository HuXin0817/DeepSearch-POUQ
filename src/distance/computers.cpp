#include "computers.h"

#include <algorithm>
#include <cmath>

#include "simd/distance_functions.h"

namespace deepsearch {
namespace distance {

// L2距离计算器实现
// template <typename T, QuantizerType Quantizer = QuantizerType::FP32>
// float L2DistanceComputer<T, Quantizer>::compute(const T* a, const T* b) const
// {
// }

// template <typename T>
// void L2DistanceComputer<T>::prefetch(const char* data) const {
//   simd::prefetch(data);
// }
//
// template <typename T>
// float L2DistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
//   float result = 0;
//   for (size_t i = 0; i < dim_; ++i) {
//     float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
//     result += diff * diff;
//   }
//   return result;
// }

// 内积距离计算器实现
// template <typename T>
// float IPDistanceComputer<T>::compute(const T* a, const T* b) const {
//   if constexpr (std::is_same_v<T, float>) {
//     // 使用新的统一SIMD接口
//     return 1.0f - simd::IP(a, b, dim_);  // 转换为距离
//   }
//   return computeGeneric(a, b);
// }
//
// template <typename T>
// void IPDistanceComputer<T>::prefetch(const char* data) const {
//   simd::prefetch(data);
// }
//
// // 移除所有SIMD特定的实现函数（computeAVX512, computeAVX2, computeSSE）
// // 这些现在由统一的simd::IP函数处理
//
// template <typename T>
// float IPDistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
//   float result = 0;
//   for (size_t i = 0; i < dim_; ++i) {
//     result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
//   }
//   return 1.0 - result;  // 转换为距离
// }

// 余弦距离计算器实现
template <typename T>
float CosineDistanceComputer<T>::compute(const T* a, const T* b) const {
  if constexpr (std::is_same_v<T, float>) {
    // 使用新的统一SIMD接口
    return simd::CosineDistance(a, b, dim_);
  }
  return computeGeneric(a, b);
}

template <typename T>
float CosineDistanceComputer<T>::computeGeneric(const T* a, const T* b) const {
  float dot_product = 0;
  float norm_a = 0;
  float norm_b = 0;

  for (size_t i = 0; i < dim_; ++i) {
    float val_a = static_cast<float>(a[i]);
    float val_b = static_cast<float>(b[i]);
    dot_product += val_a * val_b;
    norm_a += val_a * val_a;
    norm_b += val_b * val_b;
  }

  float norm_product = std::sqrt(norm_a * norm_b);
  if (norm_product == 0) return 1.0;  // 避免除零

  return 1.0 - (dot_product / norm_product);
}

std::vector<std::string> DistanceComputerFactory::getSupportedTypes() {
  return {"L2", "IP", "COSINE"};
}

bool DistanceComputerFactory::isTypeSupported(core::DistanceType type) {
  return type == core::DistanceType::L2 || type == core::DistanceType::IP ||
         type == core::DistanceType::COSINE;
}

// 显式实例化
template class L2DistanceComputer<float>;
template class IPDistanceComputer<float>;
template class CosineDistanceComputer<float>;

// 添加 uint8_t 类型的实例化
template class L2DistanceComputer<uint8_t>;
template class IPDistanceComputer<uint8_t>;
template class CosineDistanceComputer<uint8_t>;

template std::unique_ptr<core::DistanceComputerTemplate<float>>
DistanceComputerFactory::create<float>(core::DistanceType, size_t);

// 添加 uint8_t 类型的工厂方法实例化
template std::unique_ptr<core::DistanceComputerTemplate<uint8_t>>
DistanceComputerFactory::create<uint8_t>(core::DistanceType, size_t);

}  // namespace distance
}  // namespace deepsearch
