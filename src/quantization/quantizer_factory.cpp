#include <algorithm>
#include <stdexcept>

#include "allocator.h"
#include "distance/computers.h"
#include "fp32_quant.h"
#include "quantizer.h"
#include "simd/distance_functions.h"
#include "sq4_quant.h"
#include "sq8_quant.h"

namespace deepsearch {
namespace quantization {

// QuantizerComputer 实现
// template <typename QuantizerType>
// QuantizerComputer<QuantizerType>::QuantizerComputer(
//     const QuantizerType& quantizer, const float* query)
//     : quantizer_(quantizer) {
//   size_t aligned_size = do_align(quantizer_.dimension(), 16) * sizeof(float);
//   aligned_query_ = (float*)alloc64B(aligned_size);
//   std::memcpy(aligned_query_, query, quantizer_.dimension() * sizeof(float));
// }
//
// template <typename QuantizerType>
// QuantizerComputer<QuantizerType>::~QuantizerComputer() {
//   if (aligned_query_) {
//     free(aligned_query_);
//   }
// }

// template <typename QuantizerType>
// float QuantizerComputer<QuantizerType>::operator()(int u) const {
//   const auto* data = reinterpret_cast<const typename
//   QuantizerType::data_type*>(
//       quantizer_.get_data(u));
//   return distance_computer_->compute(aligned_query_, data,
//                                      quantizer_.dimension());
// }
//
// template <typename QuantizerType>
// void QuantizerComputer<QuantizerType>::prefetch(int u, int lines) const {
//   distance_computer_->prefetch(quantizer_.get_data(u), lines);
// }

// 工厂实现
template <typename InputType, typename CodeType>
std::unique_ptr<QuantizerBase<InputType, CodeType>>
QuantizerFactory<InputType, CodeType>::create(QuantizerType type,
                                              core::DistanceType distanceType,
                                              size_t dim) {
  switch (type) {
    case QuantizerType::FP32:
      if constexpr (std::is_same_v<InputType, float> &&
                    std::is_same_v<CodeType, float>) {
        if (distanceType == core::DistanceType::L2) {
          return std::make_unique<FP32Quantizer>(distanceType, dim);
        } else if (distanceType == core::DistanceType::IP) {
          return std::make_unique<FP32Quantizer>(distanceType, dim);
        }
      }
      break;

    case QuantizerType::SQ8:
      if constexpr (std::is_same_v<InputType, float> &&
                    std::is_same_v<CodeType, uint8_t>) {
        if (distanceType == core::DistanceType::L2) {
          return std::make_unique<SQ8Quantizer>(distanceType, dim);
        } else if (distanceType == core::DistanceType::IP) {
          return std::make_unique<SQ8Quantizer>(distanceType, dim);
        }
      }
      break;

    case QuantizerType::SQ4:
      if constexpr (std::is_same_v<InputType, float> &&
                    std::is_same_v<CodeType, uint8_t>) {
        if (distanceType == core::DistanceType::L2) {
          return std::make_unique<SQ4Quantizer>(distanceType, dim);
        } else if (distanceType == core::DistanceType::IP) {
          return std::make_unique<SQ4Quantizer>(distanceType, dim);
        }
      }
      break;
  }

  throw std::invalid_argument(
      "Unsupported quantizer type or template parameters");
}

template <typename InputType, typename CodeType>
std::vector<QuantizerType>
QuantizerFactory<InputType, CodeType>::get_supported_types() {
  return {QuantizerType::FP32, QuantizerType::SQ8, QuantizerType::SQ4};
}

template <typename InputType, typename CodeType>
bool QuantizerFactory<InputType, CodeType>::is_type_supported(
    QuantizerType type) {
  auto types = get_supported_types();
  return std::find(types.begin(), types.end(), type) != types.end();
}

template <typename InputType, typename CodeType>
std::string QuantizerFactory<InputType, CodeType>::type_name(
    QuantizerType type) {
  switch (type) {
    case QuantizerType::FP32:
      return "FP32";
    case QuantizerType::SQ8:
      return "SQ8";
    case QuantizerType::SQ4:
      return "SQ4";
    default:
      return "Unknown";
  }
}

// 显式实例化
template class QuantizerFactory<float, float>;
template class QuantizerFactory<float, uint8_t>;

}  // namespace quantization
}  // namespace deepsearch
