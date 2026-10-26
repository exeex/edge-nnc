#pragma once
#include "linear_weight_slots.hpp"

namespace nnedge::op {
using FP4SlotLayout = WeightSlotLayout<EDGE_TENSOR_WTYPE_FP4>;
inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x, Tensor<uint8_t> slots)
{
    return linear_weight_slots<EDGE_TENSOR_WTYPE_FP4>(y, x, slots);
}
} // namespace nnedge::op
