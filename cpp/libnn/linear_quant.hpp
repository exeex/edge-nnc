#pragma once
#include "linear_weight_slots.hpp"

namespace nnedge {
// Default E3 quant specs: K32, eight per-output-lane BF16 scales, four
// output-major 8x8 tiles. These are runtime layouts, not file-format decoders.
using FP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, 0, 4, 8>;
using Q8_0WeightPack = WeightPack<EDGE_TENSOR_WTYPE_INT8, 0, 4, 8>;
using FP8E4M3FNWeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E4M3FN, 0, 4, 8>;
using FP8E5M2WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E5M2, 0, 4, 8>;

namespace op {
using FP4SlotLayout = WeightSlotLayout<EDGE_TENSOR_WTYPE_FP4>;

inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP4WeightPack> slots)
{
    return linear_weight_slots(y, x, slots);
}
inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP4WeightPack>(y, x, slots);
}

// Q8_0-style runtime spec; GGUF FP16 scales must be converted to BF16
// and weights reordered into E3 tiles before calling this entry point.
inline bool linear_q8_0(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<Q8_0WeightPack> slots)
{
    return linear_weight_slots(y, x, slots);
}
inline bool linear_q8_0(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<Q8_0WeightPack>(y, x, slots);
}

inline bool linear_fp8_e4m3fn(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP8E4M3FNWeightPack> slots)
{
    return linear_weight_slots(y, x, slots);
}
inline bool linear_fp8_e4m3fn(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP8E4M3FNWeightPack>(y, x, slots);
}

inline bool linear_fp8_e5m2(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP8E5M2WeightPack> slots)
{
    return linear_weight_slots(y, x, slots);
}
inline bool linear_fp8_e5m2(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP8E5M2WeightPack>(y, x, slots);
}

} // namespace op
} // namespace nnedge
