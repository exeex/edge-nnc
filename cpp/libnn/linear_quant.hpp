#pragma once
#include "linear_weight_slots.hpp"

namespace nnedge {
// K32, eight per-output-lane 16-bit scales, four output-major 8x8 tiles.
// BF16 is the current scale stand-in (stype=0). Byte-weight packs are all
// 272 bytes / 256 weights = 8.5 bits/weight; scale:weight bits = 1:16.
// These are runtime layouts, not file-format decoders.
constexpr unsigned kQuantByteWeightScaleType = 0;
using FP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, 0, 4, 8>;
using MXFP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, EDGE_TENSOR_STYPE_MXFP4, 4, 8>;
using NVFP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, EDGE_TENSOR_STYPE_NVFP4, 2, 8>;
static_assert(sizeof(MXFP4WeightPack) == 144 && sizeof(NVFP4WeightPack) == 80,
              "native byte scales with aligned transport slots");
using Q8_0WeightPack = WeightPack<EDGE_TENSOR_WTYPE_INT8, kQuantByteWeightScaleType, 4, 8>;
using FP8E4M3FNWeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E4M3FN, kQuantByteWeightScaleType, 4, 8>;
using FP8E5M2WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E5M2, kQuantByteWeightScaleType, 4, 8>;
static_assert(sizeof(Q8_0WeightPack) == 272 && sizeof(FP8E4M3FNWeightPack) == 272
              && sizeof(FP8E5M2WeightPack) == 272, "K32 byte-weight storage contract");

namespace op {
using FP4SlotLayout = WeightSlotLayout<EDGE_TENSOR_WTYPE_FP4>;

inline bool linear_mxfp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x, Tensor<uint8_t> slots)
{
    return linear_weight_slots<MXFP4WeightPack>(y,x,slots);
}

inline bool linear_nvfp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                        Tensor<uint8_t> slots, uint32_t global_scale_bits)
{
    return detail::linear_weight_slots_bytes<NVFP4WeightPack>(y,x,slots,global_scale_bits);
}

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
