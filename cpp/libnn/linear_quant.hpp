#pragma once
#include "linear_weight_slots.hpp"

namespace nnedge {
// K32, eight per-output-lane 16-bit scales, four output-major 8x8 tiles.
// Q8_0 uses native FP16 scales; FP8 presets use BF16 scales. Packs are all
// 272 bytes / 256 weights = 8.5 bits/weight; scale:weight bits = 1:16.
// These are runtime layouts, not file-format decoders.
constexpr unsigned kQuantByteWeightScaleType = 0;
using FP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, 0, 4, 8>;
using MXFP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, EDGE_TENSOR_STYPE_MXFP4, 4, 8, 2>;
using NVFP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4, EDGE_TENSOR_STYPE_NVFP4, 2, 8, 2>;
static_assert(sizeof(MXFP4WeightPack) == 272 && sizeof(NVFP4WeightPack) == 144,
              "two dense quant groups per aligned transport slot");
using Q8_0WeightPack = WeightPack<EDGE_TENSOR_WTYPE_INT8, EDGE_TENSOR_STYPE_FP16, 4, 8>;
using FP8E4M3FNWeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E4M3FN, kQuantByteWeightScaleType, 4, 8>;
using FP8E5M2WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP8_E5M2, kQuantByteWeightScaleType, 4, 8>;
static_assert(sizeof(Q8_0WeightPack) == 272 && sizeof(FP8E4M3FNWeightPack) == 272
              && sizeof(FP8E5M2WeightPack) == 272, "K32 byte-weight storage contract");

namespace op {
using FP4SlotLayout = WeightSlotLayout<EDGE_TENSOR_WTYPE_FP4>;

// Compile-time token ceilings; omitted arguments preserve the 64-token kernel.
// Larger variants may borrow unused SRAM arena space for private staging.

template <unsigned MaxTokens = 64>
inline bool linear_mxfp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x, Tensor<uint8_t> slots)
{
    return linear_weight_slots<MXFP4WeightPack,MaxTokens>(y,x,slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_nvfp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                        Tensor<uint8_t> slots, uint32_t global_scale_bits)
{
    return detail::linear_weight_slots_bytes<NVFP4WeightPack,MaxTokens>(y,x,slots,global_scale_bits);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP4WeightPack> slots)
{
    return linear_weight_slots<FP4WeightPack,MaxTokens>(y, x,
        Tensor<uint8_t>(Shape(numel(slots)*sizeof(FP4WeightPack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}
template <unsigned MaxTokens = 64>
inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP4WeightPack,MaxTokens>(y, x, slots);
}

// Q8_0 runtime spec: preserve native FP16 scale bits and INT8 codes,
// reordered into E3 tiles before calling this entry point.
template <unsigned MaxTokens = 64>
inline bool linear_q8_0(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<Q8_0WeightPack> slots)
{
    return linear_weight_slots<Q8_0WeightPack,MaxTokens>(y, x,
        Tensor<uint8_t>(Shape(numel(slots)*sizeof(Q8_0WeightPack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}
template <unsigned MaxTokens = 64>
inline bool linear_q8_0(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<Q8_0WeightPack,MaxTokens>(y, x, slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp8_e4m3fn(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP8E4M3FNWeightPack> slots)
{
    return linear_weight_slots<FP8E4M3FNWeightPack,MaxTokens>(y, x,
        Tensor<uint8_t>(Shape(numel(slots)*sizeof(FP8E4M3FNWeightPack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}
template <unsigned MaxTokens = 64>
inline bool linear_fp8_e4m3fn(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP8E4M3FNWeightPack,MaxTokens>(y, x, slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp8_e5m2(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<FP8E5M2WeightPack> slots)
{
    return linear_weight_slots<FP8E5M2WeightPack,MaxTokens>(y, x,
        Tensor<uint8_t>(Shape(numel(slots)*sizeof(FP8E5M2WeightPack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}
template <unsigned MaxTokens = 64>
inline bool linear_fp8_e5m2(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                              Tensor<uint8_t> slots)
{
    return linear_weight_slots<FP8E5M2WeightPack,MaxTokens>(y, x, slots);
}

// Tensor working SRAM overloads bypass all I/O staging and COPY commands.
// Public contiguous Tensor overloads above retain their existing semantics.
template <unsigned MaxTokens = 64>
inline bool linear_mxfp4(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots)
{
    return detail::linear_weight_slots_bytes<MXFP4WeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_nvfp4(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots, uint32_t global_scale_bits)
{
    return detail::linear_weight_slots_bytes<NVFP4WeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots,global_scale_bits);
}

template <unsigned MaxTokens = 64>
inline bool linear_q8_0(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots)
{
    return detail::linear_weight_slots_bytes<Q8_0WeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp4(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots)
{
    return detail::linear_weight_slots_bytes<FP4WeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp8_e4m3fn(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots)
{
    return detail::linear_weight_slots_bytes<FP8E4M3FNWeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots);
}

template <unsigned MaxTokens = 64>
inline bool linear_fp8_e5m2(TensorTileView y, TensorTileView x,
                         Tensor<uint8_t> slots)
{
    return detail::linear_weight_slots_bytes<FP8E5M2WeightPack,MaxTokens,true>(
        y.tensor,x.tensor,slots);
}

} // namespace op
} // namespace nnedge
