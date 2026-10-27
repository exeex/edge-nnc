#pragma once
#include "tensor.hpp"

namespace nnedge {
// Explicit tensor storage view: physical [channel/8, token, 8], in SRAM or DRAM.
// Logical dimensions remain [tokens, channels], optionally with a head axis.
// Never implicitly convert a
// contiguous Tensor: NNC plans this storage before the ordinary DMA boundary.
struct TensorTileView {
    Tensor<bfloat16_t> tensor;
    explicit TensorTileView(Tensor<bfloat16_t> value) : tensor(value) {}
};
} // namespace nnedge

namespace nnedge::op {

// Preplanned tensor layout at both ends: ordinary byte-preserving DMA.
// A full tensor needs no runtime gather/scatter or SRAM-to-SRAM copy.
inline bool tensor_sram_transfer_valid(TensorTileView sram, TensorTileView dram)
{
    const auto &t = sram.tensor;
    const auto &d = dram.tensor;
    if (!t.data || !d.data || (t.shape.rank != 2 && t.shape.rank != 3) ||
        t.shape.rank != d.shape.rank || t.shape.dims[t.shape.rank-1]%8) return false;
    for (size_t axis=0; axis<t.shape.rank; ++axis)
        if (!t.shape.dims[axis] || t.shape.dims[axis] != d.shape.dims[axis]) return false;
    return is_dtcm_addr(t.data) && !is_dtcm_addr(d.data) &&
        !(reinterpret_cast<size_t>(t.data)&15u) &&
        reinterpret_cast<size_t>(t.data)-dtcm_base()+numel(t)*2 <= kDtcmBytes;
}

inline bool load_tensor_sram(TensorTileView dst, TensorTileView src)
{
    if (!tensor_sram_transfer_valid(dst,src)) return false;
    copy(dst.tensor,src.tensor);
    return !alloc_failed();
}

inline bool store_tensor_sram(TensorTileView dst, TensorTileView src)
{
    if (!tensor_sram_transfer_valid(src,dst)) return false;
    copy(dst.tensor,src.tensor);
    return !alloc_failed();
}

} // namespace nnedge::op
