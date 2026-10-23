#pragma once
#include "tensor.hpp"

namespace nnedge::op {
// One K32 group of eight output channels: BF16 scales[8], FP4 tiles[4].
struct FP4SlotLayout {
    static constexpr unsigned group_k = 32, tiles = 4, scales = 8;
    static constexpr unsigned slot_bytes = 144, ring_slots = 4;
    static constexpr unsigned wtype = EDGE_TENSOR_WTYPE_FP4, stype = 0;
    static constexpr bool transpose = false;
};

inline bool linear_fp4(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                       Tensor<uint8_t> slots)
{
    using L = FP4SlotLayout;
    if (x.shape.rank != 2 || y.shape.rank != 2) return false;
    const size_t k = x.shape.dims[1], n = y.shape.dims[1];
    const size_t xb = align_bytes(k*2), yb = align_bytes(n*2);
    const size_t count = (n/8)*(k/L::group_k);
    if (x.shape.rank != 2 || y.shape.rank != 2 || x.shape.dims[0] != 1 ||
        y.shape.dims[0] != 1 || !k || !n || k%32 || n%8 ||
        xb+yb+128+L::ring_slots*L::slot_bytes > kDtcmOpScratchBytes ||
        numel(slots) != count*L::slot_bytes || !x.data || !y.data || !slots.data)
        return false;
    auto *scratch = dtcm_op_scratch<unsigned char>();
    Tensor<bfloat16_t> xs(x.shape, reinterpret_cast<bfloat16_t *>(scratch));
    Tensor<bfloat16_t> ys(y.shape, reinterpret_cast<bfloat16_t *>(scratch+xb));
    if (is_dtcm_addr(x.data)) xs = x;
    else copy(xs, x); // Finish staging before leasing circular DMA.
    if (is_dtcm_addr(y.data)) ys = y;
    if (alloc_failed()) return false;
    auto *psum = scratch+xb+yb; // L0 at +0, H0 at +64; reserve full striped group.
    edge_tensor_setcsr<0, L::wtype>(); // Tile mode selects BF22-in-FP32 P/O.
    edge_tensor_setn(1);
    edge_dcache_clean_range(slots.data, count*L::slot_bytes);
    edge_dma_start_strided_circular(slots.data, psum+128,
        L::slot_bytes, L::slot_bytes, count, 0u, 1u,
        L::slot_bytes, L::ring_slots);
    edge_tensor_wsld_circular<L::transpose,L::tiles,L::wtype,L::scales,L::stype>();
    for (size_t ob=0; ob<n/8; ++ob) {
        for (size_t kb=0; kb<k/8; ++kb) {
            edge_tensor_setin(xs.data+kb*8);
            const bool last = kb+1 == k/8;
            edge_tensor_setout(last ? (void *)(ys.data+ob*8) : (void *)psum);
            if (kb==0) {
                edge_tensor_start<EDGE_TENSOR_START_OPT_USE_SCALE | EDGE_TENSOR_START_FIRST_TILE>();
            } else {
                edge_tensor_setpsum(psum);
                if(last) edge_tensor_start<EDGE_TENSOR_START_OPT_USE_SCALE | EDGE_TENSOR_START_LAST_TILE>();
                else edge_tensor_start<EDGE_TENSOR_START_OPT_USE_SCALE | EDGE_TENSOR_START_PSUM_TILE>();
            }
        }
    }
    edge_tensor_sync();
    edge_dma_sync();
    if (y.data != ys.data) copy(y, ys);
    return !alloc_failed();
}
} // namespace nnedge::op
