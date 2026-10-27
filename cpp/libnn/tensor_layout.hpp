#pragma once
#include "tensor.hpp"

namespace nnedge {
inline bool copy_tensor_payload(Tensor<bfloat16_t> dst, Tensor<bfloat16_t> src)
{
    if (!dst.data || !src.data || numel(dst)!=numel(src)) return false;
    if (dst.data==src.data) return true;
    if (is_dtcm_addr(dst.data) || is_dtcm_addr(src.data)) {
        copy(dst,src);
        return true;
    }
    const size_t n=numel(src), capacity=kDtcmOpScratchBytes/2;
    auto *stage=dtcm_op_scratch<bfloat16_t>();
    for(size_t base=0;base<n;base+=capacity) {
        const size_t count=n-base<capacity ? n-base : capacity;
        edge_dma_start(src.data+base,stage,count*2);
        edge_dma_sync();
        edge_dma_start(stage,dst.data+base,count*2);
        edge_dma_sync();
    }
    return true;
}

// One hardware XY descriptor gathers source fragments into a contiguous
// destination stream. No scalar payload loops or Tensor COPY instructions.
inline void layout_dma_xy(const void *src, void *dst, size_t fragment,
                          size_t xs, size_t xc, size_t ys, size_t yc)
{
    // Circular mode with a single full-transfer slot is not needed: normal
    // strided DMA accepts both XY axes through its descriptor CSRs.
    edge_dma_setn(fragment);
    edge_dma_setx(xs,xc);
    edge_dma_sety(ys,yc);
    edge_dma_setsrc(edge_addr_from_ptr(src));
    edge_dma_settar(edge_addr_from_ptr(dst));
    edge32::dma_start(static_cast<uint32_t>(fragment),1);
    edge_dma_sync();
}

inline bool convert_tensor_layout(Tensor<bfloat16_t> dst,
                                  Tensor<bfloat16_t> src, bool to_tiles)
{
    if (!src.data || !dst.data || numel(src)!=numel(dst) ||
        (src.shape.rank!=2 && src.shape.rank!=3)) return false;
    const size_t heads = src.shape.rank==3 ? src.shape.dims[0] : 1;
    const size_t rows = src.shape.dims[src.shape.rank-2];
    const size_t cols = src.shape.dims[src.shape.rank-1];
    if (!rows || !cols || cols%8) return false;
    const bool sd = is_dtcm_addr(src.data), dd = is_dtcm_addr(dst.data);
    if (sd && dd) {
        // Explicit graph boundary between incompatible ASIC layouts. Stage in
        // DRAM; DMA has no SRAM-to-SRAM gather/scatter path.
        Tensor<bfloat16_t> bridge(src.shape);
        malloc_tensor_dram(bridge);
        if (alloc_failed()) return false;
        copy(bridge,src);
        const bool ok = convert_tensor_layout(dst,bridge,to_tiles);
        free_tensor(bridge);
        return ok;
    }
    if (sd != dd) {
        for (size_t h=0;h<heads;++h) {
            auto *s=src.data+h*rows*cols;
            auto *d=dst.data+h*rows*cols;
            layout_dma_xy(s,d,16,to_tiles ? cols*2 : rows*16,
                          to_tiles ? rows : cols/8,
                          16,to_tiles ? cols/8 : rows);
        }
        return true;
    }
    // Both payloads are in DRAM (large graph edge). Stage bounded row chunks
    // through the dedicated scratch region; destination stride is not assumed.
    const size_t capacity = kDtcmOpScratchBytes/(cols*2);
    if (!capacity) return false;
    auto *stage=dtcm_op_scratch<bfloat16_t>();
    for (size_t h=0;h<heads;++h) for(size_t base=0;base<rows;base+=capacity) {
        const size_t batch=rows-base<capacity ? rows-base : capacity;
        auto *s=src.data+h*rows*cols;
        auto *d=dst.data+h*rows*cols;
        if (to_tiles) {
            layout_dma_xy(s+base*cols,stage,16,cols*2,batch,16,cols/8);
            for(size_t block=0;block<cols/8;++block) {
                edge_dma_start(stage+block*batch*8,d+(block*rows+base)*8,batch*16);
                edge_dma_sync();
            }
        } else {
            layout_dma_xy(s+base*8,stage,16,rows*16,cols/8,16,batch);
            edge_dma_start(stage,d+base*cols,batch*cols*2);
            edge_dma_sync();
        }
    }
    return true;
}
} // namespace nnedge
