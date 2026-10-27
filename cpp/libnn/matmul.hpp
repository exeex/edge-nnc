#pragma once
#include "tensor_tile_view.hpp"

#ifndef NNEDGE_MATMUL_PROFILE
#define NNEDGE_MATMUL_PROFILE 0
#endif

namespace nnedge::op {
// Runtime BF16 RHS stays in its original matrix order in DRAM. The DMA XY
// slicer builds eight 128-byte tiles in SRAM; no CPU payload manipulation.
constexpr size_t kMatmulRingSlots = 8;
constexpr size_t kMatmulRingBytes = kMatmulRingSlots*128;

inline size_t mm_read_cycle()
{
#if NNEDGE_MATMUL_PROFILE || defined(NNEDGE_PHASE_PROFILE)
    return edge_get_cycle();
#else
    return 0;
#endif
}

inline size_t mm_scratch_bytes(size_t rows, size_t k, size_t n, bool input_sram, bool output_sram)
{
    const size_t staged_rows = (rows+3u)&~3u;
    return (input_sram ? 0 : align_bytes(rows*k*2))
         + (output_sram ? 0 : align_bytes(rows*n*2))
         + align_bytes(staged_rows*n*4)+kMatmulRingBytes;
}

// START snapshots its configuration and copies it forward to the next CSR
// bank (edge_tensor_unit.v). Invariant pointers need not be reissued per tile.
template <bool Transposed>
__attribute__((always_inline)) inline void mm_next_weight(size_t &tiles_left)
{
    if (--tiles_left) {
        if constexpr (Transposed) edge_tensor_wld_circular();
        else edge_tensor_wld_t_circular();
    }
}

template <unsigned Mode, bool Transposed>
__attribute__((always_inline)) inline void mm_output_blocks(
    bfloat16_t *input, bfloat16_t *output, float *psum,
    size_t output_step, size_t psum_step, size_t blocks, size_t &tiles_left)
{
    edge_tensor_setin(input);
    for (size_t ob=0; ob<blocks; ++ob) {
        if constexpr (Mode != EDGE_TENSOR_START_FIRST_TILE && Mode != EDGE_TENSOR_START_SINGLE_TILE)
            edge_tensor_setpsum(psum);
        if constexpr (Mode == EDGE_TENSOR_START_LAST_TILE || Mode == EDGE_TENSOR_START_SINGLE_TILE)
            edge_tensor_setout(output);
        else edge_tensor_setout(psum);
        edge_tensor_start<Mode>();
        mm_next_weight<Transposed>(tiles_left);
        output += output_step;
        psum += psum_step;
    }
}

// Single storage contract: LHS/output are compiler-planned tensor tiles.
// RHS remains the unmodified runtime row-major BF16 matrix in DRAM.
template <unsigned MaxTokens, bool InputSram, bool OutputSram, bool Transposed>
inline bool matmul_tiles(const Tensor<bfloat16_t> &y, const Tensor<bfloat16_t> &lhs,
                           const Tensor<bfloat16_t> &rhs, size_t heads)
{
    constexpr bool transposed = Transposed;
    static_assert(MaxTokens > 0, "token capacity must be positive");
    [[maybe_unused]] const size_t op_start = mm_read_cycle();
    if (!heads || !lhs.data || !rhs.data || !y.data || rhs.shape.rank != 2 ||
        (lhs.shape.rank != 2 && (transposed || lhs.shape.rank != 3))) return false;
    const bool explicit_heads = !transposed && lhs.shape.rank == 3;
    if (explicit_heads && lhs.shape.dims[0] != heads) return false;
    const size_t rows = lhs.shape.dims[explicit_heads ? 1 : 0];
    const size_t lhs_width = lhs.shape.dims[lhs.shape.rank-1];
    const size_t rhs_stride = rhs.shape.dims[1];
    if ((transposed && lhs_width%heads) || (!transposed && rhs_stride%heads)) return false;
    const size_t k = transposed ? lhs_width/heads : lhs_width;
    const size_t n = transposed ? rhs.shape.dims[0] : rhs_stride/heads;
    if (!rows || !k || !n || k%8 || n%8 ||
        (transposed ? rhs_stride != lhs_width : rhs.shape.dims[0] != k)) return false;
    if (transposed) {
        if (y.shape.rank != 3 || y.shape.dims[0] != heads ||
            y.shape.dims[1] != rows || y.shape.dims[2] != n) return false;
    } else if (y.shape.rank != 2 || y.shape.dims[0] != rows || y.shape.dims[1] != heads*n)
        return false;
    if (is_dtcm_addr(rhs.data)) return false;
    constexpr bool input_sram = InputSram, output_sram = OutputSram;
    if ((reinterpret_cast<size_t>(lhs.data)&15u) || (reinterpret_cast<size_t>(y.data)&15u))
        return false;

    // Work on one head/chunk at a time. Borrow only the unused arena tail;
    // explicit SRAM views above the allocator cursor are also protected.
    size_t offset = align_bytes(dtcm_cursor());
    if (input_sram) {
        const size_t lhs_end = reinterpret_cast<size_t>(lhs.data)-dtcm_base()+numel(lhs)*2;
        if (lhs_end > offset) offset = align_bytes(lhs_end);
    }
    if (output_sram) {
        const size_t y_end = reinterpret_cast<size_t>(y.data)-dtcm_base()+numel(y)*2;
        if (y_end > offset) offset = align_bytes(y_end);
    }
    if (offset > kDtcmBytes) return false;
    size_t capacity = rows < MaxTokens ? rows : MaxTokens;
    while (capacity && mm_scratch_bytes(capacity,k,n,input_sram,output_sram) > kDtcmBytes-offset) --capacity;
    if (!capacity) return false;
    auto *scratch = reinterpret_cast<unsigned char *>(dtcm_base()+offset);
    auto *input = reinterpret_cast<bfloat16_t *>(scratch);
    auto *output = reinterpret_cast<bfloat16_t *>(scratch+(input_sram ? 0 : align_bytes(capacity*k*2)));
    auto *psum = reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(output)+
                                         (output_sram ? 0 : align_bytes(capacity*n*2)));
    auto *ring = reinterpret_cast<unsigned char *>(psum)+align_bytes(((capacity+3u)&~3u)*n*4);
    edge_tensor_setwtype<EDGE_TENSOR_WTYPE_BF16>();
    [[maybe_unused]] size_t input_cycles = 0, stream_cycles = 0, output_cycles = 0, chunks = 0;
    const size_t kb_count = k/8, ob_count = n/8;
    for (size_t head=0; head<heads; ++head) {
        for (size_t base=0; base<rows; base+=capacity) {
            const size_t batch = rows-base < capacity ? rows-base : capacity;
            const size_t input_begin = input_sram ? 0 : mm_read_cycle();
            if (!input_sram) {
                const size_t lhs_head = (explicit_heads || transposed) ? head*(k/8)*rows*8 : 0;
                auto *src = lhs.data+lhs_head+base*8;
                if (batch==rows) edge_dma_start(src,input,batch*k*2);
                else edge_dma_start_strided(src,input,batch*16,rows*16,kb_count);
                edge_dma_sync();
                input_cycles += mm_read_cycle()-input_begin;
            }
            edge_tensor_setn(batch);
            const size_t stream_begin = mm_read_cycle();
            // One XY command for the entire head/chunk. Eight 16-byte source
            // fragments form each tile, and producer/consumer recycle 8 slots.
            edge_dma_start_strided_circular(
                rhs.data+head*(transposed ? k : n),ring,16u,rhs_stride*2,
                transposed ? n : k,16u,transposed ? kb_count : ob_count,
                128u,kMatmulRingSlots);
            if (transposed) edge_tensor_wld_circular();
            else edge_tensor_wld_t_circular();
            size_t tiles_left = kb_count*ob_count;
            const size_t lhs_head = (explicit_heads || transposed) ? head*kb_count*rows*8 : 0;
            auto *x_base = input_sram ? lhs.data+lhs_head+base*8 : input;
            auto *y_base = output_sram ? y.data+(head*ob_count*rows+base)*8 : output;
            const size_t x_step = (input_sram ? rows : batch)*8;
            const size_t y_step = (output_sram ? rows : batch)*8;
            const size_t p_step = ((batch+3u)&~3u)*8;
            if (kb_count == 1) {
                mm_output_blocks<EDGE_TENSOR_START_SINGLE_TILE,Transposed>(
                    x_base,y_base,psum,y_step,p_step,ob_count,tiles_left);
            } else if constexpr (Transposed) {
                // The input vector stream is shared by every output block at
                // a fixed K block. Peel K boundaries to make modes constant.
                mm_output_blocks<EDGE_TENSOR_START_FIRST_TILE,true>(
                    x_base,y_base,psum,y_step,p_step,ob_count,tiles_left);
                auto *xp=x_base+x_step;
                for (size_t kb=1;kb+1<kb_count;++kb,xp+=x_step)
                    mm_output_blocks<EDGE_TENSOR_START_PSUM_TILE,true>(
                        xp,y_base,psum,y_step,p_step,ob_count,tiles_left);
                mm_output_blocks<EDGE_TENSOR_START_LAST_TILE,true>(
                    xp,y_base,psum,y_step,p_step,ob_count,tiles_left);
            } else {
                // P and intermediate output stay fixed throughout one K
                // reduction. Only the final tile redirects output to BF16 Y.
                auto *pp=psum;
                auto *yp=y_base;
                for (size_t ob=0;ob<ob_count;++ob,pp+=p_step,yp+=y_step) {
                    edge_tensor_setpsum(pp);
                    edge_tensor_setout(pp);
                    edge_tensor_setin(x_base);
                    edge_tensor_start<EDGE_TENSOR_START_FIRST_TILE>();
                    mm_next_weight<false>(tiles_left);
                    auto *xp=x_base+x_step;
                    for (size_t kb=1;kb+1<kb_count;++kb,xp+=x_step) {
                        edge_tensor_setin(xp);
                        edge_tensor_start<EDGE_TENSOR_START_PSUM_TILE>();
                        mm_next_weight<false>(tiles_left);
                    }
                    edge_tensor_setin(xp);
                    edge_tensor_setout(yp);
                    edge_tensor_start<EDGE_TENSOR_START_LAST_TILE>();
                    mm_next_weight<false>(tiles_left);
                }
            }
            edge_tensor_sync();
            edge_dma_sync();
            stream_cycles += mm_read_cycle()-stream_begin;
            ++chunks;
            const size_t output_begin = output_sram ? 0 : mm_read_cycle();
            if (!output_sram) {
                auto *dst=y.data+head*ob_count*rows*8+base*8;
                if (batch==rows) {
                    edge_dma_start(output,dst,batch*n*2);
                    edge_dma_sync();
                } else {
                    // A partial token slice has gaps between destination channel
                    // blocks. DMA exposes source strides only: one contiguous
                    // burst per block, never a per-token software scatter.
                    for (size_t ob=0;ob<ob_count;++ob) {
                        edge_dma_start(output+ob*batch*8,dst+ob*rows*8,batch*16);
                        edge_dma_sync();
                    }
                }
                output_cycles += mm_read_cycle()-output_begin;
            }
        }
    }
#if NNEDGE_MATMUL_PROFILE || defined(NNEDGE_PHASE_PROFILE)
    const size_t op_cycles = mm_read_cycle()-op_start;
    printf("NNC_MATMUL_PROFILE capacity=%lu chunks=%lu ring_slots=8 input_cycles=%lu stream_cycles=%lu output_cycles=%lu op_cycles=%lu\n",
           (unsigned long)capacity,(unsigned long)chunks,(unsigned long)input_cycles,
           (unsigned long)stream_cycles,(unsigned long)output_cycles,(unsigned long)op_cycles);
#endif
    return !alloc_failed();
}

// Residency only selects scratch allocation. Every specialization uses the
// same tiled operand contract and runtime row-major RHS DMA slicing.
template <unsigned MaxTokens, bool Transposed>
inline bool matmul_dispatch(const Tensor<bfloat16_t> &y, const Tensor<bfloat16_t> &lhs,
                           const Tensor<bfloat16_t> &rhs, size_t heads)
{
    if (is_dtcm_addr(lhs.data)) {
        if (is_dtcm_addr(y.data)) return matmul_tiles<MaxTokens,true,true,Transposed>(y,lhs,rhs,heads);
        return matmul_tiles<MaxTokens,true,false,Transposed>(y,lhs,rhs,heads);
    }
    if (is_dtcm_addr(y.data)) return matmul_tiles<MaxTokens,false,true,Transposed>(y,lhs,rhs,heads);
    return matmul_tiles<MaxTokens,false,false,Transposed>(y,lhs,rhs,heads);
}

template <unsigned MaxTokens = 128>
inline void matmul(const Tensor<bfloat16_t> &y, const Tensor<bfloat16_t> &lhs,
                   const Tensor<bfloat16_t> &rhs, size_t heads=1)
{
    if (!matmul_dispatch<MaxTokens,false>(y,lhs,rhs,heads)) alloc_failed() = true;
}
template <unsigned MaxTokens = 128>
inline void matmul_transpose(const Tensor<bfloat16_t> &y, const Tensor<bfloat16_t> &lhs,
                             const Tensor<bfloat16_t> &rhs, size_t heads=1)
{
    if (!matmul_dispatch<MaxTokens,true>(y,lhs,rhs,heads)) alloc_failed() = true;
}
} // namespace nnedge::op
