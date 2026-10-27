#pragma once
#include "weight_pack.hpp"
#include "tensor_tile_view.hpp"

#ifndef NNEDGE_QUANT_PROFILE
#define NNEDGE_QUANT_PROFILE 0
#endif

namespace nnedge::op {
// Legacy layout name retained for byte-oriented adapters.
template <unsigned WType> struct WeightSlotLayout : WeightPack<WType> {
    static constexpr unsigned ring_slots = 4;
    static constexpr bool transpose = false;
};

// DTCM-to-DTCM movement uses the E3 Tensor byte-copy path. DMA cannot
// address two DTCM endpoints. Both source and destination are 16-byte aligned.
inline void weight_copy_vector(bfloat16_t *dst, const bfloat16_t *src)
{
    edge_tensor_setin(src);
    edge_tensor_setout(dst);
    edge_tensor_setn(16u);
    edge_tensor_start<EDGE_TENSOR_START_OPT_COPY>();
    edge_tensor_sync();
}

inline size_t weight_psum_bytes(size_t rows)
{
    // Wide H4 uses contiguous FP32 containers. Retain the single-vector
    // legacy allocation, whose two halves are separated by 64 bytes.
    return rows < 4u ? 128u : align_bytes(rows * 8u * sizeof(float));
}

namespace detail {
// Borrow only the unused arena tail, without changing allocator ownership.
// Explicit SRAM tensors may not have been allocated by malloc_tensor.
inline size_t weight_scratch_after(size_t offset, const void *data, size_t bytes)
{
    if (is_dtcm_addr(data)) {
        const size_t end = reinterpret_cast<size_t>(data) - dtcm_base() + bytes;
        if (end > offset) offset = end;
    }
    return align_bytes(offset);
}

template <typename Pack>
inline size_t weight_scratch_bytes(size_t rows, size_t k, size_t n)
{
    return align_bytes(rows*k*2) + align_bytes(rows*n*2)
           + weight_psum_bytes(rows) + Pack::ring_bytes;
}

template <typename Pack, unsigned MaxTokens>
inline size_t weight_token_capacity(size_t rows, size_t k, size_t n, size_t bytes)
{
    static_assert(MaxTokens > 0, "token capacity must be positive");
    size_t capacity = rows < MaxTokens ? rows : MaxTokens;
    while (capacity && weight_scratch_bytes<Pack>(capacity,k,n) > bytes)
        --capacity;
    return capacity;
}

template <typename Pack,unsigned Groups>
inline void weight_start_scale_slots(const uint8_t *source, unsigned char *ring, size_t count)
{
    constexpr size_t stride = (Pack::payload_bytes*Groups+15u)&~15u;
    using Ring = EdgeCircularDmaLayout<stride,Pack::ring_slots>;
    edge_dma_start_strided_circular(source,ring,stride,stride,count,0u,1u,stride,Ring::slot_count);
    edge_tensor_wsld_circular<false,Pack::tiles,Pack::wtype,Pack::scales,Pack::stype,Groups>();
}

template <typename Pack, unsigned MaxTokens = 64, bool TensorSram = false>
inline bool linear_weight_slots_bytes(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                       Tensor<uint8_t> slots, uint32_t global_scale_bits = 0x3f800000u)
{
    static_assert(MaxTokens > 0, "token capacity must be positive");
#if NNEDGE_QUANT_PROFILE
    const size_t op_start = edge_get_cycle();
    size_t stream_cycles = 0, chunks = 0;
#endif
    using L = Pack;
    constexpr bool nvfp4 = L::stype == EDGE_TENSOR_STYPE_NVFP4;
    constexpr size_t output_bytes = 2u;
    if (nvfp4 && (!(global_scale_bits >> 16) || global_scale_bits >= 0x7f800000u))
        return false;
    constexpr unsigned ring_slots = L::ring_slots;
    if (x.shape.rank != 2 || y.shape.rank != 2) return false;
    const size_t rows = x.shape.dims[0], k = x.shape.dims[1], n = y.shape.dims[1];
    const size_t count = (n/8)*(k/L::group_k);
    const size_t full_slots = count/L::groups_per_slot;
    const size_t single_stride = (L::payload_bytes+15u)&~15u;
    const bool tail_group = count%L::groups_per_slot != 0;
    const size_t stored_bytes = full_slots*L::slot_bytes+(tail_group?single_stride:0);
    if (!rows || y.shape.dims[0] != rows || !k || !n || k%L::group_k || n%8 ||
        numel(slots) != stored_bytes || !x.data || !y.data || !slots.data)
        return false;
    if constexpr (TensorSram) {
        if (!is_dtcm_addr(x.data) || !is_dtcm_addr(y.data) || is_dtcm_addr(slots.data) ||
            (reinterpret_cast<size_t>(x.data)&15u) || (reinterpret_cast<size_t>(y.data)&15u) ||
            reinterpret_cast<size_t>(x.data)-dtcm_base()+rows*k*2 > kDtcmBytes ||
            reinterpret_cast<size_t>(y.data)-dtcm_base()+rows*n*2 > kDtcmBytes)
            return false;
    }
    // As in BF16 Matmul, one weight tile serves an entire token chunk.
    // Blocked I/O is private; the public tensors stay contiguous [M,K]/[M,N].
    size_t capacity = rows < MaxTokens ? rows : MaxTokens;
    size_t scratch_offset = kDtcmOpScratchOffset;
    if constexpr (MaxTokens > 64 || TensorSram) {
        // Keep the default kernel's fixed scratch contract. Larger compiled
        // variants can also use currently unallocated tensor-arena space.
        size_t live_end = align_bytes(dtcm_cursor());
        live_end = weight_scratch_after(live_end, x.data, rows*k*2);
        live_end = weight_scratch_after(live_end, y.data, rows*n*2);
        live_end = weight_scratch_after(live_end, slots.data, stored_bytes);
        if (live_end > kDtcmBytes) return false;
        if ((!TensorSram && weight_scratch_bytes<L>(capacity,k,n) > kDtcmOpScratchBytes) ||
            live_end > scratch_offset)
            scratch_offset = live_end;
    }
    const size_t scratch_bytes = kDtcmBytes - scratch_offset;
    if constexpr (TensorSram) {
        while (capacity && weight_psum_bytes(capacity)+L::ring_bytes > scratch_bytes)
            --capacity;
    } else {
        capacity = weight_token_capacity<L,MaxTokens>(rows,k,n,scratch_bytes);
    }
    if (!capacity) return false;
    auto *scratch = reinterpret_cast<unsigned char *>(dtcm_base()+scratch_offset);
    auto *input = reinterpret_cast<bfloat16_t *>(scratch);
    auto *output = reinterpret_cast<bfloat16_t *>(scratch+(TensorSram ? 0 : align_bytes(capacity*k*2)));
    auto *psum = reinterpret_cast<unsigned char *>(output)+(TensorSram ? 0 : align_bytes(capacity*n*output_bytes));
    auto *ring = psum+weight_psum_bytes(capacity);
    const bool input_dtcm = is_dtcm_addr(x.data), output_dtcm = is_dtcm_addr(y.data);

    for (size_t base=0; base<rows; base+=capacity) {
        const size_t batch = rows-base < capacity ? rows-base : capacity;
        auto *xs = input;
        auto *ys = output;
        if constexpr (TensorSram) {
            xs = x.data+base*8;
            ys = y.data+base*8;
        } else if (batch==1) {
            if (input_dtcm) xs=x.data+base*k;
            else {
                edge_dma_start(x.data+base*k, input, k*2);
                edge_dma_sync();
            }
            if (output_dtcm) ys=y.data+base*n;
        } else {
            // Gather [token,K] into [K/8,token,8] before leasing circular DMA.
            for (size_t kb=0; kb<k/8; ++kb) {
                if (!input_dtcm) {
                    edge_dma_start_strided(x.data+base*k+kb*8,
                        input+kb*batch*8, 16u, k*2, batch);
                    edge_dma_sync();
                } else {
                    for (size_t token=0; token<batch; ++token)
                        weight_copy_vector(input+(kb*batch+token)*8,
                                        x.data+(base+token)*k+kb*8);
                }
            }
        }
        if (alloc_failed()) return false;
        edge_tensor_setcsr<0, L::wtype>();
        edge_tensor_setn(batch);
        if constexpr (nvfp4) edge_tensor_wsld_global(global_scale_bits);
#if NNEDGE_QUANT_PROFILE
        const size_t stream_start = edge_get_cycle();
#endif
        if constexpr (L::scaled) {
            if (full_slots)
                weight_start_scale_slots<L,L::groups_per_slot>(slots.data,ring,full_slots);
            else
                weight_start_scale_slots<L,1>(slots.data,ring,1);
        } else {
            edge_dma_start_strided_circular(slots.data,ring,L::slot_bytes,L::slot_bytes,
                count,0u,1u,L::slot_bytes,ring_slots);
            edge_tensor_wld_circular();
        }
        size_t tiles_left = (n/8)*(k/8);
        constexpr unsigned scale_mode = L::scaled ? EDGE_TENSOR_START_OPT_USE_SCALE : 0;
        // P is stable throughout the chunk; O changes only on the final K tile.
        edge_tensor_setpsum(psum);
        for (size_t ob=0; ob<n/8; ++ob) {
            edge_tensor_setout(psum);
            for (size_t kb=0; kb<k/8; ++kb) {
                if constexpr (L::groups_per_slot == 2) {
                    if (tail_group && full_slots && kb%L::tiles==0 &&
                        ob*(k/L::group_k)+kb/L::tiles == full_slots*2) {
                        // Drain both producer and consumer before changing the
                        // physical ring stride for the lone final quant group.
                        edge_tensor_sync();edge_dma_sync();
                        weight_start_scale_slots<L,1>(slots.data+full_slots*L::slot_bytes,ring,1);
                    }
                }
                edge_tensor_setin(xs+kb*(TensorSram ? rows : batch)*8);
                const bool last = kb+1 == k/8;
                if (last) edge_tensor_setout(ys+ob*(TensorSram ? rows : batch)*8);
                if (kb==0 && last)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_SINGLE_TILE>();
                else if (kb==0)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_FIRST_TILE>();
                else if (last)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_LAST_TILE>();
                else
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_PSUM_TILE>();
                if constexpr (!L::scaled) {
                    if (--tiles_left) edge_tensor_wld_circular();
                }
            }
        }
        edge_tensor_sync();
        edge_dma_sync();
#if NNEDGE_QUANT_PROFILE
        stream_cycles += edge_get_cycle() - stream_start;
        ++chunks;
#endif
        // Restore [token,N] only after all circular consumers have retired.
        if constexpr (TensorSram) {
            // The final tile already wrote the caller's working SRAM.
        } else if (batch==1) {
            if (ys != y.data+base*n) {
                if (output_dtcm) {
                    for (size_t ob=0;ob<n/8;++ob)
                        weight_copy_vector(y.data+base*n+ob*8,ys+ob*8);
                } else {
                    copy(Tensor<bfloat16_t>(Shape(n), y.data+base*n),
                         Tensor<bfloat16_t>(Shape(n), ys));
                }
            }
        } else {
            for (size_t token=0; token<batch; ++token) {
                if (!output_dtcm) {
                    edge_dma_start_strided(output+token*8, y.data+(base+token)*n,
                                          16u, batch*16, n/8);
                    edge_dma_sync();
                } else {
                    for (size_t ob=0; ob<n/8; ++ob)
                        weight_copy_vector(y.data+(base+token)*n+ob*8,
                                        output+(ob*batch+token)*8);
                }
            }
        }
    }
#if NNEDGE_QUANT_PROFILE
    const size_t op_cycles = edge_get_cycle() - op_start;
    printf("NNC_QUANT_PROFILE capacity=%lu chunks=%lu stream_cycles=%lu op_cycles=%lu scratch_bytes=%lu\n",
           (unsigned long)capacity, (unsigned long)chunks,
           (unsigned long)stream_cycles, (unsigned long)op_cycles,
           (unsigned long)(TensorSram ? weight_psum_bytes(capacity)+L::ring_bytes
                                     : weight_scratch_bytes<L>(capacity,k,n)));
#endif
    return !alloc_failed();
}
} // namespace detail

// Typed tensors count complete packs. Format and K grouping come from the
// element type, so callers cannot select a conflicting WType at the call site.
template <unsigned WType, unsigned SType, unsigned Tiles, unsigned Scales,unsigned Groups,
          unsigned MaxTokens = 64>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<WeightPack<WType, SType, Tiles, Scales,Groups>> slots)
{
    using Pack = WeightPack<WType, SType, Tiles, Scales,Groups>;
    static_assert(sizeof(Pack) == Pack::slot_bytes, "pack stride must match DMA layout");
    return detail::linear_weight_slots_bytes<Pack,MaxTokens>(y, x,
        Tensor<uint8_t>(Shape(numel(slots) * sizeof(Pack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}

// A named quant preset can select its layout for compiler-generated byte
// payloads without pretending the buffer contains constructed Pack objects.
template <typename Pack, unsigned MaxTokens = 64>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<uint8_t> slots)
{
    static_assert(sizeof(Pack) == Pack::slot_bytes, "pack stride must match DMA layout");
    return detail::linear_weight_slots_bytes<Pack,MaxTokens>(y, x, slots);
}

// Compatibility adapter for existing byte-oriented callers.
template <unsigned WType, unsigned MaxTokens = 64>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<uint8_t> slots)
{
    return linear_weight_slots<WeightPack<WType>,MaxTokens>(y, x, slots);
}
} // namespace nnedge::op
