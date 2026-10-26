#pragma once
#include "tensor.hpp"

namespace nnedge::op {
// BF16 uses ordinary scale-free WLD tiles; quantized types use K32 WSLD slots.
template <unsigned WType> struct WeightSlotLayout {
    static_assert(WType >= 1 && WType <= 5, "unsupported weight type");
    static constexpr bool scaled = WType != EDGE_TENSOR_WTYPE_BF16;
    static constexpr unsigned group_k = scaled ? 32 : 8;
    static constexpr unsigned tiles = scaled ? 4 : 1, scales = 8;
    static constexpr unsigned tile_bytes = WType == EDGE_TENSOR_WTYPE_BF16 ? 128 :
                                          WType == EDGE_TENSOR_WTYPE_FP4 ? 32 : 64;
    static constexpr unsigned slot_bytes = (scaled ? 16 : 0) + tiles*tile_bytes;
    static constexpr unsigned ring_slots = 4, wtype = WType, stype = 0;
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

template <unsigned WType>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                       Tensor<uint8_t> slots)
{
    using L = WeightSlotLayout<WType>;
    if (x.shape.rank != 2 || y.shape.rank != 2) return false;
    const size_t rows = x.shape.dims[0], k = x.shape.dims[1], n = y.shape.dims[1];
    const size_t count = (n/8)*(k/L::group_k);
    if (!rows || y.shape.dims[0] != rows || !k || !n || k%L::group_k || n%8 ||
        numel(slots) != count*L::slot_bytes || !x.data || !y.data || !slots.data)
        return false;
    // As in BF16 Matmul, one weight tile serves an entire token chunk.
    // Blocked I/O is private; the public tensors stay contiguous [M,K]/[M,N].
    size_t capacity = rows < 64u ? rows : 64u;
    while (capacity && align_bytes(capacity*k*2) + align_bytes(capacity*n*2)
           + weight_psum_bytes(capacity) + L::ring_slots*L::slot_bytes > kDtcmOpScratchBytes)
        --capacity;
    if (!capacity) return false;
    auto *scratch = dtcm_op_scratch<unsigned char>();
    auto *input = reinterpret_cast<bfloat16_t *>(scratch);
    auto *output = reinterpret_cast<bfloat16_t *>(scratch+align_bytes(capacity*k*2));
    auto *psum = reinterpret_cast<unsigned char *>(output)+align_bytes(capacity*n*2);
    auto *ring = psum+weight_psum_bytes(capacity);
    const bool input_dtcm = is_dtcm_addr(x.data), output_dtcm = is_dtcm_addr(y.data);
    if (!input_dtcm) edge_dcache_clean_range(x.data, rows*k*2);
    if (!output_dtcm) edge_dcache_clean_range(y.data, rows*n*2);
    edge_dcache_clean_range(slots.data, count*L::slot_bytes);

    for (size_t base=0; base<rows; base+=capacity) {
        const size_t batch = rows-base < capacity ? rows-base : capacity;
        auto *xs = input;
        auto *ys = output;
        if (batch==1) {
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
        edge_dma_start_strided_circular(slots.data, ring,
            L::slot_bytes, L::slot_bytes, count, 0u, 1u,
            L::slot_bytes, L::ring_slots);
        if constexpr (L::scaled)
            edge_tensor_wsld_circular<L::transpose,L::tiles,L::wtype,L::scales,L::stype>();
        else
            edge_tensor_wld_circular();
        size_t tiles_left = (n/8)*(k/8);
        constexpr unsigned scale_mode = L::scaled ? EDGE_TENSOR_START_OPT_USE_SCALE : 0;
        // P is stable throughout the chunk; O changes only on the final K tile.
        edge_tensor_setpsum(psum);
        for (size_t ob=0; ob<n/8; ++ob) {
            edge_tensor_setout(psum);
            for (size_t kb=0; kb<k/8; ++kb) {
                edge_tensor_setin(xs+kb*batch*8);
                const bool last = kb+1 == k/8;
                if (last) edge_tensor_setout(ys+ob*batch*8);
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
        // Restore [token,N] only after all circular consumers have retired.
        if (batch==1) {
            if (ys != y.data+base*n)
                copy(Tensor<bfloat16_t>(Shape(n), y.data+base*n),
                     Tensor<bfloat16_t>(Shape(n), ys));
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
    if (!output_dtcm) edge_dcache_invalidate_range(y.data, rows*n*2);
    return !alloc_failed();
}
} // namespace nnedge::op
