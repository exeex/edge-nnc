#pragma once
#include "weight_pack.hpp"

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
template <typename Pack,unsigned Groups>
inline void weight_start_scale_slots(const uint8_t *source, unsigned char *ring, size_t count)
{
    constexpr size_t stride = (Pack::payload_bytes*Groups+15u)&~15u;
    using Ring = EdgeCircularDmaLayout<stride,Pack::ring_slots>;
    edge_dma_start_strided_circular(source,ring,stride,stride,count,0u,1u,stride,Ring::slot_count);
    edge_tensor_wsld_circular<false,Pack::tiles,Pack::wtype,Pack::scales,Pack::stype,Groups>();
}

// The bare-metal compiler uses a soft-float ABI. Explicit scalar FP32 ISA
// keeps this epilogue independent of compiler-rt __mulsf3 support.
inline uint32_t weight_global_scale_bits(uint32_t value, uint32_t scale)
{
    uint32_t result;
    __asm__ volatile(
        ".option push\n.option arch, +f\n"
        "fmv.w.x ft0, %1\nfmv.w.x ft1, %2\n"
        "fmul.s ft0, ft0, ft1, rne\nfmv.x.w %0, ft0\n.option pop"
        : "=r"(result) : "r"(value), "r"(scale) : "ft0", "ft1");
    return result;
}

template <typename Pack>
inline bool linear_weight_slots_bytes(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                       Tensor<uint8_t> slots, uint32_t global_scale_bits = 0x3f800000u)
{
    using L = Pack;
    constexpr bool global_epilogue = L::stype == EDGE_TENSOR_STYPE_NVFP4;
    constexpr size_t output_bytes = global_epilogue ? 4u : 2u;
    if (global_epilogue && (!global_scale_bits || global_scale_bits >= 0x7f800000u))
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
    // As in BF16 Matmul, one weight tile serves an entire token chunk.
    // Blocked I/O is private; the public tensors stay contiguous [M,K]/[M,N].
    size_t capacity = rows < 64u ? rows : 64u;
    while (capacity && align_bytes(capacity*k*2) + align_bytes(capacity*n*output_bytes)
           + weight_psum_bytes(capacity) + ring_slots*L::slot_bytes > kDtcmOpScratchBytes)
        --capacity;
    if (!capacity) return false;
    auto *scratch = dtcm_op_scratch<unsigned char>();
    auto *input = reinterpret_cast<bfloat16_t *>(scratch);
    auto *output = reinterpret_cast<bfloat16_t *>(scratch+align_bytes(capacity*k*2));
    auto *psum = reinterpret_cast<unsigned char *>(output)+align_bytes(capacity*n*output_bytes);
    auto *ring = psum+weight_psum_bytes(capacity);
    const bool input_dtcm = is_dtcm_addr(x.data), output_dtcm = is_dtcm_addr(y.data);
    if (!input_dtcm) edge_dcache_clean_range(x.data, rows*k*2);
    if (!output_dtcm) edge_dcache_clean_range(y.data, rows*n*2);
    edge_dcache_clean_range(slots.data, stored_bytes);

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
            if (output_dtcm && !global_epilogue) ys=y.data+base*n;
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
                edge_tensor_setin(xs+kb*batch*8);
                const bool last = kb+1 == k/8;
                if (last) edge_tensor_setout(global_epilogue ?
                    reinterpret_cast<void *>(reinterpret_cast<float *>(output)+ob*batch*8) :
                    reinterpret_cast<void *>(ys+ob*batch*8));
                if (kb==0 && last)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_SINGLE_TILE |
                        (global_epilogue ? EDGE_TENSOR_START_OPT_TILE_OUTPUT_FP32 : 0)>();
                else if (kb==0)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_FIRST_TILE>();
                else if (last)
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_LAST_TILE |
                        (global_epilogue ? EDGE_TENSOR_START_OPT_TILE_OUTPUT_FP32 : 0)>();
                else
                    edge_tensor_start<scale_mode | EDGE_TENSOR_START_PSUM_TILE>();
                if constexpr (!L::scaled) {
                    if (--tiles_left) edge_tensor_wld_circular();
                }
            }
        }
        edge_tensor_sync();
        edge_dma_sync();
        if constexpr (global_epilogue) {
            // Preserve BF22 output until the single FP32 global scale has been
            // applied. Compact forward in-place before the existing scatter.
            auto *wide = reinterpret_cast<volatile uint32_t *>(output);
            for (size_t i=0;i<batch*n;++i) {
                uint32_t bits = weight_global_scale_bits(wide[i],global_scale_bits);
                if ((bits & 0x7fffffffu) > 0x7f800000u) bits = 0x7fc00000u;
                else bits += 0x7fffu + ((bits >> 16) & 1u);
                output[i] = bfloat16_t::from_bits(static_cast<uint16_t>(bits >> 16));
            }
        }
        // Restore [token,N] only after all circular consumers have retired.
        if (batch==1) {
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
    if (!output_dtcm) edge_dcache_invalidate_range(y.data, rows*n*2);
    return !alloc_failed();
}
} // namespace detail

// Typed tensors count complete packs. Format and K grouping come from the
// element type, so callers cannot select a conflicting WType at the call site.
template <unsigned WType, unsigned SType, unsigned Tiles, unsigned Scales,unsigned Groups>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<WeightPack<WType, SType, Tiles, Scales,Groups>> slots)
{
    using Pack = WeightPack<WType, SType, Tiles, Scales,Groups>;
    static_assert(sizeof(Pack) == Pack::slot_bytes, "pack stride must match DMA layout");
    return detail::linear_weight_slots_bytes<Pack>(y, x,
        Tensor<uint8_t>(Shape(numel(slots) * sizeof(Pack)),
                        reinterpret_cast<uint8_t *>(slots.data)));
}

// A named quant preset can select its layout for compiler-generated byte
// payloads without pretending the buffer contains constructed Pack objects.
template <typename Pack>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<uint8_t> slots)
{
    static_assert(sizeof(Pack) == Pack::slot_bytes, "pack stride must match DMA layout");
    return detail::linear_weight_slots_bytes<Pack>(y, x, slots);
}

// Compatibility adapter for existing byte-oriented callers.
template <unsigned WType>
inline bool linear_weight_slots(Tensor<bfloat16_t> y, Tensor<bfloat16_t> x,
                               Tensor<uint8_t> slots)
{
    return linear_weight_slots<WeightPack<WType>>(y, x, slots);
}
} // namespace nnedge::op
