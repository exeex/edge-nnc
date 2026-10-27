#pragma once
#include "tensor.hpp"

namespace nnedge {
// A physical slot contains Groups output-major quant groups: scales followed
// by Tiles consecutive 8x8 weight tiles. Only the slot tail is padded to
// a 16-byte DMA beat. Shape dimensions count packs, not bytes.
// Scales=0 is the scale-free BF16 WLD layout (one tile per pack).
template <unsigned WType, unsigned SType = 0,
          unsigned Tiles = (WType == EDGE_TENSOR_WTYPE_BF16 ? 1 : 4),
          unsigned Scales = (WType == EDGE_TENSOR_WTYPE_BF16 ? 0 : 8),
          unsigned Groups = 1>
struct alignas(16) WeightPack {
    static_assert(WType >= 1 && WType <= 5, "unsupported weight type");
    static_assert(SType <= EDGE_TENSOR_STYPE_FP16, "unsupported scale encoding");
    static_assert(Tiles >= 1 && Tiles <= 255, "tiles must be in [1,255]");
    static_assert(Scales == 0 || Scales == 1 || Scales == 8,
                  "scales must be 0 (BF16 WLD), 1 (broadcast), or 8 (per lane)");
    static_assert(Scales != 0 || (WType == EDGE_TENSOR_WTYPE_BF16 && Tiles == 1),
                  "scale-free packs require one BF16 tile");
    static_assert(SType == 0 || Scales != 0, "non-BF16 scale encoding requires scales");
    static_assert(Groups == 1 || (Groups == 2 && (SType == EDGE_TENSOR_STYPE_MXFP4 || SType == EDGE_TENSOR_STYPE_NVFP4) && Scales == 8),
                  "dense pair packs require eight byte scales per quant group");
    static constexpr unsigned wtype = WType, stype = SType;
    static constexpr unsigned tiles = Tiles, scales = Scales;
    static constexpr unsigned groups_per_slot = Groups;
    static constexpr bool scaled = Scales != 0;
    static constexpr unsigned group_k = 8 * Tiles;
    static constexpr unsigned tile_bytes = WType == EDGE_TENSOR_WTYPE_BF16 ? 128 :
                                          WType == EDGE_TENSOR_WTYPE_FP4 ? 32 : 64;
    static constexpr unsigned scale_bytes = ((SType == EDGE_TENSOR_STYPE_BF16 || SType == EDGE_TENSOR_STYPE_FP16) ? 2 : 1) * Scales;
    static constexpr unsigned payload_bytes = scale_bytes + Tiles * tile_bytes;
    static constexpr unsigned slot_bytes = (Groups * payload_bytes + 15u) & ~15u;
    static_assert(Groups == 1 || Groups * payload_bytes == slot_bytes,
                  "dense transport packs must have no interior or per-group padding");
    using circular_layout = EdgeCircularDmaLayout<slot_bytes,Groups == 2 ? 8 : 4>;
    static constexpr unsigned ring_slots = circular_layout::slot_count;
    static constexpr unsigned ring_bytes = circular_layout::ring_bytes;
    uint8_t bytes[slot_bytes];
};
} // namespace nnedge
