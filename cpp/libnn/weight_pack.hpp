#pragma once
#include "tensor.hpp"

namespace nnedge {
// One output-major slot: scales, Tiles consecutive 8x8 weight tiles, then
// padding to a 16-byte DMA beat. Shape dimensions count packs, not bytes.
// Scales=0 is the scale-free BF16 WLD layout (one tile per pack).
template <unsigned WType, unsigned SType = 0,
          unsigned Tiles = (WType == EDGE_TENSOR_WTYPE_BF16 ? 1 : 4),
          unsigned Scales = (WType == EDGE_TENSOR_WTYPE_BF16 ? 0 : 8)>
struct alignas(16) WeightPack {
    static_assert(WType >= 1 && WType <= 5, "unsupported weight type");
    static_assert(SType == 0, "only BF16 scales (stype=0) are supported");
    static_assert(Tiles >= 1 && Tiles <= 255, "tiles must be in [1,255]");
    static_assert(Scales == 0 || Scales == 1 || Scales == 8,
                  "scales must be 0 (BF16 WLD), 1 (broadcast), or 8 (per lane)");
    static_assert(Scales != 0 || (WType == EDGE_TENSOR_WTYPE_BF16 && Tiles == 1),
                  "scale-free packs require one BF16 tile");
    static constexpr unsigned wtype = WType, stype = SType;
    static constexpr unsigned tiles = Tiles, scales = Scales;
    static constexpr bool scaled = Scales != 0;
    static constexpr unsigned group_k = 8 * Tiles;
    static constexpr unsigned tile_bytes = WType == EDGE_TENSOR_WTYPE_BF16 ? 128 :
                                          WType == EDGE_TENSOR_WTYPE_FP4 ? 32 : 64;
    static constexpr unsigned scale_bytes = 2 * Scales;
    static constexpr unsigned slot_bytes = (scale_bytes + Tiles * tile_bytes + 15u) & ~15u;
    uint8_t bytes[slot_bytes];
};
} // namespace nnedge
