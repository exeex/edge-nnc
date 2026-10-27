// Typed packed-weight runtime regression, independent scalar integer golden.
#include "libnn/ops.hpp"

using namespace nnedge;

static_assert(sizeof(WeightPack<EDGE_TENSOR_WTYPE_FP4>) == 144);
static_assert(sizeof(WeightPack<EDGE_TENSOR_WTYPE_BF16>) == 128);
static_assert(sizeof(WeightPack<EDGE_TENSOR_WTYPE_FP4,0,1,1>) == 48);
static_assert(sizeof(WeightPack<EDGE_TENSOR_WTYPE_FP8_E4M3FN,0,1,1>) == 80);
static_assert(WeightPack<EDGE_TENSOR_WTYPE_FP4,0,8>::group_k == 64);


static constexpr int weight(unsigned r, unsigned k)
{
    return int((r * 3 + k * 2) % 5) - 2;
}
static constexpr int input(unsigned row, unsigned k)
{
    return int((row + k) % 3) - 1;
}
template<unsigned W> static constexpr unsigned encode(int q)
{
    const unsigned m = q < 0 ? -q : q;
    if constexpr (W == EDGE_TENSOR_WTYPE_BF16)
        return (m == 2 ? 0x4000 : m == 1 ? 0x3f80 : 0) | (q < 0 ? 0x8000 : 0);
    if constexpr (W == EDGE_TENSOR_WTYPE_INT8) return unsigned(q) & 255;
    if constexpr (W == EDGE_TENSOR_WTYPE_FP4)
        return (m == 2 ? 4 : m == 1 ? 2 : 0) | (q < 0 ? 8 : 0);
    const unsigned one = W == EDGE_TENSOR_WTYPE_FP8_E4M3FN ? 0x38 : 0x3c;
    return (m == 2 ? 0x40 : m == 1 ? one : 0) | (q < 0 ? 0x80 : 0);
}

template<unsigned W, unsigned T, unsigned S> static int run()
{
    constexpr unsigned ST = W == EDGE_TENSOR_WTYPE_INT8 && T == 4 && S == 8
        ? EDGE_TENSOR_STYPE_FP16 : EDGE_TENSOR_STYPE_BF16;
    using Pack = WeightPack<W, ST, T, S>;
    constexpr unsigned rows = 9, n = 16, groups = 3, k = groups * T * 8;
    alignas(64) static Pack packs[2 * groups] = {};
    alignas(64) static nnedge::bfloat16_t x[rows * k], y[rows * n];
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned i = 0; i < k; ++i)
            x[row*k+i] = nnedge::bfloat16_t::from_float(input(row,i));
    for (unsigned ob = 0; ob < n/8; ++ob)
        for (unsigned g = 0; g < groups; ++g) {
            auto &p = packs[ob*groups+g];
            for (unsigned s = 0; s < S; ++s) {
                const unsigned mode = (g+s)%3;
                const unsigned bits = ST == EDGE_TENSOR_STYPE_FP16
                    ? (mode == 0 ? 0x3800 : mode == 1 ? 0x3c00 : 0x4000)
                    : (mode == 0 ? 0x3f00 : mode == 1 ? 0x3f80 : 0x4000);
                p.bytes[2*s] = bits;
                p.bytes[2*s+1] = bits >> 8;
            }
            for (unsigned t = 0; t < T; ++t)
                for (unsigned r = 0; r < 8; ++r)
                    for (unsigned i = 0; i < 8; ++i) {
                        const unsigned index = r*8+i;
                        const unsigned value = encode<W>(weight(ob*8+r, (g*T+t)*8+i));
                        const unsigned base = Pack::scale_bytes+t*Pack::tile_bytes;
                        if constexpr (W == EDGE_TENSOR_WTYPE_BF16) {
                            p.bytes[base+2*index] = value;
                            p.bytes[base+2*index+1] = value >> 8;
                        } else if constexpr (W == EDGE_TENSOR_WTYPE_FP4) {
                            const unsigned shift = 4*(index%2);
                            auto &byte = p.bytes[base+index/2];
                            byte = (byte & ~(15u << shift)) | (value << shift);
                        } else p.bytes[base+index] = value;
                    }
        }
    reset_allocator();
    Tensor<nnedge::bfloat16_t> tx(Shape(rows,k), x), ty(Shape(rows,n), y);
    Tensor<Pack> slots(Shape(n/8,groups), packs);
    // Reject byte counts masquerading as pack counts and incomplete K groups.
    if (op::linear_weight_slots(ty, tx, Tensor<Pack>(Shape(sizeof(packs)),packs))) return 1;
    if (op::linear_weight_slots(ty, Tensor<nnedge::bfloat16_t>(Shape(rows,k-1),x), slots)) return 2;
    Tensor<uint8_t> byte_slots(Shape(sizeof(packs)),reinterpret_cast<uint8_t *>(packs));
    bool ok;
    if constexpr (T == 4 && S == 8) {
        if constexpr (W == EDGE_TENSOR_WTYPE_FP4) ok = op::linear_fp4(ty,tx,byte_slots);
        else if constexpr (W == EDGE_TENSOR_WTYPE_INT8) ok = op::linear_q8_0(ty,tx,byte_slots);
        else if constexpr (W == EDGE_TENSOR_WTYPE_FP8_E4M3FN) ok = op::linear_fp8_e4m3fn(ty,tx,byte_slots);
        else if constexpr (W == EDGE_TENSOR_WTYPE_FP8_E5M2) ok = op::linear_fp8_e5m2(ty,tx,byte_slots);
        else ok = op::linear_weight_slots(ty,tx,slots);
    } else ok = op::linear_weight_slots(ty,tx,slots);
    if (!ok) return 3;
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned r = 0; r < n; ++r) {
            int twice_expected = 0;
            for (unsigned g = 0; g < groups; ++g) {
                int dot = 0;
                for (unsigned i = g*T*8; i < (g+1)*T*8; ++i)
                    dot += input(row,i)*weight(r,i);
                const unsigned mode = (g+(S == 8 ? r%8 : 0))%3;
                twice_expected += dot*(S == 0 ? 2 : mode == 0 ? 1 : mode == 1 ? 2 : 4);
            }
            const auto expected = nnedge::bfloat16_t::from_float(twice_expected*.5f).bits;
            if (y[row*n+r].bits != expected) {
                edge_sim_printf("WEIGHT PACK FAIL W=%u T=%u S=%u row=%u lane=%u got=%x expected=%x\n",
                                W,T,S,row,r,y[row*n+r].bits,expected);
                return 4;
            }
        }
    edge_sim_printf("WEIGHT PACK PASS W=%u T=%u S=%u\n",W,T,S);
    return 0;
}
template<unsigned W> static int formats()
{
    return run<W,4,8>() || run<W,2,1>() || run<W,8,8>();
}
extern "C" int main()
{
    if (formats<EDGE_TENSOR_WTYPE_FP4>() || formats<EDGE_TENSOR_WTYPE_FP8_E4M3FN>() ||
        formats<EDGE_TENSOR_WTYPE_FP8_E5M2>() || formats<EDGE_TENSOR_WTYPE_INT8>() ||
        formats<EDGE_TENSOR_WTYPE_BF16>() || run<EDGE_TENSOR_WTYPE_BF16,1,0>()) return 1;
    edge_sim_printf("WEIGHT PACK ALL PASS cases=16\n");
    return 0;
}
