// Capacity fallback and caller-owned SRAM extents, using the same quant model.
#define NNEDGE_QUANT_PROFILE 1
#include "bench_config.hpp"

extern "C" int main()
{
    model::init();
    using namespace nnedge;
    // Even all 128 KiB cannot stage 128 tokens of a 4096x4096 Linear.
    if (op::detail::weight_token_capacity<BenchmarkPack,128>(128,4096,4096,kDtcmBytes) != 7)
        return 10;
    if (op::detail::weight_token_capacity<BenchmarkPack,128>(128,64,64,0) != 0)
        return 11;

    Tensor<nnedge::bfloat16_t> x(model::mio::x.shape), y(model::mio::y.shape);
    malloc_tensor(x);
    malloc_tensor(y);
    Tensor<uint8_t> reserved(Shape(64u*1024u));
    malloc_tensor(reserved);
    if (alloc_failed() || dtcm_cursor() != kDtcmTensorArenaBytes) return 12;
    const TensorTileView sx(x), sy(y);
    // Working SRAM must be explicit; reject DRAM views and SRAM/SRAM load.
    if (benchmark_linear(TensorTileView(model::mio::y),TensorTileView(model::mio::x))) return 18;
    if (op::load_tensor_sram(sx,sx)) return 19;
    // Probe every cache line of the occupied arena tail for scratch overlap.
    for (size_t i=0; i<numel(reserved); i+=64) reserved.data[i]=0xa5;
    if (!op::load_tensor_sram(sx,TensorTileView(model::mio::x))) return 20;
    if (!benchmark_linear(sy,sx)) return 13;
    if (!op::store_tensor_sram(TensorTileView(model::mio::y),sy)) return 21;

    // A caller-owned input sits above the allocator cursor, in the region
    // normally reserved for op scratch. The 128-token variant must avoid it.
    Tensor<nnedge::bfloat16_t> manual(x.shape,
        reinterpret_cast<nnedge::bfloat16_t *>(dtcm_base()+108u*1024u));
    const TensorTileView sm(manual);
    if (!op::load_tensor_sram(sm,TensorTileView(model::mio::x))) return 22;
    if (!benchmark_linear(sy,sm)) return 14;
    for (size_t i=0; i<numel(reserved); i+=64)
        if (reserved.data[i] != 0xa5) return 15;
    // Both capacity-limited executions must produce the same public bytes.
    for (size_t i=0; i<numel(y); ++i)
        if (y.data[i].bits != model::mio::y.data[i].bits) return 16;
    if (!op::store_tensor_sram(TensorTileView(model::mio::y),sy)) return 23;
    return alloc_failed() ? 17 : 0;
}
