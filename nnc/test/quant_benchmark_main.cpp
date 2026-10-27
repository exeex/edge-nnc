// NNC preplans IO storage: ordinary DMA into SRAM, compute, ordinary DMA out.
// bench_config.hpp selects the quant entry point and compile-time token ceiling.
#define NNEDGE_QUANT_PROFILE 1
#include "bench_config.hpp"

extern "C" int main()
{
    model::init();
    using namespace nnedge;
    Tensor<nnedge::bfloat16_t> x(model::mio::x.shape), y(model::mio::y.shape);
    malloc_tensor(x);
    malloc_tensor(y);
    if (alloc_failed() || !is_dtcm_addr(x.data) || !is_dtcm_addr(y.data)) return 1;
    const TensorTileView sx(x), sy(y);
    const size_t input_begin = edge_get_cycle();
    if (!op::load_tensor_sram(sx,TensorTileView(model::mio::x))) return 2;
    const size_t input_cycles = edge_get_cycle()-input_begin;
    if (!benchmark_linear(sy,sx)) return 3;
    // The same transfer boundary as the reference, outside kernel timing.
    const size_t output_begin = edge_get_cycle();
    if (!op::store_tensor_sram(TensorTileView(model::mio::y),sy)) return 4;
    const size_t output_cycles = edge_get_cycle()-output_begin;
    printf("NNC_QUANT_TRANSFER input_cycles=%lu output_cycles=%lu\n",
           (unsigned long)input_cycles,(unsigned long)output_cycles);
    return alloc_failed() ? 3 : 0;
}
