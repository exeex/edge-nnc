// Scalar/DMA ownership boundaries, using integer payloads (no soft-float).
#include "libnn/tensor.hpp"
using nnedge::Tensor;
using nnedge::Shape;
alignas(64) static uint16_t a[64], b[64], c[64];

static void produce(Tensor<uint16_t> dst, unsigned seed)
{
    nnedge::scalar_tensor_acquire(dst);
    for (unsigned i=0; i<64; ++i) dst.data[i]=seed+i;
    nnedge::scalar_tensor_publish(dst);
}
static bool check(Tensor<uint16_t> src, unsigned seed, bool all_zero=false)
{
    nnedge::scalar_tensor_acquire(src);
    for (unsigned i=0; i<64; ++i)
        if (src.data[i] != (all_zero ? 0 : seed+i)) return false;
    return true;
}
extern "C" int main()
{
    nnedge::reset_allocator();
    Tensor<uint16_t> ta(Shape(64),a), tb(Shape(64),b), tc(Shape(64),c);
    Tensor<uint16_t> sram(Shape(64));
    nnedge::malloc_tensor(sram);
    // Publish scalar-initialized storage before transferring ownership to DMA.
    produce(tb,0);
    produce(tc,0);
    produce(ta,0x1234);
    nnedge::copy(sram,ta); nnedge::copy(tb,sram);
    if (!check(tb,0x1234)) return 1;
    // tb is now cached by the CPU; DMA replaces it without invalidation.
    produce(ta,0x4321);
    nnedge::copy(sram,ta); nnedge::copy(tb,sram);
    if (!check(tb,0x4321)) return 2;
    // Scalar zero must publish its result before the next DMA consumer.
    nnedge::zero(ta);
    nnedge::copy(sram,ta); nnedge::copy(tb,sram);
    if (!check(tb,0,true)) return 3;
    // DRAM-to-DRAM copy is a scalar producer and publishes internally.
    produce(ta,0x2468);
    nnedge::copy(sram,ta); nnedge::copy(tb,sram);
    nnedge::copy(tc,tb);
    nnedge::copy(sram,tc); nnedge::copy(tb,sram);
    if (!check(tb,0x2468)) return 4;
    printf("CACHE OWNERSHIP PASS scalar_publish DMA scalar_acquire zero scalar_copy\n");
    return nnedge::alloc_failed() ? 5 : 0;
}
