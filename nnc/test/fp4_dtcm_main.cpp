// Force public operands into DTCM to cover accelerator gather/scatter.
// Use any smoke_linear_fp4 model that fits the tensor arena.
#include "forward.hpp"
extern "C" void *memcpy(void *dst, const void *src, unsigned long n) {
    auto *d = static_cast<unsigned char *>(dst);
    auto *s = static_cast<const unsigned char *>(src);
    for (unsigned long i=0; i<n; ++i) d[i]=s[i];
    return dst;
}
extern "C" int main() {
    model::init();
    nnedge::Tensor<nnedge::bfloat16_t> x(model::mio::x.shape), y(model::mio::y.shape);
    nnedge::malloc_tensor(x);
    nnedge::malloc_tensor(y);
    if (nnedge::alloc_failed() || !nnedge::is_dtcm_addr(x.data) || !nnedge::is_dtcm_addr(y.data)) return 1;
    nnedge::copy(x, model::mio::x);
    if (!nnedge::op::linear_fp4(y, x, model::mio::ctx.q_scale_fp4_slots)) return 2;
    nnedge::copy(model::mio::y, y);
    return nnedge::alloc_failed() ? 3 : 0;
}
