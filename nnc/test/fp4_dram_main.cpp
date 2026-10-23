// Exercise the public runtime with external input/output, bypassing the
// generated forward's usual DTCM allocation. Use smoke_linear_fp4.py.
#include "forward.hpp"
extern "C" void *memcpy(void *dst, const void *src, unsigned long n) {
    auto *d = static_cast<unsigned char *>(dst);
    auto *s = static_cast<const unsigned char *>(src);
    for (unsigned long i=0; i<n; ++i) d[i]=s[i];
    return dst;
}
extern "C" int main() {
    model::init();
    return nnedge::op::linear_fp4(model::mio::y, model::mio::x,
                                 model::mio::ctx.q_scale_fp4_slots) ? 0 : 1;
}
