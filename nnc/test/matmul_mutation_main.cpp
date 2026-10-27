// RHS is runtime data: modify it between two executions of the same image.
#define NNEDGE_MATMUL_PROFILE 1
#include "forward.hpp"
extern "C" int main()
{
    model::init();
    if (model::forward()) return 1;
    model::mio::rhs.data[0].bits ^= 0x8000u;
    edge_dcache_clean_range(model::mio::rhs.data,2);
    nnedge::reset_allocator();
    return model::forward();
}
