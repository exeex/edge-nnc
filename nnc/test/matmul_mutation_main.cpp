// RHS is runtime data: modify it between two executions of the same image.
#define NNEDGE_MATMUL_PROFILE 1
#include "forward.hpp"
extern "C" int main()
{
    model::init();
    if (model::forward()) return 1;
    nnedge::scalar_tensor_acquire(model::mio::rhs);
    model::mio::rhs.data[0].bits ^= 0x8000u;
    nnedge::scalar_tensor_publish(model::mio::rhs);
    nnedge::reset_allocator();
    return model::forward();
}
