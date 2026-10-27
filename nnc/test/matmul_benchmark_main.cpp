#define NNEDGE_MATMUL_PROFILE 1
#include "forward.hpp"
extern "C" int main()
{
    model::init();
    return model::forward();
}
