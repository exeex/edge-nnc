// Verify the NNC-generated SRAM path as well as the direct benchmark call.
#define NNEDGE_QUANT_PROFILE 1
#include "forward.hpp"
extern "C" int main()
{
    model::init();
    return model::forward();
}
