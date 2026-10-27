// Same generated Q8_0 fixture, through the public typed WeightPack overload.
#include "init.hpp"
#include "intrinsic/edge_sim_console.hpp"

static_assert(nnedge::Q8_0WeightPack::stype == EDGE_TENSOR_STYPE_FP16);
static_assert(nnedge::Q8_0WeightPack::scale_bytes == 16);
static_assert(sizeof(nnedge::Q8_0WeightPack) == 272);

extern "C" int main()
{
    model::init();
    const auto x = model::mio::x;
    const auto y = model::mio::y;
    nnedge::Tensor<nnedge::Q8_0WeightPack> slots(
        nnedge::Shape(y.shape.dims[1]/8, x.shape.dims[1]/32),
        reinterpret_cast<nnedge::Q8_0WeightPack *>(
            const_cast<unsigned char *>(model::weight_begin)));
    const auto begin = edge_get_cycle();
    const bool ok = nnedge::op::linear_q8_0(y,x,slots);
    const auto cycles = edge_get_cycle()-begin;
    edge_sim_printf("NNC_PROFILE_BEGIN\n");
    edge_sim_printf("NNC_PROFILE\t0\tlinear_q8_0\tlinear_q8_0\t%llu\n",
                    (unsigned long long)cycles);
    edge_sim_printf("NNC_PROFILE_END\n");
    return ok ? 0 : 1;
}
