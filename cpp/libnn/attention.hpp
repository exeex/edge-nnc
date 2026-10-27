#pragma once

#include "matmul.hpp"
#include "softmax.hpp"
#include "tensor_layout.hpp"

namespace nnedge::op {

template <typename DType>
inline void attention(Tensor<DType> y, Tensor<DType> xq, Tensor<DType> keys,
                      Tensor<DType> values, Tensor<DType> mask_or_params,
                      Tensor<DType> eye_weight, size_t head_count)
{
    (void)mask_or_params;

    Tensor<DType> score(Shape{head_count, xq.shape.dims[0], keys.shape.dims[0]});
    Tensor<DType> prob(Shape{head_count, xq.shape.dims[0], keys.shape.dims[0]});
    malloc_tensor_dram(score);
    malloc_tensor_dram(prob);

    Tensor<DType> q_tiles(xq.shape), score_tiles(score.shape), prob_tiles(prob.shape), y_tiles(y.shape);
    malloc_tensor_dram(q_tiles);
    malloc_tensor_dram(score_tiles);
    malloc_tensor_dram(prob_tiles);
    malloc_tensor_dram(y_tiles);
    if (!convert_tensor_layout(q_tiles,xq,true)) { alloc_failed()=true; return; }
    matmul_transpose(score_tiles, q_tiles, keys, head_count);
    if (!convert_tensor_layout(score,score_tiles,false)) { alloc_failed()=true; return; }
    edge_sim_printf("NNC_PHASE attention_scores\n");
    softmax(prob, score, eye_weight, head_count);
    edge_sim_printf("NNC_PHASE attention_softmax\n");
    if (!convert_tensor_layout(prob_tiles,prob,true)) { alloc_failed()=true; return; }
    matmul(y_tiles, prob_tiles, values, head_count);
    if (!convert_tensor_layout(y,y_tiles,false)) { alloc_failed()=true; return; }
    edge_sim_printf("NNC_PHASE attention_values\n");

    free_tensor(y_tiles);
    free_tensor(prob_tiles);
    free_tensor(score_tiles);
    free_tensor(q_tiles);
    free_tensor(prob);
    free_tensor(score);
}

template <typename DType>
inline void attention(Tensor<DType> y, Tensor<DType> xq, Tensor<DType> keys,
                      Tensor<DType> values, Tensor<DType> mask_or_params,
                      Tensor<DType> eye_weight)
{
    attention(y, xq, keys, values, mask_or_params, eye_weight, 1u);
}

} // namespace nnedge::op
