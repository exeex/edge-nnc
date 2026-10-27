"""Reusable custom-op boundaries for the current E3 transformer semantics.

Attention is unscaled/unmasked, and KV update is the existing prefill
pass-through. These functions preserve the semantics of the tiny Llama model.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.library.custom_op("nnedge::rms_norm", mutates_args=())
def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return ((x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * weight).type_as(x)


@rms_norm.register_fake
def _(x, weight, eps):
    del weight, eps
    return torch.empty_like(x)


def rope_reference(x: Tensor, start_pos: Tensor) -> Tensor:
    tokens, dim = x.shape
    pair = torch.arange(0, dim, 2, dtype=torch.float32, device=x.device)
    theta = 1.0 / (10000.0 ** (pair / dim))
    pos = torch.arange(tokens, dtype=torch.float32, device=x.device) + start_pos.to(torch.float32)
    angles = pos[:, None] * theta[None, :]
    cos = torch.cos(angles).type_as(x)
    sin = torch.sin(angles).type_as(x)
    even = x[:, 0::2]
    odd = x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = even * sin + odd * cos
    return out


@torch.library.custom_op("nnedge::rope", mutates_args=())
def rope(xq: Tensor, xk: Tensor, start_pos: Tensor) -> tuple[Tensor, Tensor]:
    return rope_reference(xq, start_pos), rope_reference(xk, start_pos)


@rope.register_fake
def _(xq, xk, start_pos):
    del start_pos
    return torch.empty_like(xq), torch.empty_like(xk)


@torch.library.custom_op("nnedge::kv_cache_update", mutates_args=())
def kv_cache_update(xk: Tensor, xv: Tensor, start_pos: Tensor) -> tuple[Tensor, Tensor]:
    del start_pos
    return xk.clone(), xv.clone()


@kv_cache_update.register_fake
def _(xk, xv, start_pos):
    del start_pos
    return torch.empty_like(xk), torch.empty_like(xv)


@torch.library.custom_op("nnedge::attention", mutates_args=())
def attention(xq: Tensor, keys: Tensor, values: Tensor,
                     mask_or_params: Tensor, head_count: int) -> Tensor:
    del mask_or_params
    query_len, width = xq.shape
    key_len = keys.shape[0]
    head_dim = width // head_count
    q_heads = xq.view(query_len, head_count, head_dim).transpose(0, 1)
    k_heads = keys.view(key_len, head_count, head_dim).transpose(0, 1)
    v_heads = values.view(key_len, head_count, head_dim).transpose(0, 1)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    context = torch.matmul(scores, v_heads)
    return context.transpose(0, 1).contiguous().view(query_len, width)


@attention.register_fake
def _(xq, keys, values, mask_or_params, head_count):
    del keys, mask_or_params, head_count
    return torch.empty_strided(xq.shape, xq.stride(), dtype=values.dtype, device=values.device)


@torch.library.custom_op("nnedge::linear", mutates_args=())
def linear(x: Tensor, weight: Tensor) -> Tensor:
    return F.linear(x, weight)


@linear.register_fake
def _(x, weight):
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)
