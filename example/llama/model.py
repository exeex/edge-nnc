"""Tiny Llama block composed from reusable NNC modules."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
import nnc
from nnc import Linear, RMSNorm, RoPE, KVCacheUpdate, Attention as TensorAttention


@dataclass(frozen=True)
class ModelArgs:
    dim: int = 64
    n_heads: int = 2
    n_kv_heads: int | None = None
    multiple_of: int = 64
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1.0e-5


def llama_hidden_dim(args: ModelArgs) -> int:
    hidden_dim = int(2 * (4 * args.dim) / 3)
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    return args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False))
        self.wk = Linear(nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False))
        self.wv = Linear(nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False))
        self.wo = Linear(nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False))
        self.rope = RoPE()
        self.kv_cache = KVCacheUpdate()
        self.attend = TensorAttention(self.n_heads)

    def forward(self, x: Tensor, start_pos: Tensor, mask_or_params: Tensor) -> Tensor:
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        xq, xk = self.rope(xq, xk, start_pos)
        keys, values = self.kv_cache(xk, xv, start_pos)
        attn = self.attend(xq, keys, values, mask_or_params)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = llama_hidden_dim(args)
        self.w1 = Linear(nn.Linear(args.dim, hidden_dim, bias=False))
        self.w2 = Linear(nn.Linear(hidden_dim, args.dim, bias=False))
        self.w3 = Linear(nn.Linear(args.dim, hidden_dim, bias=False))
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        w1 = self.w1(x)
        w3 = self.w3(x)
        return self.w2(self.silu(w1) * w3)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: Tensor, start_pos: Tensor, mask_or_params: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, mask_or_params)
        return h + self.feed_forward(self.ffn_norm(h))


def make_tiny_llama_block() -> TransformerBlock:
    torch.manual_seed(0)
    # nn.Linear constructors consume RNG; keep the original fixture weights.
    with torch.random.fork_rng(devices=[]):
        model = TransformerBlock(ModelArgs())
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            param.data.fill_(1.0)
        else:
            param.data.copy_(torch.randn_like(param) * 0.02)
    return model


def make_example_args() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.randn(32, 64, dtype=torch.float32),
        torch.tensor(0, dtype=torch.int64),
        torch.zeros(32, 32, dtype=torch.float32),
    )


def make_model_and_args() -> tuple[TransformerBlock, tuple[Tensor, Tensor, Tensor]]:
    return make_tiny_llama_block(), make_example_args()


if __name__ == "__main__":
    m = make_tiny_llama_block().eval()
    nnc.compile(m, make_example_args(), out_dir=Path(__file__).parent)
