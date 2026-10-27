"""Reusable modules over the E3 transformer custom-op boundaries."""
import torch
from nnc import ops


class Linear(torch.nn.Module):
    """Wrap nn.Linear without changing its parameters; export nnedge::linear."""
    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError('expected torch.nn.Linear')
        if min(linear.in_features, linear.out_features) <= 0 or linear.in_features % 8 or linear.out_features % 8:
            raise ValueError('BF16 Linear requires positive feature counts divisible by 8')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x):
        y = ops.linear(x, self.weight)
        return y + self.bias if self.bias is not None else y


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return ops.rms_norm(x, self.weight, self.eps)


class RoPE(torch.nn.Module):
    def forward(self, xq, xk, start_pos):
        return ops.rope(xq, xk, start_pos)


class KVCacheUpdate(torch.nn.Module):
    """Current prefill pass-through, not a persistent decoding cache."""
    def forward(self, xk, xv, start_pos):
        return ops.kv_cache_update(xk, xv, start_pos)


class Attention(torch.nn.Module):
    """Current unscaled/unmasked attention with an explicit head count."""
    def __init__(self, head_count: int = 1):
        super().__init__()
        if head_count <= 0:
            raise ValueError('head_count must be positive')
        self.head_count = head_count

    def forward(self, xq, keys, values, mask_or_params):
        return ops.attention(xq, keys, values, mask_or_params, self.head_count)
