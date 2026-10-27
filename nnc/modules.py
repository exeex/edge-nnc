"""Inference modules that replace nn.Linear with an explicit NNC quant op."""
import torch
from nnc.fp4 import linear_fp4
from nnc.quant import linear_q8_0, linear_fp8_e4m3fn, linear_fp8_e5m2
from nnc.microscale import linear_mxfp4, linear_nvfp4


class _QuantLinear(torch.nn.Module):
    group_k = 32
    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        if not isinstance(linear, torch.nn.Linear):
            raise TypeError('expected torch.nn.Linear')
        if min(linear.in_features, linear.out_features) <= 0 or linear.in_features % self.group_k or linear.out_features % 8:
            raise ValueError(f'quant Linear requires in_features%{self.group_k}=0 and out_features%8=0')
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = torch.nn.Parameter(linear.weight.detach().to(torch.bfloat16).clone(), requires_grad=False)
        self.bias = (torch.nn.Parameter(linear.bias.detach().to(torch.bfloat16).clone(), requires_grad=False)
                     if linear.bias is not None else None)

    def forward(self, x):
        y = self.op(x, self.weight)
        return y + self.bias if self.bias is not None else y

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class LinearFp4(_QuantLinear):
    op = staticmethod(linear_fp4)


class LinearQ8_0(_QuantLinear):
    op = staticmethod(linear_q8_0)


class LinearFp8E4M3FN(_QuantLinear):
    op = staticmethod(linear_fp8_e4m3fn)


class LinearFp8E5M2(_QuantLinear):
    op = staticmethod(linear_fp8_e5m2)


class LinearMxFp4(_QuantLinear):
    op = staticmethod(linear_mxfp4)


class LinearNvFp4(_QuantLinear):
    group_k = 16
    op = staticmethod(linear_nvfp4)
