"""Pointwise producers and consumers share Matmul's tiled SRAM storage."""
import torch
from nnc.test.smoke_matmul import nnedge_matmul
import torch.nn.functional as F

class Model(torch.nn.Module):
    def forward(self,x,rhs):
        # Runtime RHS is independently produced, and must remain row-major in
        # DRAM; LHS and the output pointwise chain use tensor SRAM order.
        return F.silu(nnedge_matmul(F.silu(x), F.silu(rhs), 1))

def make_model_and_args():
    x=((torch.arange(9*16).reshape(1,9,16)%13-6).float()/32).to(torch.bfloat16)
    rhs=((torch.arange(16*16).reshape(16,16)%11-5).float()/16).to(torch.bfloat16)
    return Model(),(x,rhs)
