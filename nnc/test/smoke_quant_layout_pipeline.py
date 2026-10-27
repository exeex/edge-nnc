"""A row-reduction producer transfers into the quant kernel's SRAM layout."""
import torch
import nnc
from nnc.ops import rms_norm
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(32,dtype=torch.bfloat16))
        self.linear=nnc.LinearQ8_0(torch.nn.Linear(32,16,bias=False))
    def forward(self,x):
        return self.linear(rms_norm(x,self.weight,1e-5))
def make_model_and_args():
    torch.manual_seed(0)
    return Model().eval(), (torch.randn(9,32,dtype=torch.bfloat16)/16,)
