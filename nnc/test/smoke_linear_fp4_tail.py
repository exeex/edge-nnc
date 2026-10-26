"""FP4 9-token scheduling and contiguous layout regression."""
import torch
from nnc.fp4 import quantize, linear_fp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = ((torch.arange(24*128).reshape(24,128)*7).remainder(41)-20).float()/16
        w *= torch.arange(1,24+1).reshape(24,1)/32
        w[0,:32] = 0
        q,s = quantize(w)
        self.q = torch.nn.Parameter(q,requires_grad=False)
        self.scale = torch.nn.Parameter(s,requires_grad=False)
    def forward(self,x):
        return linear_fp4(x,self.q,self.scale)

def make_model_and_args():
    x = ((torch.arange(9*128).remainder(13)-6).float()/32).reshape(9,128).to(torch.bfloat16)
    return Model(), (x,)
