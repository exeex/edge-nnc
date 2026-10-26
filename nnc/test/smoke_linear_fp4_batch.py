"""FP4 64-token scheduling and contiguous layout regression."""
import torch
from nnc.fp4 import quantize, linear_fp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = ((torch.arange(64*64).reshape(64,64)*7).remainder(41)-20).float()/16
        w *= torch.arange(1,64+1).reshape(64,1)/32
        w[0,:32] = 0
        q,s = quantize(w)
        self.q = torch.nn.Parameter(q,requires_grad=False)
        self.scale = torch.nn.Parameter(s,requires_grad=False)
    def forward(self,x):
        return linear_fp4(x,self.q,self.scale)

def make_model_and_args():
    x = ((torch.arange(64*64).remainder(13)-6).float()/32).reshape(64,64).to(torch.bfloat16)
    return Model(), (x,)
