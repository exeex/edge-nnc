"""BF22 P/O retains tiny tile contributions that BF16 PSUM loses."""
import torch
from nnc.fp4 import linear_fp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        q = torch.zeros(8,256,dtype=torch.bfloat16)
        q[:,::8] = 1
        self.q = torch.nn.Parameter(q,requires_grad=False)
        self.scale = torch.nn.Parameter(torch.ones(8,8,dtype=torch.bfloat16),requires_grad=False)
    def forward(self,x):
        return linear_fp4(x,self.q,self.scale)

def make_model_and_args():
    x = torch.zeros(1,256,dtype=torch.bfloat16)
    x[:,::8] = 2**-12
    x[:,0] = 1
    # Exact sum = 1+31*2^-12 -> BF16 1.0078125; per-tile BF16 stays at 1.
    return Model(), (x,)
