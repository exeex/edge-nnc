"""One FP4 slot with distinct signed lanes and exact binary scales."""
import torch
from nnc.fp4 import linear_fp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        q = torch.zeros(8,32,dtype=torch.bfloat16)
        for lane in range(8):
            q[lane,lane] = -1 if lane%2 else 1
        self.q = torch.nn.Parameter(q,requires_grad=False)
        self.scale = torch.nn.Parameter(torch.tensor([.5,1,2,4,.5,1,2,4],dtype=torch.bfloat16).reshape(8,1),requires_grad=False)
    def forward(self,x):
        return linear_fp4(x,self.q,self.scale)

def make_model_and_args():
    return Model(), (torch.arange(32,dtype=torch.bfloat16).reshape(1,32)/8,)
