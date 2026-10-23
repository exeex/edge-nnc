"""Two K groups must scale only new dots, never accumulated partial sums."""
import torch
from nnc.fp4 import linear_fp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        q = torch.zeros(16,64,dtype=torch.bfloat16)
        for lane in range(16):
            q[lane,lane%8] = 1
            q[lane,32+lane%8] = -2
        self.q = torch.nn.Parameter(q,requires_grad=False)
        self.scale = torch.nn.Parameter(torch.tensor([[.5,2.],[2.,.5]],dtype=torch.bfloat16).repeat(8,1),requires_grad=False)
    def forward(self,x):
        return linear_fp4(x,self.q,self.scale)

def make_model_and_args():
    return Model(), (torch.ones(1,64,dtype=torch.bfloat16),)
