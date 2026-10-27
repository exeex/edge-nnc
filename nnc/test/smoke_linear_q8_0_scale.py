"""Native half-scale bits, extrema, and subnormals across K32 groups."""
import torch
from nnc.quant import linear_q8_0


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # FP16 low fraction bits must reach BF20, not a BF16 stand-in.
        bits = torch.tensor([0x3fff,0x3c01,0x3555,0x0001,0x03ff,0x0400,0x7bff,0],dtype=torch.int16)
        self.register_buffer('scale',bits.view(torch.float16).reshape(8,1).repeat(1,3))
        q = torch.zeros(8,96,dtype=torch.bfloat16)
        q[:,0] = 1
        q[:,32] = -128
        q[:,64] = 127
        self.register_buffer('q',q)

    def forward(self,x):
        return linear_q8_0(x,self.q,self.scale)


def make_model_and_args():
    x = torch.zeros(9,96,dtype=torch.bfloat16)
    x[:3,0] = torch.tensor([.2578125,-.2578125,.515625])
    x[3:6,32] = torch.tensor([.03125,-.03125,.0625])
    x[6:,64] = torch.tensor([.03125,-.03125,.0625])
    return Model().eval(),(x,)
