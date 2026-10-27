"""BF20 local/global product has observable low fraction bits at BF16 output."""
import torch
from nnc.microscale import linear_nvfp4


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('q',torch.ones(8,16,dtype=torch.bfloat16))
        self.register_buffer('scales',torch.full((8,1),0x3f,dtype=torch.uint8))
        # Compiler rounds this FP32 input to BF16 1.9921875 before WSLD setup.
        self.register_buffer('global_scale',torch.tensor(1.993))

    def forward(self,x):
        return linear_nvfp4(x,self.q,self.scales,self.global_scale)


def make_model_and_args():
    x=torch.zeros(3,16,dtype=torch.bfloat16)
    x[:,0]=torch.tensor([.2578125,-.2578125,.515625])
    return Model().eval(),(x,)
