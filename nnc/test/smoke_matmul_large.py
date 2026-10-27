"""Tiled IO larger than SRAM uses the same Matmul kernel with DMA chunks."""
import torch
from nnc.test.smoke_matmul import nnedge_matmul
class Model(torch.nn.Module):
    def forward(self,x,rhs):
        return nnedge_matmul(x,rhs,1)
def make_model_and_args():
    x=((torch.arange(1024*64).reshape(1,1024,64)%13-6).float()/32).to(torch.bfloat16)
    rhs=((torch.arange(64*64).reshape(64,64)%11-5).float()/16).to(torch.bfloat16)
    return Model(),(x,rhs)
