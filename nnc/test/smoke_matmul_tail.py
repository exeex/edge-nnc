"""Two-head runtime Matmul with nine output rows."""
import torch
from nnc.test.smoke_matmul import SmokeMatmul


def make_model_and_args():
    lhs = (torch.arange(2*9*16).reshape(2,9,16).float()%11-5)/8
    rhs = (torch.arange(16*16).reshape(16,16).float()%13-6)/16
    return SmokeMatmul(), (lhs, rhs)
