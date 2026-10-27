"""Linear-style static [N] bias broadcast over nine contiguous BF16 rows."""
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter((torch.arange(16)-4).to(torch.bfloat16)/8,
                                       requires_grad=False)

    def forward(self,x):
        return x + self.bias


def make_model_and_args():
    x = ((torch.arange(9*16).reshape(9,16)%13-6).float()/16).to(torch.bfloat16)
    return Model(), (x,)
