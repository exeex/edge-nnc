"""Nonidentity static Linear with nine tokens (partial 32-token batch)."""
import torch
from nnc.test.smoke_linear_dynamic import SmokeLinearDynamic


def make_model_and_args():
    model = SmokeLinearDynamic()
    model.weight = torch.nn.Parameter((torch.arange(8*16).reshape(8,16).float()%13-6)/16)
    x = (torch.arange(9*16).reshape(9,16).float()%11-5)/8
    return model, (x,)
