"""Two-head transposed Matmul with nine output rows and one output tile/head."""
import torch
from nnc.test.smoke_matmul_transpose import SmokeMatmulTranspose


def make_model_and_args():
    lhs = (torch.arange(9 * 32).reshape(9, 32).float() % 11 - 5) / 8
    rhs = (torch.arange(8 * 32).reshape(8, 32).float() % 13 - 6) / 16
    return SmokeMatmulTranspose(), (lhs, rhs)
