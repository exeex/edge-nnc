"""NVFP4 K16 scale updates and FP32 global-scale epilogue."""
from nnc.test.smoke_linear_mxfp4 import make_case


def make_model_and_args():
    return make_case('nvfp4',width=48)
