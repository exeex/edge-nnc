"""NVFP4 global scale on every token chunk including the one-token tail."""
from nnc.test.smoke_linear_mxfp4 import make_case


def make_model_and_args():
    return make_case('nvfp4',rows=65)
