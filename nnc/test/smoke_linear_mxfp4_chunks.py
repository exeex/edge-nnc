"""MXFP4 token-chunk restart and tail."""
from nnc.test.smoke_linear_mxfp4 import make_case


def make_model_and_args():
    return make_case('mxfp4',rows=65)
