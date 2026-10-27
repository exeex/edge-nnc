"""Sixteen physical slots traverse the eight-slot ring twice."""
from nnc.test.smoke_linear_mxfp4 import make_case


def make_model_and_args():
    return make_case('mxfp4',width=1024,outputs=8)
