"""NVFP4 dense stream crosses the physical ring wrap with retained scale bytes."""
from nnc.test.smoke_linear_mxfp4 import make_case


def make_model_and_args():
    return make_case('nvfp4',width=512,outputs=8)
