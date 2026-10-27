"""PyTorch export and lowering tools for edge-e3."""
from nnc.modules import LinearFp4, LinearQ8_0, LinearFp8E4M3FN, LinearFp8E5M2


def compile(model, example_args, *, out_dir="build/generated", save_reference=False):
    """Generate C++ and packed weights for an eval model and example inputs."""
    from nnc.compiler import compile as compile_to_cpp
    return compile_to_cpp(model, example_args, out_dir=out_dir, save_reference=save_reference)
