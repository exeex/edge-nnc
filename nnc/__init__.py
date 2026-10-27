"""PyTorch export and lowering tools for edge-e3."""
from nnc.modules import LinearFp4, LinearQ8_0, LinearFp8E4M3FN, LinearFp8E5M2, LinearMxFp4, LinearNvFp4


def compile(model, example_args, *, out_dir="build/generated", save_reference=False,
            quant_token_capacity=None, matmul_token_capacity=None):
    """Generate C++ and packed weights for an eval model and example inputs."""
    from nnc.compiler import compile as compile_to_cpp
    return compile_to_cpp(model, example_args, out_dir=out_dir, save_reference=save_reference,
                          quant_token_capacity=quant_token_capacity,
                          matmul_token_capacity=matmul_token_capacity)


# Legacy smoke models still define their own BF16 custom ops. Register the
# reusable transformer ops only when a transformer module is requested.
def __getattr__(name):
    if name in {"Linear", "RMSNorm", "RoPE", "KVCacheUpdate", "Attention"}:
        from importlib import import_module
        value = getattr(import_module("nnc.transformer"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
