# Named quantized Linear models

Each spec has a readable PyTorch source model and a seven-line C++ main:

| Model directory | Python custom op / C++ entry | Weight encoding | Pack bytes |
| --- | --- | --- | ---: |
| `example/linear_fp4` | `linear_fp4` | E2M1 | 144 |
| `example/linear_q8_0` | `linear_q8_0` | signed INT8 | 272 |
| `example/linear_fp8_e4m3fn` | `linear_fp8_e4m3fn` | E4M3FN | 272 |
| `example/linear_fp8_e5m2` | `linear_fp8_e5m2` | E5M2 | 272 |

All four use BF16 input/output, K32 groups, and eight BF16 scales per pack.
Weights and scales are static parameters. Public inputs and outputs remain
contiguous `[M,K]` and `[M,N]`; N must be a positive multiple of 8 and K a
positive multiple of 32. Each model uses M=9, N=16, K=64 to exercise token
batching, multiple output blocks and changing K groups.

## Source to C++

Use `LinearFp4`, `LinearQ8_0`, `LinearFp8E4M3FN`, or `LinearFp8E5M2` from
`nnc` to wrap an existing `torch.nn.Linear`:

```python
import torch
import nnc
from nnc import LinearFp4

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearFp4(torch.nn.Linear(64, 16))

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    m = Model().eval()
    x = torch.ones(9, 64, dtype=torch.bfloat16)
    nnc.compile(m, (x,), out_dir="build/generated", save_reference=True)
```

These inference modules copy the original weight and optional bias into frozen
BF16 parameters without modifying the source layer. They preserve the input
and output feature counts. The module imports and calls the corresponding
registered custom op with `(x, weight)`. NNC quantizes the static BF16 weight
according to the op name, then serializes the matching packed slots. A bias
is an ordinary BF16 per-row add; its broadcast is lowered through Tensor/DMA.
`register_fake` infers output shape/dtype only and preserves the BF16 carrier.
The real Python op quantizes the weight and evaluates the BF22 reference for
comparison; fake execution does not quantize.

`nnc.compile(model, example_args, out_dir=...)` is the public artifact API; it
returns the output directory. The compiler CLI and regenerating model runner use this same
implementation. `save_reference=True` also snapshots the BF16 output as
`expected.bin`. Each example's `model.py` has an explicit `__main__` call. In an
in-place checkout, run it as `python -m example.linear_fp4.model` from the repo
root so the local `nnc` package is importable.

The earlier `(x,q,scale)` custom-op signatures remain supported for explicit
format/precision fixtures. `nnc.fp4.quantize(weight)` produces exact E2M1
values in a BF16 carrier and BF16 scales. `nnc.quant.quantize(weight, spec)`
does the same for `q8_0`, `fp8_e4m3fn`, or `fp8_e5m2`. These helpers use
per-group max-absolute scaling rounded to BF16, then nearest INT8 or native
PyTorch FP8 conversion. Zero groups use scale one. INT8 uses a scale
denominator of 127 and clamps to [-128,127]; FP8 normalizes to the selected
dtype's maximum finite magnitude and clamps before conversion. The compiler
rejects nonfinite or inexact explicit carriers and requires positive finite
BF16 scales.

Export reads the named custom op; shared quant lowering packs the static
parameters and codegen emits the same named `nnedge::op::linear_*` call.
`cpp/libnn/linear_quant.hpp` fixes the corresponding `WeightPack` and forwards
the byte payload to `linear_weight_slots`. The operation returns synchronously
with the BF16 output ready. The numerical reference uses the same normal-range
BF22 dot, scale multiplication and partial-sum boundaries as FP4; it is not a
bit-true exceptional-value hardware model. See [FP4_LINEAR.md](FP4_LINEAR.md).

Generate artifacts without a platform build:

```sh
python -m nnc.compiler --model-file example/linear_q8_0/model.py --out-dir build/linear_q8_0
```

Outputs are `forward.hpp`, `init.hpp`, `weight.bin`, and `input.bin`.
The four named example models explicitly compile to their own directory with
`save_reference=True`, so their Git-tracked artifact set also has `expected.bin`.
The workspace example runners pass `--artifacts-dir` and rebuild/replay this
saved set without importing PyTorch or evaluating the model. Model invocation
is the explicit regeneration step; a Torch upgrade does not replace saved bytes.

The consuming workspace supplies the SDK, bare-metal build and simulator
through `nnc/run_smoke.py`. It also owns the user-facing `run.sh` entry points.
The runner links generic freestanding `memcpy` support, so model mains do not
need libc shims. An existing consumer's strong definition remains supported.

## Slot layout

Packs are ordered `[N/8,K/32]`. Each has eight little-endian BF16 scales,
followed immediately by four output-major 8x8 weight tiles. FP4 packs two
values per byte (earlier value in the low nibble); INT8 and FP8 use one byte
per value. These layouts require no tail padding at the default K32 group size.

`linear_q8_0` is the E3 Q8_0-style runtime spec. GGUF Q8_0 files use FP16 scales
and a different layout; they require conversion to BF16 scales and E3 tile
order. This quantizer is not a byte-identical implementation of llama.cpp's
quantizer or a GGUF importer.

## Regression ownership

`nnc/test_quant.py` checks module wrapping, named export with BF16 fake
metadata, compiler-side weight quantization, the public artifact API, all finite INT8/FP8 codes with an independent
decoder, exact serialized tile order and signed zeros, invalid carriers and
scales, named codegen, payload sizes, and group-scale reference semantics.
The four example source models are also linked into `nnc/test/smoke_*.py` and
are discovered by the full model harness. Existing FP4 precision, tail,
capacity and ring-wrap tests remain in that suite.

`nnc/test/packed_wsld_main.cpp` keeps the 26-case low-level WSLD regression;
`nnc/test/weight_pack_main.cpp` covers 16 generic and named runtime layouts.
Their platform adapter remains in the consuming workspace's test tree.
