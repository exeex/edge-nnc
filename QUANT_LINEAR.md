# Named quantized Linear models

Each spec has a readable PyTorch source model and a seven-line C++ main:

| Model directory | Python custom op / C++ entry | Weight encoding | Pack bytes |
| --- | --- | --- | ---: |
| `example/linear_fp4` | `linear_fp4` | E2M1 | 144 |
| `example/linear_q8_0` | `linear_q8_0` | signed INT8 | 272 |
| `example/linear_fp8_e4m3fn` | `linear_fp8_e4m3fn` | E4M3FN | 272 |
| `example/linear_fp8_e5m2` | `linear_fp8_e5m2` | E5M2 | 272 |

All four use BF16 input/output, K32 groups, and eight 16-bit scales per pack.
Q8_0 uses native FP16 scales (`stype=3`); FP4 and both FP8 presets use BF16
scales (`stype=0`).
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
returns exact code carriers and native scales for `q8_0` (FP16),
`fp8_e4m3fn` or `fp8_e5m2` (BF16).
`nnc.quant.quantize_source(weight, spec)` separates source code generation
from the current scale encoding:

- Q8_0 uses the FP32 K32 maximum/127 scale to generate signed INT8 codes,
  with ties rounded away from zero, then stores the source scale in FP16.
  An all-zero group has zero codes and zero scale.
- Both scaled FP8 policies use K32 max-absolute BF16 scales, normalize to the
  selected dtype's maximum finite magnitude, and use finite saturated native
  PyTorch FP8 conversion. All-zero FP8 groups use scale one.
- `quantize` preserves source scale bits and codes. Q8_0 never converts its
  FP16 scales to BF16. Explicit checkpoint codes/scales use the same packer
  without re-quantization.

The compiler rejects nonfinite or inexact explicit carriers and requires
nonnegative finite native scales (FP16 for Q8_0, BF16 for FP8). Zero scale
is valid, including Q8_0 zero blocks. FP4 keeps its existing quantizer rules.

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

Packs are ordered `[N/8,K/32]`. Each has eight little-endian native scales,
followed immediately by four output-major 8x8 weight tiles. FP4 packs two
values per byte (earlier value in the low nibble); INT8 and FP8 use one byte
per value. These layouts require no tail padding at the default K32 group size.

Q8_0, E4M3FN and E5M2 all use 8 x 16-bit scales plus 256 one-byte weights:
272 bytes per 8 channels x K32, or 8.5 stored bits/weight. Scale bits and
weight-code bits have ratio 1:16. Each channel's scale is shared across its
four K8 tiles; no scale is duplicated per tile and no zero-point is stored.
The layout is a reversible permutation of eight `[scale16, codes32]` row
blocks with no per-slot padding. This contract requires N%8=0 and K%32=0.

Q8_0 uses `EDGE_TENSOR_STYPE_FP16=3`. WSLD consumes the same 16-byte scale
prefix and exactly expands IEEE FP16 into BF20 (1/8/11), including FP16
subnormals as normal BF20 values. Signed zero is preserved, infinity expands
to infinity and NaNs canonicalize. FP16's eleven significand bits fit within
BF20's twelve; no scale rounding or new post-scale multiplier is needed.
The existing BF22 x BF20 product rounds directly to BF22, followed by BF22
PSUM addition and final BF16 output. The Q8_0 reference models shared-exponent
Dot8 alignment and direct product rounding without an intermediate FP32 RNE.

The source row-block payload and tiled payload are reversible permutations:
original INT8 codes and FP16 scale bits are preserved with identical storage
overhead. DMA remains format-independent, reading seventeen 128-bit beats
per slot. FP8 presets retain their BF16 scale policy. No GGUF file importer
is provided; callers supply extracted blocks or static codes/scales. These
format guarantees do not imply bit-identical llama.cpp GEMM accumulation.

## Regression ownership

`nnc/test_quant.py` checks module wrapping, named export with BF16 fake
metadata, compiler-side weight quantization, the public artifact API, all finite INT8/FP8 codes with an independent
decoder, exact serialized tile order and signed zeros, invalid carriers and
scales, named codegen, payload sizes, all nonnegative finite FP16 scale bits,
and group-scale reference semantics. A typed-pack C++ main runs the same
core fixture through `Tensor<Q8_0WeightPack>` and checks all outputs exactly.
WSLD leaf tests independently check all
65,536 FP16 encodings at read latency 1 and 4, scalar broadcast and stalls.
NNC Q8_0 tests cover native low fraction bits, subnormals/extrema, single-token
DTCM operands, multi-chunk tails and repeated circular wrap. These core cases
compare every BF16 output with zero tolerance. The original biased example
uses a 2^-9 absolute tolerance (one BF16 ULP at this fixture's largest affected
output magnitude) for the separate BF16 add: that existing unit
RNE-aligns each operand to the common exponent before adding, which can differ
from PyTorch single-rounding addition. Its matching core-only case uses zero
tolerance and the same weights/input.
The four example source models are also linked into `nnc/test/smoke_*.py` and
are discovered by the full model harness. Existing FP4 precision, tail,
capacity and ring-wrap tests remain in that suite.

`nnc/test/packed_wsld_main.cpp` keeps the 26-case low-level WSLD regression;
`nnc/test/weight_pack_main.cpp` covers 16 generic and named runtime layouts.
Their platform adapter remains in the consuming workspace's test tree.
