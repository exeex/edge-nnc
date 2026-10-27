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

## Compile-time capacity and preplanned tensor IO

Each quant spec has one C++ entry point, matching NNC codegen: BF16 `Tensor`
input/output and `Tensor<uint8_t>` packed weight bytes. There are no separate
typed-pack or `TensorTileView` overloads in `linear_quant.hpp`.

All entries accept `MaxTokens=64` and `TensorSram=false`. NNC can
explicitly select the 32/64/128-token variants. Capacity remains bounded by
available SRAM; a smaller final chunk is supported.

The preplanned path selects `TensorSram=true`: logical shape
`[tokens, channels]`, physical storage `[channels/8, tokens, 8]`. NNC arranges
this storage before execution. The template flag performs no copy or conversion.
Inputs and outputs already in working SRAM are used directly. The kernel
allocates only FP32 partial sums and the circular weight ring; it does not
stage another copy of the input/output and submits no Tensor COPY commands.
A chunk points into each block using the full tensor's token extent, so
32/64-token versions and capacity-limited tails also need no SRAM repacking.

```cpp
// x/y are already in tensor working SRAM; slots are in DRAM.
nnedge::op::linear_mxfp4<128, true>(y, x, slots);
nnedge::op::linear_nvfp4<128, true>(y, x, slots, global_scale_bits);
nnedge::op::linear_q8_0<128, true>(y, x, slots);
```

For a complete tensor whose DRAM and SRAM storage use the same planned layout,
`load_tensor_sram` and `store_tensor_sram` each submit **one ordinary DMA**.
They reject SRAM-to-SRAM transfers. They do not use strided DMA. Circular
weight DMA still follows the format's scale/weight slot layout.

Export the preplanned ABI through NNC:

```python
nnc.compile(model, (x,), out_dir="build/linear", save_reference=True,
            tensor_tile_io=True, quant_token_capacity=128)
```

Or use `nnc.compiler --tensor-tile-io --quant-token-capacity 128 ...`.
The compiler serializes the example input into tensor order, generates SRAM
allocation and `linear_*<capacity, true>(y, x, slots)` calls, and records the physical IO ABI in
`io_layout.json`. `expected.bin` and the exported output storage use this
same physical order. A dynamic input producer must supply this planned order;
the option does not transpose dynamic data invisibly at runtime. Logical
PyTorch model inputs/outputs and the reference computation are unchanged.

This quant export supports **one quant Linear without bias**, with
its complete I/O fitting the 96 KiB tensor arena. It rejects incompatible
multi-op graphs or oversized I/O instead of applying the layout to operators
that do not understand it. The same compiler option separately supports a
single BF16 Matmul with an unmodified runtime RHS; see [MATMUL.md](MATMUL.md).
Layout propagation through a whole model and
optimal capacity selection are future NNC work. The default export remains
the existing contiguous ABI through the same entry point with
`TensorSram=false`. Set `TensorSram=true` only for compiler-planned SRAM
storage; it does not reinterpret contiguous input bytes correctly.

For 128 tokens and K=N=64, working input/output occupy 32 KiB. Additional
scratch is 5,248 bytes (NVFP4), 6,272 bytes (MXFP4), or 5,184 bytes (Q8_0).
Scratch avoids the allocator high-water mark and explicitly supplied SRAM
buffers, including buffers placed above the allocator cursor. All consumers
finish before the synchronous op returns. Other manually placed live buffers
must be represented in the allocator high-water mark.

The original contiguous path retains its default 32 KiB scratch contract;
variants above 64 may borrow unused arena space for their private staging.
The 4096×4096 example cannot stage 128 tokens even with all 128 KiB free:
its contiguous-path capacity planner permits at most seven before accounting
for other live tensors. That example is a planner check, not a claim that its
complete I/O can reside in the SRAM-native mode.

## 64×64 weights, 128-token SRAM benchmark

From the NNC repository root, with the consuming workspace's wide-H4 simulator
already built:

```sh
python nnc/benchmark_quant.py \
  --out-dir build/quant-dma-owned \
  --sim-exe /path/to/Vedge32_tensor_integration_tb \
  --edge32-include /path/to/edge-32/include \
  --e3-include /path/to/edge-e3/include
```

This runs NVFP4, MXFP4 and Q8_0 at capacities 32, 64 and 128. Repeated `--spec`
and `--capacity` options select subsets. The same dense source weights are
compiled through NNC. Weights stay in DRAM; planned input/output use working
SRAM directly. There is no warmup invocation. All 8192 BF16 output values
must exactly match the format-specific NNC golden in its planned storage.

`NNEDGE_QUANT_PROFILE=1` enables two kernel measurements. The stream window
starts before circular weight DMA submission and ends after tensor and DMA
sync, matching the original tiled circular benchmark's timing scope. For
multiple chunks, stream windows are summed. Operator time also includes setup
and capacity checks. Profile printing is outside both windows. The report
additionally lists ordinary I/O DMA costs and their sum with operator cycles,
so moving a transfer outside kernel timing does not hide its cost.

The runner checks 524,288 MACs, 8192 ideal cycles at 64 MACs/cycle, the compute
START count and zero extra COPY starts. Each 128-token format also verifies
its generated `forward.hpp`, allocator guards, manually placed SRAM input,
reduced capacity, and equality across chunk boundaries. With 4 KiB scratch
remaining, native capacities reduce to 92 (NVFP4), 60 (MXFP4), or 94 (Q8_0).
The legacy contiguous export has separate correctness regressions.

Artifacts include per-case generated headers, physical-layout metadata, ELF
images, raw logs, `results.md/csv/json`, and `provenance.json` with source,
compiler, simulator and payload hashes plus revisions/dirty state. The
consuming workspace's `build/quant-dma-owned/` contains this run's evidence.
A Q8_0 generated-forward command trace additionally verifies exactly three
DMA starts: ordinary 16 KiB input, circular weight stream, ordinary 16 KiB
output; zero Tensor COPY starts.

Measured 2026-10-28 on E3 `851bc5b6`, Edge-32 `55a089b3`, wide H4, LLVM
22.1.8, RV32IM/Zba, `-O2`, with this compiler/runtime change applied:

| Quant, capacity 128 | Stream cycles | Stream utilization | SRAM operator cycles | SRAM operator utilization | With ordinary I/O DMA | With DMA utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 8,833 | 92.74% | 9,423 | 86.94% | 20,559 | 39.85% |
| MXFP4 | 8,861 | 92.45% | 9,403 | 87.12% | 20,541 | 39.88% |
| Q8_0 | 8,765 | 93.46% | 9,210 | 88.95% | 20,344 | 40.27% |

These measurements follow the [tensor ownership contract](MEMORY_OWNERSHIP.md):
ASIC/DMA operators do not clean or invalidate D-cache. Removing redundant
cache maintenance reduced SRAM operator cycles from 9,789/9,761/9,900 to
9,423/9,403/9,210 for NVFP4/MXFP4/Q8_0 respectively.

The previous benchmark unnecessarily loaded contiguous SRAM operands and then
repacked them inside each op with 2048 Tensor COPY starts. Its ~132k-cycle
operator/~6.2% utilization describes that older path. The new measurement
uses NNC-planned physical IO; it is not a speedup of an unchanged contiguous
SRAM ABI. All preparation required by the new planned ABI is explicit, and
ordinary boundary DMA is reported separately above.

The original `matmul64x64_128tokens_tiled_circular.cpp` reference measured
8,638 stream cycles (94.84%) on the same simulator/compiler. Its INT8 identity
weights and legacy partial-sum mode differ from quant's dense scaled weights
and FP32 partial sums, so the stream gap is not isolated WSLD latency.

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
and group-scale reference semantics. An internal typed-pack C++ main runs
the same core fixture through the shared `linear_weight_slots` kernel and
checks all outputs exactly. Named quant entries take only NNC byte payloads.
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
