# FP4 Linear with BF16 scale

The Python model uses ordinary BF16 tensors. `nnc.fp4.quantize(W)` returns
`q[N,K]` containing exact E2M1 values in BF16 and `scale[N,K/32]` in BF16.
`nnc.fp4.linear_fp4(x, q, scale)` supplies the PyTorch golden operation and an
exportable custom-op boundary. Store q and scale as non-trainable Parameters
for the compiler to serialize them. No PyTorch FP4 dtype is required.

Quantization chooses a BF16-rounded max-absolute/6 scale per output channel
and K32 group, then nearest E2M1 values with ties to smaller magnitude.
Zero groups use scale 1. Nonfinite inputs and unrepresentable positive scales
are rejected. The quantized values are ±{0, .5, 1, 1.5, 2, 3, 4, 6}.
`encode_bf16` maps the BF16 sign and exact magnitude bits to nibbles; invalid
carrier values are rejected instead of silently requantizing them.

Slots are ordered by output block, then K group. Each slot contains eight
little-endian BF16 scales (16 bytes), then four output-major 8x8 FP4 tiles
(32 bytes each). Earlier elements occupy the low nibble. A slot is 144 bytes,
with no internal padding. Constant payload bases are aligned to 16 bytes.
Compared with 512 bytes of BF16 weights per group, the ratio is 3.56x including
scale. Public tensors remain contiguous; the packed payload is Tensor<uint8_t>.

The E3 runtime uses wtype=4, stype=0, transpose=false, four tiles per slot,
eight scales and a four-slot circular ring. Tile mode stores intermediate
PSUM as BF22 in FP32 words: first tile skips P, middle tiles read/write P,
and the last tile reads P and writes BF16. The runtime reserves one private
128-byte region (only vector zero is used). The current E3 wide P/O layout
uses eight contiguous FP32 containers in its first 32 bytes. The legacy
+4-channel configuration instead places the two 16-byte halves 64 bytes apart.

Each tile computes `R22(P + R22(scale * R22(dot(x,q))))`. WSLD owns weight
and scale tokens; one start consumes one tile. Only the final output is
rounded to BF16. Python supplies a normal-range numerical reference with
these precision boundaries; it is not a bit-true model of dot alignment or
exceptional-value hardware behavior.

Initial supported shapes: x[1,K], q[N,K], K a positive multiple of 32, N a
positive multiple of 8; weights/scales must be static. The compiler rejects
multiple tokens and operands exceeding the 32 KiB scratch budget. Runtime
handles DTCM operands directly; external operands are staged via DMA before or
after circular loading. Multi-token batching and group sizes other than 32
are future work. This version requires the E3 BF22 P/O extension; older RTL
that stores `{bf16,16'b0}` loses the extra precision.

Validation:

- `python -m unittest nnc.test_compiler nnc.test_fp4`: compiler tests, all 16
  encodings including signed zero, ties, zero groups, invalid inputs,
  independent slot decoding, static lowering and multi-token rejection.
- `smoke_linear_fp4_group32.py`: single slot, distinct signed output lanes.
- `smoke_linear_fp4_group64.py`: changing scales across accumulated K groups.
- `smoke_linear_fp4.py`: N=24, K=128, nonidentity weights, 12 slots with ring wrap.
- `fp4_dram_main.cpp` with the last model: direct external input/output path.

Use the consumer's run_smoke wrapper and `--model-file nnc/test/<model>`;
for the external path additionally use `--main-file nnc/test/fp4_dram_main.cpp`.
Keep RTL error versus this quantized reference separate from quantization
error versus the original unquantized model.

## Initial BF16-PSUM validation checkpoint (2026-10-23)

Consumer n906 baseline `1edc8673`; NNC baseline `a8659e3` plus this change.
Python 3.12, PyTorch 2.13.0, Verilator 5.050; no RTL changes.
Compiler/packing tests: 14/14. Complete model RTL suite: 24/24.

| Fixture | Whole-program cycles | Max absolute error |
| --- | ---: | ---: |
| K32 / N8 | 10,181 | 0 |
| K64 / N16 | 10,707 | 0 |
| K128 / N24, ring wrap | 12,109 | 0.00292969 |
| K128 / N24, external IO | 10,597 | 0.00292969 |

The external IO case used a 0.004 comparison threshold. Whole-suite defaults
remain 0.05; the exact fixtures had zero error. Cycle counts include startup
and software overhead; the external IO entry point omits the generated
forward's allocations and extra copy, so these are not kernel-only comparisons.
Local consumer logs: `build/nnc-fp4/suite` and `build/nnc-fp4/dram`.

## BF22 P/O validation

The K128/N24 ring-wrap case now matches the BF22 numerical reference exactly
(12,509 whole-program cycles). K32/N8 and K64/N16 remain exact. The added
`smoke_linear_fp4_precision.py` accumulates 1+31*2^-12 across 32 tiles and
returns BF16 1.0078125; the old per-tile BF16 PSUM would return 1.0. This case
passes with zero tolerance. The full 25-model suite passes; direct external input/output also matches
exactly (10,997 whole-program cycles). Compiler/packing/reference tests: 15/15.
