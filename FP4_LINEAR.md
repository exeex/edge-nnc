# FP4 Linear with BF16 scale

Native byte-scale alternatives are now `nnc.LinearMxFp4` (K32 E8M0) and
`nnc.LinearNvFp4` (K16 E4M3FN plus FP32 tensor scale). The original
`LinearFp4` remains the BF16-scale preset. See
[MICROSCALE_LINEAR.md](MICROSCALE_LINEAR.md) for their storage and numerical
contracts; these require the matching WSLD stype decoder in E3/Edge32.

Use `nnc.LinearFp4(torch.nn.Linear(...))` in a model to replace a standard
Linear with an explicit `nnedge::linear_fp4` custom op. The module copies the
source weight and optional bias into frozen BF16 inference parameters; NNC
quantizes and packs the weight during compile. `register_fake` only returns
the BF16 output metadata. The model's `__main__` can call
`nnc.compile(m, example_args, out_dir=...)` to write C++ and payloads directly.
See [QUANT_LINEAR.md](QUANT_LINEAR.md) for the complete model flow and other
quant specs.

The explicit fixture form `nnc.fp4.linear_fp4(x, q, scale)` remains supported.
`nnc.fp4.quantize(W)` returns `q[N,K]` containing exact E2M1 values in BF16
and `scale[N,K/32]` in BF16. Store q and scale as non-trainable Parameters for
those prequantized compiler tests. No PyTorch FP4 dtype is required.

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

`cpp/libnn/linear_quant.hpp` provides the named `op::linear_fp4` entry point.
It fixes `FP4WeightPack = WeightPack<EDGE_TENSOR_WTYPE_FP4,0,4,8>` and forwards
typed packs or the compiler's byte payload to the shared `linear_weight_slots`
kernel. The same header also defines `linear_q8_0`, `linear_fp8_e4m3fn`, and
`linear_fp8_e5m2` runtime presets; [QUANT_LINEAR.md](QUANT_LINEAR.md) describes their graph
lowering and packing. GGUF import remains outside this path. Q8_0 preserves FP16 scales and uses E3 tile order after byte-preserving repacking.

The E3 runtime uses wtype=4, stype=0, transpose=false, four tiles per slot,
eight scales and a four-slot circular ring. Tile mode stores intermediate
PSUM as BF22 in FP32 words: first tile skips P, middle tiles read/write P,
and the last tile reads P and writes BF16. The E3 wide P/O layout uses eight contiguous FP32 containers per token.
A private P buffer holds the entire token chunk (32 bytes/token, minimum
128 bytes). The legacy single-vector allocation remains large enough for its
+4-channel separated halves; multi-token validation targets the E3 wide product.

Each tile computes `R22(P + R22(scale * R22(dot(x,q))))`. WSLD owns weight
and scale tokens; one start consumes one tile. Only the final output is
rounded to BF16. Python supplies a normal-range numerical reference with
these precision boundaries; it is not a bit-true model of dot alignment or
exceptional-value hardware behavior.

Supported shapes: x[M,K], q[N,K], M positive, K a positive multiple of 32,
N a positive multiple of 8; weights/scales must be static. Like BF16 Matmul,
the runtime uses one START per weight tile with `setn=token_count`. It chunks
M at 64 tokens or a smaller count fitting the 32 KiB scratch budget; the
compiler rejects shapes for which even one token cannot fit. Each chunk
replays the same packed weight slots through the four-slot circular ring.

Private input storage is [K/8,token,8], output storage is [N/8,token,8], and
P is [token,8] in FP32 containers. Public tensors remain contiguous [M,K]
and [M,N]. Multi-token generated calls prefer external operands for
source-strided DMA gather/scatter, as BF16 Matmul does. DTCM operands use
E3 Tensor COPY for each aligned 16-byte vector, avoiding unsupported
DTCM-to-DTCM DMA. A single token can directly alias DTCM tensors. Input packing
finishes before circular DMA starts; Tensor and DMA finish before output
scatter or scratch reuse. P is staged once per chunk and O changes only
between intermediate and final K tiles. Group sizes other than 32 remain
unsupported. This version requires the E3 BF22 P/O extension; older RTL
that stores `{bf16,16'b0}` loses the extra precision.

Validation:

- `python -m unittest nnc.test_compiler nnc.test_fp4`: compiler tests, all 16
  encodings including signed zero, ties, zero groups, invalid inputs,
  independent slot decoding, static lowering and multi-token shapes.
- `smoke_linear_fp4_group32.py`: single slot, distinct signed output lanes.
- `smoke_linear_fp4_group64.py`: changing scales across accumulated K groups.
- `smoke_linear_fp4.py`: N=24, K=128, nonidentity weights, 12 slots with ring wrap.
- `smoke_linear_fp4_batch.py`: M64/K64/N64, one weight tile serves 64 tokens.
- `smoke_linear_fp4_tail.py`: M9/K128/N24, token tail and distinct rows.
- `smoke_linear_fp4_chunks.py`: M65/K128/N24, 64+1 chunks and ring restart.
- `smoke_linear_fp4_capacity.py`: M33/K512/N16, scratch-limited 29+4 chunks.
- `fp4_dtcm_main.cpp`: force DTCM inputs/outputs; use tail and chunks models.
- `fp4_dram_main.cpp` with `smoke_linear_fp4.py`: direct external input/output path.

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
