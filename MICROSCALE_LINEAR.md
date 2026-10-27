# Native MXFP4 / NVFP4 weight Linear

`nnc.LinearMxFp4(linear)` and `nnc.LinearNvFp4(linear)` wrap `torch.nn.Linear`
using the same frozen BF16 weight/bias module convention as `LinearFp4`.
NNC emits `linear_mxfp4` / `linear_nvfp4`; activations and public output are
BF16. This is weight-only inference, not a full FP4-activation GPU GEMM recipe.

| Preset | Weight | K group | Local scale | WSLD stype | Global scale |
| --- | --- | ---: | --- | ---: | --- |
| Existing FP4 | E2M1 | 32 | BF16 | 0 | none |
| MXFP4 | E2M1 | 32 | E8M0 byte | 1 | none |
| NVFP4 | E2M1 | 16 | E4M3FN byte | 2 | BF16 in FP32 container, per weight tensor |

Neither new preset has a weight zero point or shift. Public operands are
contiguous `[M,K]` / `[M,N]`; N must be divisible by 8 and K by the indicated
group size. M is positive; token chunks and tails reuse the existing runtime.

## Runnable Linear examples

The public [MXFP4 model](example/linear_mxfp4/model.py) and
[NVFP4 model](example/linear_nvfp4/model.py) each wrap an ordinary
`torch.nn.Linear(64, 16, bias=False)` and consume BF16 input `[9,64]`.
Both have a minimal C++ main and are linked into the NNC smoke discovery as
`smoke_linear_mxfp4_example.py` and `smoke_linear_nvfp4_example.py`.

The edge-cores workspace exposes the corresponding `example/linear_mxfp4`
and `example/linear_nvfp4` directories, including saved headers, packed binary
payloads, goldens and runners:

```sh
./example/linear_mxfp4/run.sh
./example/linear_nvfp4/run.sh
```

These runners compare all 144 BF16 outputs at zero tolerance without importing
PyTorch. From that workspace, explicitly regenerate the saved artifacts with
`python -m example.linear_mxfp4.model` or
`python -m example.linear_nvfp4.model` using a PyTorch environment.

## Source codes and quantization

`nnc.microscale.quantize(weight,spec)` returns exact E2M1 values in a BF16
carrier, original byte scale codes, and a scalar BF16 global scale in an FP32 container
(one for MXFP4). The deterministic MX quantizer chooses a non-clipping power-of-two
scale `2**ceil(log2(amax/6))` per K32 group; this is an explicit encoder policy,
not a promise to reproduce another tool's quantization choices. NVFP4 uses
`global_amax/(448*6)` for the global scale, rounds it to BF16 with RNE,
keeps it in an FP32 container with low16 bits zero, and then converts
`group_amax/(6*global_scale)` to E4M3FN. Zero tensors use a global scale of one.
Nearest E2M1 values use ties to lower magnitude, as the existing FP4 encoder.

Existing checkpoint codes can bypass quantization with static Parameters or
immutable registered buffers:
`linear_mxfp4(x,q,scale_codes)` or
`linear_nvfp4(x,q,scale_codes,global_scale)`. Codes must be uint8; global scale
must be one positive finite FP32 value that rounds to positive finite BF16.
Explicit checkpoint globals follow the same BF16 RNE conversion; weight
nibbles and E4M3 scale codes are preserved. There is no checkpoint file importer.
MX scale code 255 and negative
or NaN NV scales are rejected by the software packer.

## WSLD and transport storage

An eight-channel group stores eight scale bytes followed by four MX / two NV
output-major 8x8 E2M1 tiles, earlier nibble low. Native payload sizes are
136 bytes (MX) and 72 bytes (NV). Two adjacent groups share one physical
DMA slot: **272 bytes / 512 weights** for MX and **144 bytes / 256 weights**
for NV. This attains 4.25 / 4.5 bits per weight, respectively, plus NV's
single four-byte global container. There is no interior padding. An odd final
group has eight trailing padding bytes, so total storage is
`align_up(group_count * native_group_bytes, 16)`.

The circular transport contract requires slot bases, physical slot strides
and ring wrap to align to **128 bits (16 bytes)**. `EdgeCircularDmaLayout`
checks slot and whole-ring alignment with C++ `static_assert`; `WeightPack`
and runtime DMA setup instantiate it. Native presets use eight physical
slots. DMA continues copying complete 128-bit beats and publishing physical
slot credits without knowing quantization formats. WSLD descriptor bit 20
selects two groups per slot; the tile field remains tiles per group. WSLD
retains residual bytes across the middle boundary and requires a fresh scale
before the second group's weights. It releases the physical slot only after
both groups complete. For an odd tail, runtime drains DMA/tensor work and
restarts with one padded group, retaining the partial sums.

WSLD expands E8M0 exactly to BF16 with four appended zero fraction bits.
For NVFP4 it multiplies the E4M3 local scale by the BF16 global operand:
eight global significand bits times four local significand bits form an
exact twelve-bit significand, carried as **BF20 (1/8/11)**. No BF19 scale
rounding is inserted. Scale response high16 bits and low4 fraction bits per
lane travel together through the scale token, buffers and reuse path.

`edge_tensor_wsld_global(uint32_t)` sets the positive BF16-in-FP32 operand
before circular NVFP4 starts. The existing WSLD opcode carries bit31 as a
setup tag; hardware captures bits30:16 with sign zero and ignores bits15:0.
Setup waits for the previous tensor/WSLD consumer to drain. Reset selects
one; the value persists across physical slots and an odd-tail restart.
BF16 and MXFP4 stypes ignore this global operand. It is stored once per
weight tensor and is not duplicated in DMA slots.

The byte cursor and DMA/slot owner use
the one-byte prefix sizes. Scale sharing is across four K8 tiles for MX and
two for NV; a new group supplies a fresh scale token.

## Numerical and execution boundaries

The execution order is:

1. Quantization rounds the NVFP4 global scalar to BF16 before selecting local
   E4M3 scales and weight codes. Imported explicit codes keep their existing
   nibbles/scales and use the same BF16 global conversion.
2. WSLD forms the BF20 local/global product without scale RNE. The existing
   normal-result policy flushes scale-product underflow to signed zero;
   overflow produces infinity and NaN inputs canonicalize to NaN.
3. Dot8 aligns each 16-bit significand product to the shared maximum exponent
   with RNE, reduces the aligned products and rounds the dot to BF22.
4. Post-scale multiplies **BF22 x BF20** with a 14x12 unsigned significand
   multiplier and rounds directly to BF22. BF22 PSUM addition retains its
   existing rounding boundary. This multiplier has two register cuts and II=1.
5. The final tile rounds to contiguous BF16 output; a separate bias is added
   afterward. There is no scalar FP32 global-scale epilogue or FP32 output
   scratch for NVFP4.

E8M0/E4M3 encodings are unchanged. Relative to an NVFP4 implementation using
a full FP32 per-tensor global, this runtime uses BF16 RNE for that scalar and
applies the combined local/global scale at each K8 dot before accumulation.
The reference models these settings, including dot alignment. It is not a
vendor GEMM bit-identity contract; maintained ordinary-range software tests
compare every BF16 output with zero tolerance.

MXFP4 uses its original E8M0 codes, no global scale and the same BF22 arithmetic
boundaries. The original `linear_fp4` name continues selecting K32/BF16 scales.
E8M0 code0 expands to BF16 subnormal 2**-127 and 255 to NaN at transport ingress;
the existing tensor arithmetic policies apply afterward.

## Full FP32 global-scale extension

To support NVFP4 with its full FP32 global-scale precision, replace the
current post-scale product with native FP32 multiplication. The wider
significand multiplier may require additional pipeline stages; stage count
and timing must be checked with synthesis and detailed routing.

This extension must also retain the original FP32 global throughout
quantization, setup and scale transport: remove the software BF16 RNE
conversion and the hardware high16-only capture. Preserve the E4M3 local
scale and define its multiplication/rounding boundary with the FP32 global,
either through an appropriately widened combined-scale path or a separate
global-scale multiplication. Widening only the final multiplier cannot
recover global bits discarded earlier. Update scale storage, valid timing
and PSUM alignment to match the selected pipeline.

This describes preserving NVFP4's FP32 global-scale precision. Matching a
particular GEMM implementation bit for bit additionally requires matching
its dot accumulation, scale application order and output rounding.

## Verification

`python -m unittest nnc.test_microscale` checks byte scales, slot order and
padding, preserved codes, group sizes, global-scale placement, static lowering
and generated global-scale bits. Fourteen native FP4 smoke
models exercise multiple groups/channels, token tails, K16-only shapes and
single-token DTCM operands, repeated circular wrap, odd K-group tails and
physical pairs crossing output-channel blocks, an observable BF20 low-fraction
case and NVFP4/MXFP4/NVFP4 global reprogramming. The consumer registers them
under CTest's `microscale` label with exact output comparison.
The two public wrapper examples above are additional smoke entries and
exercise fresh compiler generation as well as saved-artifact replay in the
consuming workspace.
