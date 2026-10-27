# Native MXFP4 / NVFP4 weight Linear

`nnc.LinearMxFp4(linear)` and `nnc.LinearNvFp4(linear)` wrap `torch.nn.Linear`
using the same frozen BF16 weight/bias module convention as `LinearFp4`.
NNC emits `linear_mxfp4` / `linear_nvfp4`; activations and public output are
BF16. This is weight-only inference, not a full FP4-activation GPU GEMM recipe.

| Preset | Weight | K group | Local scale | WSLD stype | Global scale |
| --- | --- | ---: | --- | ---: | --- |
| Existing FP4 | E2M1 | 32 | BF16 | 0 | none |
| MXFP4 | E2M1 | 32 | E8M0 byte | 1 | none |
| NVFP4 | E2M1 | 16 | E4M3FN byte | 2 | FP32 per weight tensor |

Neither new preset has a weight zero point or shift. Public operands are
contiguous `[M,K]` / `[M,N]`; N must be divisible by 8 and K by the indicated
group size. M is positive; token chunks and tails reuse the existing runtime.

## Source codes and quantization

`nnc.microscale.quantize(weight,spec)` returns exact E2M1 values in a BF16
carrier, original byte scale codes, and a scalar FP32 global scale (one for
MXFP4). The deterministic MX quantizer chooses a non-clipping power-of-two
scale `2**ceil(log2(amax/6))` per K32 group; this is an explicit encoder policy,
not a promise to reproduce another tool's quantization choices. NVFP4 uses
`global_amax/(448*6)` for the global scale and converts
`group_amax/(6*global_scale)` to E4M3FN. Zero tensors use a global scale of one.
Nearest E2M1 values use ties to lower magnitude, as the existing FP4 encoder.

Existing checkpoint codes can bypass quantization with static Parameters or
immutable registered buffers:
`linear_mxfp4(x,q,scale_codes)` or
`linear_nvfp4(x,q,scale_codes,global_scale)`. Codes must be uint8; global scale
must be one positive finite FP32 value. Packing preserves nibbles and scale
codes. There is no checkpoint file importer. MX scale code 255 and negative
or NaN NV scales are rejected by the software packer.

## WSLD and transport storage

An eight-channel group stores eight scale bytes followed by four MX / two NV
output-major 8x8 E2M1 tiles, earlier nibble low. Native payload sizes are
136 bytes (MX) and 72 bytes (NV). The current circular transport requires
16-byte aligned slot bases, so the serialized runtime packs include eight
tail padding bytes: **144 bytes / 256 weights** for MX and
**80 bytes / 128 weights** for NV. These are 4.5 and 5 stored bits/weight,
respectively, plus NV's single four-byte global scalar in the generated code.
They do not yet attain dense checkpoint storage ratios 4.25 / 4.5 bits/weight.

This padding is explicit in `WeightPack`; WSLD reads only one-byte scales,
not BF16 substitutes. Runtime DMA entries use the aligned stride. Dense source
packing requires a separate transport adaptation: the current AXI-to-DTCM
stream requires 16-byte aligned sources, while alternate 136/72-byte group
bases are eight bytes off that alignment. Do not silently change DMA stride
without handling this source-alignment and group-boundary contract.

The WSLD decoder exactly expands E8M0/E4M3FN to the existing BF16 scale lanes
before issuing the matched response. The byte cursor and DMA/slot owner use
the one-byte prefix sizes. Scale sharing is across four K8 tiles for MX and
two for NV; a new group supplies a fresh scale token.

## Numerical and execution boundaries

The reference follows the existing normal-range BF22 dot, scale product and
PSUM rounding boundaries. MX rounds final output to BF16. NV instead writes
the final BF22 tile to private FP32 containers, applies the single global
scale with scalar FP32 `fmul.s` (RNE), then rounds to BF16 before any separate
bias addition. The soft-float build ABI uses explicit scalar FP instructions;
no compiler-rt floating-point helper is required. This scalar epilogue is a
correctness path, not a throughput claim or a fused hardware global-scale stage.

NV epilogue compacts scratch forward in-place, then uses Tensor COPY for
DTCM output or DMA for external output. It completes before scratch reuse.
Its four-byte-per-output scratch reservation participates in token capacity.
E8M0 code 0 is decoded to BF16 subnormal 2**-127, and 255 to NaN by transport;
existing tensor arithmetic's underflow/saturation/NaN policies are unchanged.
The software reference does not claim exceptional-range or vendor GEMM
bit identity; tested ordinary-range outputs use zero comparison tolerance.

## Verification

`python -m unittest nnc.test_microscale` checks byte scales, slot order and
padding, preserved codes, group sizes, global-scale placement, static lowering
and generated global-scale bits. Six `smoke_linear_{mx,nv}fp4{,_chunks,_single}`
models exercise multiple groups/channels, token tails, K16-only shapes and
single-token DTCM operands. The consumer registers them under CTest's
`microscale` label with exact output comparison.
