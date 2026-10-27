"""Named K32 quant specs: BF16 scales and exact BF16 value carriers.

Q8_0 here describes the E3 runtime spec, not the GGUF FP16-scale file layout.
"""
import struct
import torch
from nnc.fp4 import round_bf22

FP8_DTYPES = {
    'fp8_e4m3fn': torch.float8_e4m3fn,
    'fp8_e5m2': torch.float8_e5m2,
}


def _validate_shape(q, scale=None):
    if q.ndim != 2 or min(q.shape) <= 0 or q.shape[0] % 8 or q.shape[1] % 32:
        raise ValueError('quant Linear requires positive [N%8=0,K%32=0] weights')
    if scale is not None and tuple(scale.shape) != (q.shape[0], q.shape[1] // 32):
        raise ValueError('quant Linear requires scales[N,K/32]')


def quantize(weight, spec):
    """Per-output K32 max-absolute scaling; serialize with BF16 scales."""
    _validate_shape(weight)
    w = weight.detach().cpu().float()
    if not torch.isfinite(w).all():
        raise ValueError('quant Linear weights must be finite')
    limit = 127. if spec == 'q8_0' else torch.finfo(FP8_DTYPES[spec]).max
    blocks = w.reshape(w.shape[0], -1, 32)
    maximum = blocks.abs().amax(-1)
    scale = (maximum / limit).to(torch.bfloat16)
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('scale must be representable as positive finite BF16')
    normalized = blocks / scale.float().unsqueeze(-1)
    if spec == 'q8_0':
        q = normalized.round().clamp(-128, 127)
    else:
        q = normalized.clamp(-limit, limit).to(FP8_DTYPES[spec]).float()
    return q.reshape(w.shape).to(torch.bfloat16), scale


def pack_slots(q, scale, spec):
    """Eight BF16 scales followed by four output-major 8x8 byte tiles."""
    _validate_shape(q, scale)
    if q.dtype != torch.bfloat16 or scale.dtype != torch.bfloat16:
        raise ValueError('quant values and scales must be BF16 carriers')
    value, s = q.detach().cpu(), scale.detach().cpu()
    if not torch.isfinite(value).all() or not torch.isfinite(s).all() or (s <= 0).any():
        raise ValueError('quant values must be finite and BF16 scales positive finite')
    if spec == 'q8_0':
        if (value < -128).any() or (value > 127).any() or not torch.equal(value, value.round()):
            raise ValueError('Q8_0 carrier must contain exact signed INT8 values')
        encoded = value.to(torch.int8)
    else:
        encoded = value.to(FP8_DTYPES[spec])
        if not torch.equal(encoded.to(torch.bfloat16), value):
            raise ValueError('FP8 carrier must contain exact representable values')
    codes = encoded.contiguous().view(torch.uint8)
    n, k = value.shape
    payload = bytearray()
    for ob in range(n // 8):
        for group in range(k // 32):
            scales = s[ob*8:ob*8+8, group].contiguous().view(torch.int16)
            payload.extend(struct.pack('<8H', *(int(v) & 0xffff for v in scales)))
            for tile in range(4):
                block = codes[ob*8:ob*8+8, group*32+tile*8:group*32+tile*8+8]
                payload.extend(bytes(block.flatten().tolist()))
    return bytes(payload)


def _linear_reference(x, q, scale):
    # Match the existing FP4 normal-range BF22 numerical reference.
    result = torch.zeros((*x.shape[:-1], q.shape[0]), device=x.device)
    for offset in range(0, q.shape[1], 8):
        dot = x[..., offset:offset+8].float() @ q[:, offset:offset+8].float().T
        scaled = round_bf22(round_bf22(dot) * scale[:, offset // 32].float())
        result = round_bf22(result + scaled)
    return result.to(torch.bfloat16).to(x.dtype)


@torch.library.custom_op('nnedge::linear_q8_0', mutates_args=())
def linear_q8_0(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        q, scale = quantize(q, 'q8_0')
        q, scale = q.to(x.device), scale.to(x.device)
    return _linear_reference(x, q, scale)


@linear_q8_0.register_fake
def _(x, q, scale=None):
    return x.new_empty((*x.shape[:-1], q.shape[0]))


@torch.library.custom_op('nnedge::linear_fp8_e4m3fn', mutates_args=())
def linear_fp8_e4m3fn(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        q, scale = quantize(q, 'fp8_e4m3fn')
        q, scale = q.to(x.device), scale.to(x.device)
    return _linear_reference(x, q, scale)


@linear_fp8_e4m3fn.register_fake
def _(x, q, scale=None):
    return x.new_empty((*x.shape[:-1], q.shape[0]))


@torch.library.custom_op('nnedge::linear_fp8_e5m2', mutates_args=())
def linear_fp8_e5m2(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        q, scale = quantize(q, 'fp8_e5m2')
        q, scale = q.to(x.device), scale.to(x.device)
    return _linear_reference(x, q, scale)


@linear_fp8_e5m2.register_fake
def _(x, q, scale=None):
    return x.new_empty((*x.shape[:-1], q.shape[0]))
