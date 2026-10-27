"""Weight-only MXFP4/NVFP4: E2M1 codes and native byte block scales.

Groups are K32/E8M0 for MXFP4, K16/E4M3FN plus one FP32 tensor scale for
NVFP4. Slots retain the current 16-byte transport alignment (144/80 bytes).
"""
import struct
import torch
from nnc.fp4 import VALUES, encode_bf16, round_bf22


def group_k(spec):
    if spec not in ('mxfp4', 'nvfp4'):
        raise ValueError('expected mxfp4 or nvfp4')
    return 32 if spec == 'mxfp4' else 16


def decode_scales(codes, spec):
    if codes.dtype != torch.uint8:
        raise ValueError('microscale codes must be uint8')
    if spec == 'mxfp4':
        if (codes == 255).any():
            raise ValueError('NaN E8M0 scale')
        return torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes.int()-127)
    group_k(spec)
    values = codes.contiguous().view(torch.float8_e4m3fn).float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('NVFP4 block scales must be nonnegative finite')
    return values


def quantize(weight, spec):
    size = group_k(spec)
    w = weight.detach().cpu().float()
    if w.ndim != 2 or min(w.shape) <= 0 or w.shape[0] % 8 or w.shape[1] % size or not torch.isfinite(w).all():
        raise ValueError('microscale weights require finite [N%8=0,K%group=0]')
    blocks = w.reshape(w.shape[0], -1, size)
    maximum = blocks.abs().amax(-1)
    if spec == 'mxfp4':
        # Deterministic non-clipping power-of-two policy. Checkpoint import
        # instead supplies its original codes/scales and never requantizes.
        exponent = torch.where(maximum == 0, torch.zeros_like(maximum),
                               torch.ceil(torch.log2(maximum / 6)))
        if (exponent < -127).any() or (exponent > 127).any():
            raise ValueError('MXFP4 scale out of E8M0 range')
        codes = (exponent + 127).to(torch.uint8)
        global_scale = torch.tensor(1., dtype=torch.float32)
    else:
        amax = float(maximum.max())
        global_scale = torch.tensor(amax / (448 * 6) if amax else 1., dtype=torch.float32)
        codes = (maximum / (6 * global_scale)).clamp(0, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = decode_scales(codes, spec) * global_scale
    normalized = torch.where(scale.unsqueeze(-1) != 0, blocks / scale.unsqueeze(-1), torch.zeros_like(blocks))
    table = torch.tensor(VALUES)
    indices = (normalized.abs().unsqueeze(-1) - table).abs().argmin(-1)
    q = torch.copysign(table[indices], normalized).to(torch.bfloat16)
    return q.reshape(w.shape), codes, global_scale


def pack_slots(q, codes, spec):
    size = group_k(spec)
    if q.ndim != 2 or min(q.shape) <= 0 or q.shape[0] % 8 or q.shape[1] % size or tuple(codes.shape) != (q.shape[0], q.shape[1] // size):
        raise ValueError('invalid microscale slot shapes')
    decode_scales(codes, spec)
    nibbles = encode_bf16(q)
    n, k = q.shape
    payload = bytearray()
    for ob in range(n // 8):
        for group in range(k // size):
            payload.extend(bytes(codes[ob*8:ob*8+8, group].tolist()))
            for tile in range(size // 8):
                values = nibbles[ob*8:ob*8+8, group*size+tile*8:group*size+tile*8+8].flatten()
                payload.extend(bytes((values[::2] | (values[1::2] << 4)).tolist()))
            payload.extend(bytes((-len(payload)) % 16))
    return bytes(payload)


def global_bits(value):
    if value.numel() != 1 or value.dtype != torch.float32 or not torch.isfinite(value).all() or float(value) <= 0:
        raise ValueError('NVFP4 global scale must be one positive finite FP32')
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def reference(x, q, codes, global_scale, spec):
    size = group_k(spec)
    scales = decode_scales(codes, spec).to(x.device)
    result = torch.zeros((*x.shape[:-1], q.shape[0]), device=x.device)
    for offset in range(0, q.shape[1], 8):
        dot = round_bf22(x[..., offset:offset+8].float() @ q[:, offset:offset+8].float().T)
        result = round_bf22(result + round_bf22(dot * scales[:, offset // size]))
    global_bits(global_scale)
    return (result * global_scale.to(x.device)).to(torch.bfloat16).to(x.dtype)


@torch.library.custom_op('nnedge::linear_mxfp4', mutates_args=())
def linear_mxfp4(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor | None = None) -> torch.Tensor:
    if scales is None:
        weight, scales, _ = quantize(weight, 'mxfp4')
        weight = weight.to(x.device)
    return reference(x, weight, scales, torch.tensor(1.,dtype=torch.float32), 'mxfp4')


@linear_mxfp4.register_fake
def _(x, weight, scales=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op('nnedge::linear_nvfp4', mutates_args=())
def linear_nvfp4(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor | None = None,
                 global_scale: torch.Tensor | None = None) -> torch.Tensor:
    if scales is None:
        weight, scales, global_scale = quantize(weight, 'nvfp4')
        weight = weight.to(x.device)
    if global_scale is None:
        raise ValueError('NVFP4 requires global scale')
    return reference(x, weight, scales, global_scale, 'nvfp4')


@linear_nvfp4.register_fake
def _(x, weight, scales=None, global_scale=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
