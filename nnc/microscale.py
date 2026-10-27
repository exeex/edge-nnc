"""Weight-only MXFP4/NVFP4: E2M1 codes and native byte block scales.

Groups are K32/E8M0 for MXFP4, K16/E4M3FN plus one FP32 tensor scale for
NVFP4. Global is rounded to BF16 in an FP32 container before local quantization.
NVFP4 local*global forms BF20 before the per-tile dot scale. Groups are dense; only the entire payload has final 16-byte padding.
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
        global_scale = rounded_global(global_scale)
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


def rounded_global(value):
    if value.numel() != 1 or value.dtype != torch.float32 or not torch.isfinite(value).all() or float(value) <= 0:
        raise ValueError('NVFP4 global scale must be one positive finite FP32')
    result = value.to(torch.bfloat16).float()
    if not torch.isfinite(result).all() or float(result) <= 0:
        raise ValueError('NVFP4 global scale must round to positive finite BF16')
    return result


def global_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(rounded_global(value))))[0]


def combined_scales(codes, global_scale):
    # 8-bit BF16 significand times 4-bit E4M3 significand is exact in BF20.
    result = decode_scales(codes, 'nvfp4') * rounded_global(global_scale).to(codes.device)
    # Same normal-result FTZ boundary as the tensor arithmetic.
    return torch.where(result.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(result), result)


def tensor_dot8(x, q):
    """E3 dot8: RNE each 16-bit product into the shared exponent domain."""
    a = x.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int64) & 0xffff
    b = q.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int64) & 0xffff
    # Tensor ingress maps NaN to zero and infinity to the largest finite BF16.
    def sanitize(raw):
        magnitude = raw & 0x7fff
        raw = torch.where(magnitude > 0x7f80, torch.zeros_like(raw), raw)
        return torch.where(magnitude == 0x7f80, (raw & 0x8000) | 0x7f7f, raw)
    a, b = sanitize(a)[:,None,:], sanitize(b)[None,:,:]
    ae, be = (a >> 7) & 255, (b >> 7) & 255
    product = (((ae != 0).long() << 7) | (a & 127)) * (((be != 0).long() << 7) | (b & 127))
    valid = ((a & 0x7fff) != 0) & ((b & 0x7fff) != 0)
    exponent = torch.where(ae == 0, 1, ae) + torch.where(be == 0, 1, be) - 134
    maximum = torch.where(valid, exponent, -512).amax(-1,keepdim=True)
    maximum = torch.where(valid.any(-1,keepdim=True), maximum, 0)
    shift = (maximum - exponent).clamp(0,16)
    quotient = product >> shift
    remainder = product & ((1 << shift)-1)
    half = 1 << (shift-1).clamp(min=0)
    increment = (shift != 0) & ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0)))
    aligned = torch.where(valid & (maximum-exponent < 17), quotient+increment, 0)
    aligned = torch.where(((a^b) & 0x8000) != 0, -aligned, aligned)
    dot = torch.ldexp(aligned.sum(-1).float(), maximum.squeeze(-1).int()-134)
    return round_bf22(dot)


def reference(x, q, codes, global_scale, spec):
    size = group_k(spec)
    scales = (combined_scales(codes, global_scale) if spec == 'nvfp4' else
              decode_scales(codes, spec)).to(x.device)
    result = torch.zeros((*x.shape[:-1], q.shape[0]), device=x.device)
    for offset in range(0, q.shape[1], 8):
        dot = tensor_dot8(x[..., offset:offset+8], q[:, offset:offset+8])
        if spec == 'nvfp4':
            # Exact 26-bit significand product, rounded directly to BF22.
            product = dot.double() * scales[:, offset // size].double()
            mantissa, exponent = torch.frexp(product)
            scaled = torch.ldexp(torch.round(mantissa * (1 << 14)), exponent - 14).float()
            scaled = torch.where(scaled.abs() < torch.finfo(torch.float32).tiny,
                                 torch.zeros_like(scaled), scaled)
        else:
            scaled = round_bf22(dot * scales[:, offset // size])
        result = round_bf22(result + scaled)
    global_bits(global_scale)
    return result.to(torch.bfloat16).to(x.dtype)


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
