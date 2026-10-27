"""E2M1 values carried in BF16; only serialized weights use packed nibbles."""
import struct
import torch

# Indexed by the three magnitude bits. Sign is bit 3.
VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
BF16_BITS = (0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0)


def quantize(weight):
    """Per-row K32 groups; nearest value, ties to lower magnitude."""
    w = weight.detach().cpu().float()
    if w.ndim != 2 or w.shape[0] % 8 or w.shape[1] % 32 or not torch.isfinite(w).all():
        raise ValueError('FP4 requires finite [N%8=0, K%32=0] weights')
    blocks = w.reshape(w.shape[0], -1, 32)
    maximum = blocks.abs().amax(-1)
    scale = (maximum / 6).to(torch.bfloat16)
    # Zero groups use one; reject unrepresentable nonzero scales explicitly.
    scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('FP4 scale is not representable as positive finite BF16')
    normalized = blocks / scale.float().unsqueeze(-1)
    table = torch.tensor(VALUES)
    index = (normalized.abs().unsqueeze(-1) - table).abs().argmin(-1)
    q = torch.copysign(table[index], normalized).to(torch.bfloat16)
    return q.reshape(w.shape), scale


def encode_bf16(value):
    """Bit mapping, not a second quantization: reject non-E2M1 BF16 values."""
    if value.dtype != torch.bfloat16:
        raise ValueError('FP4 carrier must be BF16')
    bits = value.detach().cpu().to(torch.bfloat16).contiguous().view(torch.int16)
    raw = bits.to(torch.int32) & 0xffff
    mag = raw & 0x7fff
    codes = torch.zeros_like(raw)
    valid = torch.zeros_like(raw, dtype=torch.bool)
    for code, pattern in enumerate(BF16_BITS):
        match = mag == pattern
        codes = torch.where(match, code, codes)
        valid |= match
    if not valid.all():
        raise ValueError('weight tensor must contain exact FP4 values in BF16')
    return (codes | ((raw >> 12) & 8)).to(torch.uint8)


def pack_slots(q, scale):
    n, k = q.shape
    if n % 8 or k % 32 or tuple(scale.shape) != (n, k // 32):
        raise ValueError('invalid FP4 slot shapes')
    s = scale.detach().cpu().to(torch.bfloat16)
    if not torch.isfinite(s).all() or (s <= 0).any():
        raise ValueError('scale must be positive finite BF16')
    codes = encode_bf16(q)
    payload = bytearray()
    for ob in range(n // 8):
        for group in range(k // 32):
            sb = s[ob*8:ob*8+8, group].contiguous().view(torch.int16)
            payload.extend(struct.pack('<8H', *(int(v) & 0xffff for v in sb)))
            for tile in range(4):
                # Output-major tile, loaded with transpose=false like BF16 Linear.
                c = codes[ob*8:ob*8+8, group*32+tile*8:group*32+tile*8+8].flatten()
                payload.extend(bytes((c[0::2] | (c[1::2] << 4)).tolist()))
    return bytes(payload)


def round_bf22(value):
    raw = value.contiguous().float().view(torch.int32).to(torch.int64) & 0xffffffff
    rounded = ((raw + 0x1ff + ((raw >> 10) & 1)) & 0xfffffc00).to(torch.int32)
    return rounded.view(torch.float32)


@torch.library.custom_op('nnedge::linear_fp4', mutates_args=())
def linear_fp4(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        q, scale = quantize(q)
        q, scale = q.to(x.device), scale.to(x.device)
    # Normal-range numerical reference: BF22 dot, multiply, and partial sum.
    # Public output is rounded to BF16 only after the final tile.
    result = torch.zeros((*x.shape[:-1], q.shape[0]), device=x.device)
    for offset in range(0, q.shape[1], 8):
        dot = x[..., offset:offset+8].float() @ q[:, offset:offset+8].float().T
        scaled = round_bf22(round_bf22(dot) * scale[:, offset // 32].float())
        result = round_bf22(result + scaled)
    return result.to(torch.bfloat16).to(x.dtype)


@linear_fp4.register_fake
def _(x, q, scale=None):
    return x.new_empty((*x.shape[:-1], q.shape[0]))
