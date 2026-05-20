import torch
from ....config_params import DDR_BASE, L0_SIZE_BYTES, NEST_NUM, SPU_NUM

_BYTES_PER_ELEM = {'fp16': 2, 'fp32': 4, 'fp64': 8,
                   'fp8': 1, 'int8': 1, 'int32': 4}

_CVT_DTYPE_IN = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64,
                 'fp8': torch.uint8, 'int8': torch.int8, 'int32': torch.int32}

# =============================================================================
# FP16 bit-pattern helpers + L0 block view + warp routing
# =============================================================================
def _fp16_low16(packed: int) -> torch.Tensor:
    """Decode bits[15:0] of an integer as an FP16 scalar (LE bit pattern)."""
    u16 = torch.tensor([packed & 0xFFFF], dtype=torch.int16)
    return u16.view(torch.float16)[0]

def _fp16_high16(packed: int) -> torch.Tensor:
    """Decode bits[31:16] of an integer as an FP16 scalar (LE bit pattern)."""
    u16 = torch.tensor([(packed >> 16) & 0xFFFF], dtype=torch.int16)
    return u16.view(torch.float16)[0]

def _fp16_raw_bits(scalar: torch.Tensor) -> int:
    """Reinterpret an FP16 scalar as its little-endian uint16 bit pattern."""
    t = scalar.to(torch.float16).reshape(1).contiguous().view(torch.int16)
    return int(t[0]) & 0xFFFF

def _l0_block_view(npu, nest: int, spu: int, reg: int) -> torch.Tensor:
    """Return an FP16 view of ``L0[(reg & 0x1F)*32 .. +32]`` (16 elements)."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    return l0.view(torch.float16)[off // 2:off // 2 + 16]

def _write_l0_fp16_scalar(npu, nest: int, spu: int, l0_offset: int,
                          scalar: torch.Tensor) -> None:
    """Write a single FP16 LE word at ``L0[l0_offset]``."""
    l0 = npu.mem.l0_byte(nest, spu)
    u16 = _fp16_raw_bits(scalar)
    l0[l0_offset] = u16 & 0xFF
    l0[l0_offset + 1] = (u16 >> 8) & 0xFF
