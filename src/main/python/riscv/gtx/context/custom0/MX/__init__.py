import numpy as np
from ....config_params import (
    DDR_BASE, L0_SIZE_BYTES, NEST_NUM, SPU_NUM, MX_IO_DTYPE, MX_IO_BYTES,
)

_BYTES_PER_ELEM = {'fp16': 2, 'fp32': 4, 'fp64': 8,
                   'fp8': 1, 'int8': 1, 'int32': 4, 'io': MX_IO_BYTES}

_CVT_DTYPE_IN = {'fp16': np.float16, 'fp32': np.float32, 'fp64': np.float64,
                 'fp8': np.uint8, 'int8': np.int8, 'int32': np.int32,
                 'io': MX_IO_DTYPE}

# =============================================================================
# FP16 bit-pattern helpers + L0 block view + warp routing
# =============================================================================
def _fp16_low16(packed: int) -> np.ndarray:
    """Decode bits[15:0] of an integer as an FP16 scalar (LE bit pattern).

    Must use uint16, not int16: bit patterns >= 0x8000 (negative FP16, e.g. a
    -inf max-seed) overflow int16's [-32768, 32767] range and raise at tensor
    construction (crashed SOFT_MAX/ESUM before this fix).
    """
    u16 = np.array([packed & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]

def _fp16_high16(packed: int) -> np.ndarray:
    """Decode bits[31:16] of an integer as an FP16 scalar (LE bit pattern)."""
    u16 = np.array([(packed >> 16) & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]

def _fp16_raw_bits(scalar: np.ndarray) -> int:
    """Reinterpret an FP16 scalar as its little-endian uint16 bit pattern."""
    t = scalar.astype(np.float16).reshape(1).copy().view(np.int16)
    return int(t[0]) & 0xFFFF

def _l0_block_view(npu, nest: int, spu: int, reg: int) -> np.ndarray:
    """Return an FP16 view of ``L0[(reg & 0x1F)*32 .. +32]`` (16 elements)."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    return l0.view(np.float16)[off // 2:off // 2 + 16]

def _write_l0_fp16_scalar(npu, nest: int, spu: int, l0_offset: int,
                          scalar: np.ndarray) -> None:
    """Write a single FP16 LE word at ``L0[l0_offset]``."""
    l0 = npu.mem.l0_byte(nest, spu)
    u16 = _fp16_raw_bits(scalar)
    l0[l0_offset] = u16 & 0xFF
    l0[l0_offset + 1] = (u16 >> 8) & 0xFF


# =============================================================================
# MX_IO_DTYPE bit-pattern helpers + I/O-width views (FP32 default / FP16 toggle)
# Each mirrors its _fp16_* / _l0_block_view sibling but the element width tracks
# config_params.MX_IO_DTYPE so flipping the toggle flips every MX op at once.
# =============================================================================
def _io_low(packed: int) -> np.ndarray:
    """Low MX_IO_DTYPE element of a packed operand register, as a 0-d tensor
    (bits[15:0] for FP16, bits[31:0] for FP32)."""
    if MX_IO_BYTES == 4:
        u = np.array([packed & 0xFFFFFFFF], dtype=np.uint32)
    else:
        u = np.array([packed & 0xFFFF], dtype=np.uint16)
    return u.view(MX_IO_DTYPE)[0]


def _io_high(packed: int) -> np.ndarray:
    """High MX_IO_DTYPE element of a packed operand register
    (bits[31:16] for FP16, bits[63:32] for FP32)."""
    if MX_IO_BYTES == 4:
        u = np.array([(packed >> 32) & 0xFFFFFFFF], dtype=np.uint32)
    else:
        u = np.array([(packed >> 16) & 0xFFFF], dtype=np.uint16)
    return u.view(MX_IO_DTYPE)[0]


def _io_raw_bits(scalar: np.ndarray) -> int:
    """Reinterpret an MX_IO_DTYPE scalar as its little-endian unsigned bits."""
    s = scalar.astype(MX_IO_DTYPE).reshape(1).copy()
    if MX_IO_BYTES == 4:
        return int(s.view(np.int32)[0]) & 0xFFFFFFFF
    return int(s.view(np.int16)[0]) & 0xFFFF


def _l0_block_view_io(npu, nest: int, spu: int, reg: int) -> np.ndarray:
    """MX_IO_DTYPE view of the 32-byte SVR block ``L0[(reg & 0x1F)*32 .. +32]``
    (16 elements for FP16, 8 for FP32)."""
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    n = 32 // MX_IO_BYTES
    return npu.mem.l0_io(nest, spu)[off // MX_IO_BYTES: off // MX_IO_BYTES + n]


# Raw unsigned-int view + full-width mask at the MX I/O width — for the bitwise
# LOGIC ops (AND/OR/NOT/SHIFT), which act on bit patterns, not numeric values.
_IO_UINT: type = {2: np.uint16, 4: np.uint32}[MX_IO_BYTES]
_IO_MASK: int = (1 << (8 * MX_IO_BYTES)) - 1


def _l0_block_view_uint(npu, nest: int, spu: int, reg: int) -> np.ndarray:
    """Raw ``_IO_UINT`` view of the 32-byte SVR block (16 uint16 / 8 uint32)."""
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    n = 32 // MX_IO_BYTES
    base = npu.mem.l0_byte(nest, spu).view(_IO_UINT)
    return base[off // MX_IO_BYTES: off // MX_IO_BYTES + n]


def _l1_view_addr_io(npu, nest: int, spu: int, addr_byte: int,
                     length: int) -> np.ndarray:
    """MX_IO_DTYPE view of ``L1[addr : addr + length*MX_IO_BYTES]`` (no copy)."""
    off = addr_byte // MX_IO_BYTES
    return npu.mem.l1_io(nest, spu)[off: off + length]


def _write_l0_io_pair(npu, nest: int, spu: int, off: int,
                      low: np.ndarray, high: np.ndarray) -> None:
    """Write two MX_IO_DTYPE scalars into a 32-byte SVR block: ``low`` at the
    low element, ``high`` at the next, the rest of the block zeroed.

    FP32: low[31:0], high[63:32]; FP16: low[15:0], high[31:16]."""
    l0 = npu.mem.l0_byte(nest, spu)
    l0[off:off + 32] = 0
    lb = _io_raw_bits(low)
    hb = _io_raw_bits(high)
    for i in range(MX_IO_BYTES):
        l0[off + i] = (lb >> (8 * i)) & 0xFF
        l0[off + MX_IO_BYTES + i] = (hb >> (8 * i)) & 0xFF
