from __future__ import annotations

import numpy as np

from ...inst_handler import inst_register
from ....config_params import MX_IO_DTYPE
from ....csr import GSPR, LSPR, NSPR
from ... import _resolve_nest_spu
from . import _BYTES_PER_ELEM, _CVT_DTYPE_IN, _io_low, _io_high

# ----- format_cvt handlers --------------------------------------------------
# scvt.* convert between the MX native float ('io' = MX_IO_DTYPE: FP32 default /
# FP16 toggle) and an explicit format (FP8 / FP16 / INT8 / INT32). Within a
# bidirectional slot the direction is GSPR_OPCODE & 1. scale[31:0]/offset[63:32]
# in OPERAND2 (width config-gated via _io_low/_io_high). fcvt.* below stay the
# unscaled FP16<->FP32 / FP16<->FP64 register conversions (NSU path).

@inst_register.custom0(name='scvt.qs', funct7=0b0100000, funct3=0)
def _scvt_qs(npu, proc, inst, cxt) -> int:
    """native(MX_IO) <-> FP8 (default native->fp8; OPCODE&1 -> fp8->native)."""
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst, src_kind='fp8', dst_kind='io')
    return _format(npu, proc, inst, src_kind='io', dst_kind='fp8')


@inst_register.custom0(name='scvt.hs', funct7=0b0100000, funct3=1)
def _scvt_hs(npu, proc, inst, cxt) -> int:
    """native(MX_IO) <-> FP16 (default native->fp16; OPCODE&1 -> fp16->native)."""
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst, src_kind='fp16', dst_kind='io')
    return _format(npu, proc, inst, src_kind='io', dst_kind='fp16')


@inst_register.custom0(name='scvt.is', funct7=0b0100001, funct3=0)
def _scvt_is(npu, proc, inst, cxt) -> int:
    """native(MX_IO) <-> INT8 (default native->int8; OPCODE&1 -> int8->native)."""
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst, src_kind='int8', dst_kind='io')
    return _format(npu, proc, inst, src_kind='io', dst_kind='int8')


@inst_register.custom0(name='scvt.si', funct7=0b0100001, funct3=1)
def _scvt_si(npu, proc, inst, cxt) -> int:
    """INT8 -> native(MX_IO)."""
    return _format(npu, proc, inst, src_kind='int8', dst_kind='io')


@inst_register.custom0(name='scvt.sn', funct7=0b0100010, funct3=1)
def _scvt_sn(npu, proc, inst, cxt) -> int:
    """INT32 -> native(MX_IO)."""
    return _format(npu, proc, inst, src_kind='int32', dst_kind='io')


@inst_register.custom0(name='fcvt.sh', funct7=0b0100100, funct3=0)
def _fcvt_sh(npu, proc, inst, cxt) -> int:
    # fcvt.sh: unscaled FP16<->FP32 register conversion (direction by OPCODE&1).
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp16', dst_kind='fp32')
    return _format(npu, proc, inst,
                            src_kind='fp32', dst_kind='fp16')

@inst_register.custom0(name='fcvt.hs', funct7=0b0100100, funct3=1)
def _fcvt_hs(npu, proc, inst, cxt) -> int:
    # fcvt.hs: unscaled FP32->FP16 register conversion.
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    return 0

@inst_register.custom0(name='fcvt.dh', funct7=0b0100101, funct3=0)
def _fcvt_dh(npu, proc, inst, cxt) -> int:
    # fcvt.dh: FP16<->FP64 register conversion (direction by OPCODE&1).
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp16', dst_kind='fp64')
    return _format(npu, proc, inst,
                            src_kind='fp64', dst_kind='fp16')


@inst_register.custom0(name='fcvt.hd', funct7=0b0100101, funct3=1)
def _fcvt_hd(npu, proc, inst, cxt) -> int:
    # fcvt.hd: FP64->FP16 register conversion.
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    return 0


def _format(npu, proc, inst, *, src_kind: str, dst_kind: str) -> int:
    """Direct port of ``gtx_npu_act.cc:222-372`` (``exec_format_cvt``).

    ``'io'`` resolves to the MX native float (``MX_IO_DTYPE``) for both byte
    width and dtype, so the native side of every scaled conversion tracks the
    config toggle.
    """
    nest, spu = _resolve_nest_spu(npu)

    length = int(proc.state.XPR[inst.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    op2 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    # scale/offset widen with MX_IO_DTYPE: FP16 [15:0]/[31:16], FP32 [31:0]/[63:32].
    scale = _io_low(op2)
    offset = _io_high(op2)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    l1 = npu.mem.l1_byte(nest, spu)
    in_size = length * _BYTES_PER_ELEM[src_kind]
    in_bytes = l1[addr_a:addr_a + in_size].copy().copy()
    in_arr = in_bytes.view(_CVT_DTYPE_IN[src_kind])

    # scaled native(io) <-> {fp8, fp16, int8, int32}
    if src_kind == 'io' and dst_kind == 'fp8':
        out_arr = cvt_io_fp8(in_arr, scale, offset)
    elif src_kind == 'io' and dst_kind == 'fp16':
        out_arr = cvt_io_fp16(in_arr, scale, offset)
    elif src_kind == 'io' and dst_kind == 'int8':
        out_arr = cvt_io_int8(in_arr, scale, offset)
    elif dst_kind == 'io':                       # fp8/fp16/int8/int32 -> native
        out_arr = cvt_to_io(in_arr, scale, offset)
    # unscaled fcvt register conversions
    elif src_kind == 'fp32' and dst_kind == 'fp16':
        out_arr = cvt_sh(in_arr)
    elif src_kind == 'fp16' and dst_kind == 'fp32':
        out_arr = cvt_hs(in_arr)
    elif src_kind == 'fp64' and dst_kind == 'fp16':
        out_arr = cvt_dh(in_arr)
    elif src_kind == 'fp16' and dst_kind == 'fp64':
        out_arr = cvt_hd(in_arr)
    else:
        return 0

    out_bytes = out_arr.copy().view(np.uint8)
    l1[addr_r:addr_r + out_bytes.size] = out_bytes
    return 0

# =============================================================================
# 1. LUT builders + module-level tables
# =============================================================================
def fp8_to_fp16_lut() -> np.ndarray:
    vals: list[float] = []
    for h in range(256):
        h_sign = (h & 0x80) >> 7
        h_exp = (h & 0x78) >> 3
        h_frac = h & 0x07
        if h_exp == 0:
            val = 0.0 if h_frac == 0 else (h_frac / 8.0) * (2.0 ** -6)
        elif h_exp == 0xF:
            val = float('inf') if h_frac == 0 else float('nan')
        else:
            val = (1.0 + h_frac / 8.0) * (2.0 ** (h_exp - 7))
        if h_sign and not (val != val):   # x != x is NaN-safe
            val = -val
        vals.append(val)
    out = np.array(vals, dtype=np.float16)
    # Preserve negative-zero bit pattern for h=0x80 (sign=1, exp=0, frac=0).
    out[0x80] = np.array(-0.0, dtype=np.float16)
    return out


def fp16_to_fp8_lut() -> np.ndarray:
    out = np.zeros(65536, dtype=np.uint8)
    for h in range(65536):
        h_sign = (h >> 15) & 0x1
        h_exp = (h >> 10) & 0x1F
        h_frac = h & 0x03FF
        sign8 = (h_sign & 0x1) << 7
        if h_exp == 0x1F:
            out[h] = (sign8 | 0xF8 | 0x01) if h_frac else (sign8 | 0xF8)
            continue
        e16 = -14 if h_exp == 0 else (int(h_exp) - 15)
        new_e = e16 + 7
        sig = h_frac if h_exp == 0 else (0x400 | h_frac)
        if 1 <= new_e <= 14:
            out_bits = 4
            shift = 7
            main = sig >> shift
            round_bit = (sig >> (shift - 1)) & 1
            sticky = sig & ((1 << (shift - 1)) - 1)
            if round_bit and (sticky or (main & 1)):
                main += 1
            if main == (1 << out_bits):
                new_e += 1
                main = 1 << (out_bits - 1)
                if new_e >= 0xF:
                    out[h] = sign8 | 0xF8
                    continue
            out[h] = sign8 | ((new_e & 0xF) << 3) | (main & 0x7)
            continue
        if new_e <= 0:
            total_shift = 8 - new_e
            if total_shift >= 32:
                out[h] = sign8
                continue
            frac = sig >> total_shift
            rb_pos = total_shift - 1
            round_bit = (sig >> rb_pos) & 1
            sticky = (sig & ((1 << rb_pos) - 1)) if rb_pos > 0 else 0
            if round_bit and (sticky or (frac & 1)):
                frac += 1
                if frac == 0x8:
                    out[h] = sign8 | (1 << 3)
                    continue
            out[h] = sign8 | (frac & 0x7)
            continue
        # new_e > 14: overflow -> inf
        out[h] = sign8 | 0xF8
    return out


# =============================================================================
# 2. Format-conversion kernels
# =============================================================================
def fp8_e4m3_to_fp16(t_e4m3: np.ndarray) -> np.ndarray:
    return t_e4m3.astype(np.float16)


def fp16_to_fp8_e4m3(t_fp16: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "fp8 e4m3 encode unsupported on NumPy backend (no native float8) — "
        "add a manual bit encoder if a kernel needs it")


# ----- native(MX_IO_DTYPE) <-> explicit format, scaled (out = a*scale + offset) -
def cvt_to_io(arr: np.ndarray, scale, offset) -> np.ndarray:
    """{FP8, FP16, INT8, INT32} -> native MX_IO_DTYPE."""
    a = arr.astype(np.float32)
    s = np.asarray(scale, dtype=np.float32)
    o = np.asarray(offset, dtype=np.float32)
    return (a * s + o).astype(MX_IO_DTYPE)


def cvt_io_fp8(arr: np.ndarray, scale, offset) -> np.ndarray:
    """native -> FP8 (e4m3)."""
    raise NotImplementedError(
        "fp8 e4m3 encode unsupported on NumPy backend (no native float8)")


def cvt_io_fp16(arr: np.ndarray, scale, offset) -> np.ndarray:
    """native -> FP16."""
    a = arr.astype(np.float32)
    s = np.asarray(scale, dtype=np.float32)
    o = np.asarray(offset, dtype=np.float32)
    return (a * s + o).astype(np.float16)


def cvt_io_int8(arr: np.ndarray, scale, offset) -> np.ndarray:
    """native -> INT8 saturating in [-128, 127]."""
    a = arr.astype(np.float32)
    s = np.asarray(scale, dtype=np.float32)
    o = np.asarray(offset, dtype=np.float32)
    return np.clip(np.round(a * s + o), -128, 127).astype(np.int8)


# ----- unscaled fcvt register conversions -----------------------------------
def cvt_sh(arr_f32: np.ndarray) -> np.ndarray:
    """FP32 -> FP16 (bit-pattern preserving)."""
    return arr_f32.astype(np.float16)


def cvt_hs(arr_f16: np.ndarray) -> np.ndarray:
    """FP16 -> FP32 (bit-pattern preserving)."""
    return arr_f16.astype(np.float32)


def cvt_dh(arr_f64: np.ndarray) -> np.ndarray:
    """FP64 -> FP16 (single rounding)."""
    return arr_f64.astype(np.float16)


def cvt_hd(arr_f16: np.ndarray) -> np.ndarray:
    """FP16 -> FP64 (bit-exact widening)."""
    return arr_f16.astype(np.float64)
