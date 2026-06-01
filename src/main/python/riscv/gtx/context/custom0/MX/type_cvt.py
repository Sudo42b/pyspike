from __future__ import annotations

import numpy as np

from ...inst_handler import inst_register
from ....config_params import MX_IO_DTYPE
from ....csr import GSPR, LSPR, NSPR
from ... import _resolve_nest_spu
from . import _BYTES_PER_ELEM, _CVT_DTYPE_IN, _io_low, _io_high

# ----- format_cvt handlers (SMM_ISA v2.0.0d, sheet "conversion") ------------
# Each scvt.* is ONE-directional and owns a distinct (funct7, funct3) slot — the
# 2.0.0d re-encoding dropped the old OPCODE&1 bidirectional packing. The MX
# native float is ``'io'`` = MX_IO_DTYPE (FP32 default / FP16 toggle). The scaled
# transform is ``out = src*scale + offset`` with scale[31:0]/offset[63:32] in
# OPERAND2 (width config-gated via _io_low/_io_high).
#
#   scvt.qs  0x20/0  io   -> fp8     (down-cast, fp32->fp8 quant)
#   scvt.hq  0x20/1  fp8  -> fp16    (up-cast)
#   scvt.is  0x21/0  io   -> int8    (quant)
#   scvt.bi  0x21/1  int8 -> bf16    (dequant to bf16)
#   scvt.sn  0x22/1  int32-> io      (int32 -> fp32 native)
#   scvt.bs  0x23/0  io   -> bf16    (down-cast)
#   scvt.hs  0x23/1  io   -> fp16    (down-cast)
#   scvt.hf  0x23/2  fp4  -> fp16    (up-cast)
#   scvt.si  0x23/3  int8 -> io      (dequant)
# fcvt.* (0x24/0x25) stay the unscaled FP16<->FP32 / FP16<->FP64 register
# conversions on the NSU path.

@inst_register.custom0(name='scvt.qs', funct7=0b0100000, funct3=0)
def _scvt_qs(npu, proc, inst, cxt) -> int:
    """fp32(native) -> fp8."""
    return _format(npu, proc, inst, src_kind='io', dst_kind='fp8')


@inst_register.custom0(name='scvt.hq', funct7=0b0100000, funct3=1)
def _scvt_hq(npu, proc, inst, cxt) -> int:
    """fp8 -> fp16."""
    return _format(npu, proc, inst, src_kind='fp8', dst_kind='fp16')


@inst_register.custom0(name='scvt.is', funct7=0b0100001, funct3=0)
def _scvt_is(npu, proc, inst, cxt) -> int:
    """fp32(native) -> int8."""
    return _format(npu, proc, inst, src_kind='io', dst_kind='int8')


@inst_register.custom0(name='scvt.bi', funct7=0b0100001, funct3=1)
def _scvt_bi(npu, proc, inst, cxt) -> int:
    """int8 -> bf16."""
    return _format(npu, proc, inst, src_kind='int8', dst_kind='bf16')


@inst_register.custom0(name='scvt.sn', funct7=0b0100010, funct3=1)
def _scvt_sn(npu, proc, inst, cxt) -> int:
    """int32 -> fp32(native)."""
    return _format(npu, proc, inst, src_kind='int32', dst_kind='io')


@inst_register.custom0(name='scvt.bs', funct7=0b0100011, funct3=0)
def _scvt_bs(npu, proc, inst, cxt) -> int:
    """fp32(native) -> bf16."""
    return _format(npu, proc, inst, src_kind='io', dst_kind='bf16')


@inst_register.custom0(name='scvt.hs', funct7=0b0100011, funct3=1)
def _scvt_hs(npu, proc, inst, cxt) -> int:
    """fp32(native) -> fp16."""
    return _format(npu, proc, inst, src_kind='io', dst_kind='fp16')


@inst_register.custom0(name='scvt.hf', funct7=0b0100011, funct3=0b010)
def _scvt_hf(npu, proc, inst, cxt) -> int:
    """fp4 -> fp16."""
    return _format(npu, proc, inst, src_kind='fp4', dst_kind='fp16')


@inst_register.custom0(name='scvt.si', funct7=0b0100011, funct3=0b011)
def _scvt_si(npu, proc, inst, cxt) -> int:
    """int8 -> fp32(native)."""
    return _format(npu, proc, inst, src_kind='int8', dst_kind='io')


# fcvt.* are NSU **scalar register** conversions (exec=nsu, one-directional per
# v2.0.0d): input data in rs1[63:0], result returned to rd[63:0] (the handler's
# return value becomes rd). Unscaled. Packed SIMD: widening fits 2/1 elements in
# the 64-bit lane; narrowing the same back. NOT L1 memory (that path is scvt.*).
@inst_register.custom0(name='fcvt.sh', funct7=0b0100100, funct3=0)
def _fcvt_sh(npu, proc, inst, cxt) -> int:
    """fp16 -> fp32: rs1[31:0] = 2x fp16 -> rd[63:0] = 2x fp32."""
    return _fcvt_reg(proc, inst, src=np.float16, dst=np.float32, n=2)


@inst_register.custom0(name='fcvt.hs', funct7=0b0100100, funct3=1)
def _fcvt_hs(npu, proc, inst, cxt) -> int:
    """fp32 -> fp16: rs1[63:0] = 2x fp32 -> rd[31:0] = 2x fp16."""
    return _fcvt_reg(proc, inst, src=np.float32, dst=np.float16, n=2)


@inst_register.custom0(name='fcvt.dh', funct7=0b0100101, funct3=0)
def _fcvt_dh(npu, proc, inst, cxt) -> int:
    """fp16 -> fp64: rs1[15:0] = 1x fp16 -> rd[63:0] = 1x fp64."""
    return _fcvt_reg(proc, inst, src=np.float16, dst=np.float64, n=1)


@inst_register.custom0(name='fcvt.hd', funct7=0b0100101, funct3=1)
def _fcvt_hd(npu, proc, inst, cxt) -> int:
    """fp64 -> fp16: rs1[63:0] = 1x fp64 -> rd[15:0] = 1x fp16."""
    return _fcvt_reg(proc, inst, src=np.float64, dst=np.float16, n=1)


def _fcvt_reg(proc, inst, *, src, dst, n: int) -> int:
    """NSU register float conversion: decode ``n`` ``src`` elements from the low
    bits of rs1, convert (unscaled) to ``dst``, return the packed little-endian
    bit pattern for rd."""
    raw = int(proc.state.XPR[inst.rs1]) & 0xFFFFFFFFFFFFFFFF
    sbits = np.dtype(src).itemsize * 8
    words = {16: np.uint16, 32: np.uint32, 64: np.uint64}[sbits]
    in_arr = np.array(
        [(raw >> (sbits * i)) & ((1 << sbits) - 1) for i in range(n)],
        dtype=words,
    ).view(src)
    ob = np.ascontiguousarray(in_arr.astype(dst)).view(np.uint8)
    res = 0
    for i, byte in enumerate(ob.tolist()):
        res |= int(byte) << (8 * i)
    return res


def _format(npu, proc, inst, *, src_kind: str, dst_kind: str) -> int:
    """Port of ``gtx_npu_act.cc:222-372`` (``exec_format_cvt``) for the v2.0.0d
    one-directional **scvt** set (SPU vector, scaled).

    Per SMM_ISA v2.0.0d (summary sheet: scvt ``in=R, res=A``) the input comes
    from the **R bank** (SPM_ADDRR) and the result is written to the **A bank**
    (SPM_ADDRA). The input array (``src_kind``) is decoded to FP32, the scaled
    transform ``out = a*scale + offset`` is applied (scale[31:0]/offset[63:32]
    from OPERAND2), then encoded to ``dst_kind``. ``'io'`` resolves to the MX
    native float (``MX_IO_DTYPE`` = FP32 default).
    """
    nest, spu = _resolve_nest_spu(npu)

    length = int(proc.state.XPR[inst.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    op2 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    # scale/offset widen with MX_IO_DTYPE: FP16 [15:0]/[31:16], FP32 [31:0]/[63:32].
    scale = np.float32(_io_low(op2))
    offset = np.float32(_io_high(op2))

    # v2.0.0d: scvt reads the R bank, writes the A bank (in=R, res=A).
    addr_in = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    addr_out = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)

    l1 = npu.mem.l1_byte(nest, spu)

    if src_kind == 'fp4':
        # 4-bit floats pack two elements per byte (low nibble first).
        in_size = (length + 1) // 2
        in_bytes = l1[addr_in:addr_in + in_size].copy()
        nibbles = np.empty(in_bytes.size * 2, dtype=np.uint8)
        nibbles[0::2] = in_bytes & 0x0F
        nibbles[1::2] = (in_bytes >> 4) & 0x0F
        in_f32 = _fp4_to_fp32(nibbles[:length])
    else:
        in_size = length * _BYTES_PER_ELEM[src_kind]
        in_bytes = l1[addr_in:addr_in + in_size].copy()
        in_arr = in_bytes.view(_CVT_DTYPE_IN[src_kind])
        if src_kind == 'fp8':
            in_f32 = _FP8_TO_FP16[in_arr].astype(np.float32)
        elif src_kind == 'bf16':
            in_f32 = _bf16_to_fp32(in_arr)
        else:
            in_f32 = in_arr.astype(np.float32)

    out_f32 = in_f32 * scale + offset
    out_arr = _encode_from_fp32(out_f32, dst_kind)

    out_bytes = np.ascontiguousarray(out_arr).view(np.uint8)
    l1[addr_out:addr_out + out_bytes.size] = out_bytes
    return 0


def _encode_from_fp32(f32: np.ndarray, dst_kind: str) -> np.ndarray:
    """Encode an FP32 array to ``dst_kind`` (the conversion output format)."""
    if dst_kind == 'io':
        return f32.astype(MX_IO_DTYPE)
    if dst_kind == 'fp16':
        return f32.astype(np.float16)
    if dst_kind == 'fp32':
        return f32.astype(np.float32)
    if dst_kind == 'fp64':
        return f32.astype(np.float64)
    if dst_kind == 'bf16':
        return _fp32_to_bf16(f32)
    if dst_kind == 'int8':
        return np.clip(np.round(f32), -128, 127).astype(np.int8)
    if dst_kind == 'int32':
        return np.clip(np.round(f32), -(2**31), 2**31 - 1).astype(np.int32)
    if dst_kind == 'fp8':
        raise NotImplementedError(
            "fp8 e4m3 encode unsupported on NumPy backend (no native float8) — "
            "add a manual bit encoder if a kernel needs it")
    raise ValueError(f"unknown conversion dst_kind {dst_kind!r}")


# =============================================================================
# 1. bf16 / fp4 codecs (NumPy has no native bf16 or fp4)
# =============================================================================
def _fp32_to_bf16(f32: np.ndarray) -> np.ndarray:
    """FP32 -> bf16 raw uint16 (round-to-nearest-even truncation of the low 16
    mantissa bits)."""
    u32 = np.ascontiguousarray(f32.astype(np.float32)).view(np.uint32).astype(np.uint64)
    rounding = ((u32 >> 16) & np.uint64(1)) + np.uint64(0x7FFF)
    return ((u32 + rounding) >> np.uint64(16)).astype(np.uint16)


def _bf16_to_fp32(u16: np.ndarray) -> np.ndarray:
    """bf16 raw uint16 -> FP32 (zero-extend the mantissa into the low 16 bits)."""
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32) if u32.flags['C_CONTIGUOUS'] else \
        np.ascontiguousarray(u32).view(np.float32)


# fp4 e2m1 (OCP MXFP4) magnitudes for the 8 unsigned codes.
_FP4_E2M1_MAG = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _fp4_to_fp32(nibbles: np.ndarray) -> np.ndarray:
    """fp4 e2m1 nibbles (uint8 0..15) -> FP32 (bit3 = sign)."""
    sign = (nibbles >> 3) & 0x1
    mag = _FP4_E2M1_MAG[nibbles & 0x7]
    return np.where(sign == 1, -mag, mag).astype(np.float32)


# =============================================================================
# 2. fp8 <-> fp16 LUTs (e4m3) — fp8 decode used by scvt.hq
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


# Module-level fp8 decode LUT (indexed by the raw uint8 fp8 byte).
_FP8_TO_FP16: np.ndarray = fp8_to_fp16_lut()
