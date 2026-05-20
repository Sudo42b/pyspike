from __future__ import annotations

import torch

from ...inst_handler import inst_register
from ....csr import GSPR, LSPR, NSPR
from ... import _resolve_nest_spu
from . import _BYTES_PER_ELEM, _CVT_DTYPE_IN, _fp16_low16, _fp16_high16
# ----- format_cvt @handlers (7 directions including FP64) -------------------
# Direction is selected by ``GSPR_OPCODE & 1`` per gtx_npu_act.cc:245.

@inst_register.custom0(name='scvt.qh', funct7=0b0100000, funct3=0)
def _scvt_qh(npu, proc, inst, cxt) -> int:
    # scvt.qh	4'b0100	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	N/A	vector_size[23:0]	scale[15:0], offset[31:16]	N/A	r2_sel[8:0]	N/A	N/A	format conversion fp16 to fp8	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp8', dst_kind='fp16')
    return _format(npu, proc, inst,
                            src_kind='fp16', dst_kind='fp8')

# scvt.hq	4'b0100	3'b000	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	N/A	vector_size[23:0]	scale[15:0], offset[31:16]	N/A	r2_sel[8:0]	N/A	N/A	format conversion fp8 to fp16	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
@inst_register.custom0(name='scvt.hq', funct7=0b0100000, funct3=1)
def _scvt_hq(npu, proc, inst, cxt) -> int:
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp8', dst_kind='fp16')
    return _format(npu, proc, inst,
                            src_kind='fp16', dst_kind='fp8')

@inst_register.custom0(name='scvt.ih', funct7=0b0100001, funct3=0)
def _scvt_ih(npu, proc, inst, cxt) -> int:
    # scvt.ih	4'b0100	3'b001	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	N/A	vector_size[23:0]	scale[15:0], offset[31:16]	N/A	r2_sel[8:0]	N/A	N/A	format conversion fp16 to int8	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='int8', dst_kind='fp16')
    return _format(npu, proc, inst,
                            src_kind='fp16', dst_kind='int8')

@inst_register.custom0(name='scvt.hi', funct7=0b0100001, funct3=1)
def _scvt_hi(npu, proc, inst, cxt) -> int:
    # scvt.hi	4'b0100	3'b001	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	N/A	vector_size[23:0]	scale[15:0], offset[31:16]	N/A	r2_sel[8:0]	N/A	N/A	format conversion fp16 to int8	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    return 0

@inst_register.custom0(name='scvt.hn', funct7=0b0100010, funct3=1)
def _scvt_hn(npu, proc, inst, cxt) -> int:
    # scvt.hn	4'b0100	3'b010	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	N/A	vector_size[23:0]	scale[15:0], offset[31:16]	N/A	r2_sel[8:0]	N/A	N/A	format conversion int32 to fp16	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return _format(npu, proc, inst,
                            src_kind='int32', dst_kind='fp16')


@inst_register.custom0(name='fcvt.sh', funct7=0b0100100, funct3=0)
def _fcvt_sh(npu, proc, inst, cxt) -> int:
    # fcvt.sh	4'b0100	3'b100	rsvd	gpr	3'b000	gpr	gtx op	yes	no	nsu	1	N/A	input data[63:0]	N/A	N/A	N/A	result data[63:0]	N/A	format conversion fp16 to fp32	-
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp16', dst_kind='fp32')
    return _format(npu, proc, inst,
                            src_kind='fp32', dst_kind='fp16')

@inst_register.custom0(name='fcvt.hs', funct7=0b0100100, funct3=1)
def _fcvt_hs(npu, proc, inst, cxt) -> int:
    # fcvt.hs	4'b0100	3'b100	rsvd	gpr	3'b001	gpr	gtx op	yes	no	nsu	1	N/A	input data[63:0]	N/A	N/A	N/A	result data[63:0]	N/A	format conversion fp32 to fp16	-
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    return 0

@inst_register.custom0(name='fcvt.dh', funct7=0b0100101, funct3=0)
def _fcvt_dh(npu, proc, inst, cxt) -> int:
    # fcvt.dh	4'b0100	3'b101	rsvd	gpr	3'b000	gpr	gtx op	yes	no	nsu	1	N/A	input data[63:0]	N/A	N/A	N/A	result data[63:0]	N/A	format conversion fp16 to fp64	-
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    if sub_op & 1:
        return _format(npu, proc, inst,
                                src_kind='fp16', dst_kind='fp64')
    return _format(npu, proc, inst,
                            src_kind='fp64', dst_kind='fp16')


@inst_register.custom0(name='fcvt.hd', funct7=0b0100101, funct3=1)
def _fcvt_hd(npu, proc, inst, cxt) -> int:
    # fcvt.hd	4'b0100	3'b101	rsvd	gpr	3'b001	gpr	gtx op	yes	no	nsu	1	N/A	input data[63:0]	N/A	N/A	N/A	result data[63:0]	N/A	format conversion fp64 to fp16	-
    sub_op = int(npu.gspr.get(GSPR['GSPR_GTX_OPCODE'].address, 0)) & 0xFF
    return 0


def _format(npu, proc, inst, *, src_kind: str, dst_kind: str) -> int:
    """Direct port of ``gtx_npu_act.cc:222-372`` (``exec_format_cvt``)."""
    nest, spu = _resolve_nest_spu(npu)

    length = int(proc.state.XPR[inst.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    op2 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    scale = _fp16_low16(op2)
    offset = _fp16_high16(op2)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    l1 = npu.mem.l1_byte(nest, spu)
    in_size = length * _BYTES_PER_ELEM[src_kind]
    in_bytes = l1[addr_a:addr_a + in_size].clone().contiguous()
    in_arr = in_bytes.view(_CVT_DTYPE_IN[src_kind])

    if src_kind == 'fp16' and dst_kind == 'fp8':
        out_arr = cvt_qh(in_arr, scale, offset)
    elif src_kind == 'fp8' and dst_kind == 'fp16':
        out_arr = cvt_hq(in_arr, scale, offset)
    elif src_kind == 'fp16' and dst_kind == 'int8':
        out_arr = cvt_ih(in_arr, scale, offset)
    elif src_kind == 'int8' and dst_kind == 'fp16':
        out_arr = cvt_hi(in_arr, scale, offset)
    elif src_kind == 'int32' and dst_kind == 'fp16':
        out_arr = cvt_hn(in_arr, scale, offset)
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

    out_bytes = out_arr.contiguous().view(torch.uint8)
    l1[addr_r:addr_r + out_bytes.numel()] = out_bytes
    return 0

# =============================================================================
# 1. LUT builders + module-level tables
# =============================================================================
def fp8_to_fp16_lut() -> torch.Tensor:
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
    out = torch.tensor(vals, dtype=torch.float16)
    # Preserve negative-zero bit pattern for h=0x80 (sign=1, exp=0, frac=0).
    out[0x80] = torch.tensor(-0.0, dtype=torch.float16)
    return out


def fp16_to_fp8_lut() -> torch.Tensor:
    out = torch.zeros(65536, dtype=torch.uint8)
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
def fp8_e4m3_to_fp16(t_e4m3: torch.Tensor) -> torch.Tensor:
    return t_e4m3.to(torch.float16)


def fp16_to_fp8_e4m3(t_fp16: torch.Tensor) -> torch.Tensor:
    return t_fp16.to(torch.float8_e4m3fn)


def cvt_qh(arr_f16: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """FP16 -> FP8. ``a = a * scale + offset``."""
    a_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (a_f32 * s_f32 + o_f32).to(torch.float8_e4m3fn)


def cvt_hq(arr_f8: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """FP8 -> FP16. ``out = decoded * scale + offset``."""
    arr_f32 = arr_f8.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (arr_f32.to(torch.float16) * s_f32 + o_f32).to(torch.float16)


def cvt_ih(arr_f16: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """FP16 -> INT8 saturating in [-128, 127]. ``gtx_npu_act.cc:288-297``."""
    a_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return torch.clamp(torch.round(a_f32 * s_f32 + o_f32), -128, 127).to(torch.int8)


def cvt_hi(arr_f16: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """INT8 -> FP16. ``out = int8 * scale + offset``."""
    arr_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (arr_f32 * s_f32 + o_f32).to(torch.float16)


def cvt_hn(arr_i32: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """INT32 -> FP16 normalize. ``gtx_npu_act.cc:301-313``."""
    arr_f32 = arr_i32.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (arr_f32 * s_f32 + o_f32).to(torch.float16)


def cvt_sh(arr_f32: torch.Tensor) -> torch.Tensor:
    """FP32 -> FP16 (bit-pattern preserving)."""
    return arr_f32.to(torch.float16)


def cvt_hs(arr_f16: torch.Tensor) -> torch.Tensor:
    """FP16 -> FP32 (bit-pattern preserving)."""
    return arr_f16.to(torch.float32)


def cvt_dh(arr_f64: torch.Tensor) -> torch.Tensor:
    """FP64 -> FP16 (single rounding)."""
    return arr_f64.to(torch.float16)


def cvt_hd(arr_f16: torch.Tensor) -> torch.Tensor:
    """FP16 -> FP64 (bit-exact widening)."""
    return arr_f16.to(torch.float64)
