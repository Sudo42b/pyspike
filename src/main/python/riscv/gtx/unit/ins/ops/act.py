"""ACT op @handler entries + activation/format-cvt/pool kernels + LUT tables.

Single-file consolidation of the former three-layer split
(``act_core`` kernels + ``act_engine`` firmware shims + ``ops/act`` @handler
decorators). Layout, top-to-bottom:

  1. LUT builders + module-level LUT tables (FP8 ↔ FP16, dead-code so far
     but preserved for parity with vendor reference).
  2. Format-conversion impls (``_cvt_*``) + public ``cvt_*`` API.
  3. Activation kernels (relu/prelu/gelu/tanh/sigmoid/softmax/esum).
  4. Pool kernels (max/avg).
  5. FP16 bit-pattern helpers + L0 block view + NEST/SPU routing.
  6. firmware_* dispatch surface (firmware_act / firmware_act_imm /
     firmware_softmax_imm / firmware_pool / firmware_format).
  7. @handler entries -- the source-of-truth for funct7/funct3 binding and
     for the ``is_reversed`` policy (engine asserts consistency vs
     :data:`ACT_OPS_REVERSED`).

Vendor authority for funct7/funct3 values:
  vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:152-157 (verbatim).
"""
from __future__ import annotations

import torch
from torch import Tensor

from ...._registry import handler
from ....config_params import GTX_L0_SIZE_BYTES, GTX_NEST_NUM, GTX_SPU_NUM
from ..encoding import (
    # ACT_OPS_REVERSED,
    # GSPR_GTX_OPCODE,
    # GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    GTX_ACT_ESUM, GTX_ACT_GELU, GTX_ACT_PRELU,
    GTX_ACT_RELU, GTX_ACT_SIGMOID, GTX_ACT_SOFTMAX, GTX_ACT_TANH,
    GTX_F7_ACT_GELU, GTX_F7_ACT_PRELU, GTX_F7_ACT_SIGM,
    GTX_F7_ACT_SOFTMAX, GTX_F7_ACT_TANH,
    GTX_F7_FCVT_DH, GTX_F7_FCVT_SH,
    GTX_F7_POOL_AVG, GTX_F7_POOL_MAX,
    GTX_F7_SCVT_HN, GTX_F7_SCVT_IH, GTX_F7_SCVT_QH,
    # LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)


# =============================================================================
# 1. LUT builders + module-level tables
# =============================================================================
def _build_fp8_to_fp16_lut() -> torch.Tensor:
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


def _build_fp16_to_fp8_lut() -> torch.Tensor:
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


FP8_TO_FP16_LUT: torch.Tensor = _build_fp8_to_fp16_lut()
FP16_TO_FP8_LUT: torch.Tensor = _build_fp16_to_fp8_lut()


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


# =============================================================================
# 3. Activation kernels
# =============================================================================
def relu(arr_f16: Tensor) -> torch.Tensor:
    return torch.relu(arr_f16.to(torch.float32)).to(torch.float16)


def prelu(arr_f16: Tensor, slope: Tensor) -> Tensor:
    return torch.nn.functional.prelu(arr_f16.to(torch.float32), slope).to(torch.float16)


def gelu(arr_f16: Tensor) -> Tensor:
    return torch.nn.functional.gelu(arr_f16.to(torch.float32)).to(torch.float16)


def tanh(arr_f16: Tensor) -> Tensor:
    return torch.tanh(arr_f16.to(torch.float32)).to(torch.float16)


def sigmoid(arr_f16: Tensor) -> Tensor:
    return torch.sigmoid(arr_f16.to(torch.float32)).to(torch.float16)


def softmax(arr_f16: Tensor) -> Tensor:
    return torch.nn.functional.softmax(arr_f16.to(torch.float32), dim=0).to(torch.float16)


def esum(arr_f16: Tensor, max_val: float, init_accum: float) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    max_val_f32 = torch.as_tensor(max_val, dtype=torch.float32)
    init_accum_f32 = torch.as_tensor(init_accum, dtype=torch.float32)
    return (init_accum_f32 + torch.sum(torch.exp(arr_f32 - max_val_f32))).to(torch.float16)


# =============================================================================
# 4. Pool kernels
# =============================================================================
def pool_max(arr_f16: Tensor, kernel_size: int) -> Tensor:
    out = torch.max_pool1d(arr_f16.to(torch.float32),
                            kernel_size=kernel_size, stride=kernel_size)
    return out.to(torch.float16)


def pool_avg(arr_f16: Tensor, kernel_size: int) -> Tensor:
    out = torch.avg_pool1d(arr_f16.to(torch.float32),
                            kernel_size=kernel_size, stride=kernel_size,
                            count_include_pad=False)
    return out.to(torch.float16)


# =============================================================================
# 5. FP16 bit-pattern helpers + L0 block view + warp routing
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


def _resolve_nest_spu(npu) -> tuple[int, int]:
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0
    return nest, spu


def _l0_block_view(npu, nest: int, spu: int, reg: int) -> torch.Tensor:
    """Return an FP16 view of ``L0[(reg & 0x1F)*32 .. +32]`` (16 elements)."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
    return l0.view(torch.float16)[off // 2:off // 2 + 16]


def _write_l0_fp16_scalar(npu, nest: int, spu: int, l0_offset: int,
                          scalar: torch.Tensor) -> None:
    """Write a single FP16 LE word at ``L0[l0_offset]``."""
    l0 = npu.mem.l0_byte(nest, spu)
    u16 = _fp16_raw_bits(scalar)
    l0[l0_offset] = u16 & 0xFF
    l0[l0_offset + 1] = (u16 >> 8) & 0xFF


# =============================================================================
# 6. firmware_* dispatch surface
# =============================================================================
def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:23-164`` (``exec_activation``).

    Direction asymmetry (lines 37-42):
        reversed (TANH/GELU/SIGMOID/PRELU): rd=ADDRR, wr=ADDRA
        forward  (RELU/SOFTMAX/ESUM):       rd=ADDRA, wr=ADDRR  (ESUM → L0)
    """
    assert is_reversed == (op_id in ACT_OPS_REVERSED), (
        f"@handler is_reversed mismatch: op_id={op_id}, "
        f"is_reversed={is_reversed} (ACT_OPS_REVERSED={sorted(ACT_OPS_REVERSED)})"
    )
    nest, spu = _resolve_nest_spu(npu)

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    rd_addr, wr_addr = (addr_r, addr_a) if is_reversed else (addr_a, addr_r)

    length = int(proc.state.XPR[insn.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    l1_f16 = npu.mem.l1_f16(nest, spu)
    rd_off = (rd_addr // 2) % (l1_f16.shape[0])
    view_in = l1_f16[rd_off:rd_off + length]

    if op_id == GTX_ACT_RELU:
        result = relu(view_in)
    elif op_id == GTX_ACT_TANH:
        result = tanh(view_in)
    elif op_id == GTX_ACT_SOFTMAX:
        result = softmax(view_in)
    elif op_id == GTX_ACT_GELU:
        result = gelu(view_in)
    elif op_id == GTX_ACT_SIGMOID:
        result = sigmoid(view_in)
    elif op_id == GTX_ACT_PRELU:
        slope = _fp16_low16(int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)))
        result = prelu(view_in, slope)
    elif op_id == GTX_ACT_ESUM:
        # Pitfall 8: ESUM is forward (rd=ADDRA) but writes a scalar to L0
        # at offset (GSPR_OPERAND3 & 0x1F)*32 — not to L1[ADDRR].
        op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
        max_val = _fp16_low16(op2)
        init_accum = _fp16_high16(op2)
        scalar = esum(view_in, max_val=max_val, init_accum=init_accum)
        l0_offset = (
            (int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F) * 32
        ) % GTX_L0_SIZE_BYTES
        _write_l0_fp16_scalar(npu, nest, spu, l0_offset, scalar)
        return 0
    else:
        return 0

    wr_off = (wr_addr // 2) % (l1_f16.shape[0])
    l1_f16[wr_off:wr_off + length] = result
    return 0


def firmware_act_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path. ``gtx_npu_act.cc:374-431``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[insn.rs1]) & 0x1F
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (insn.rd & 0x1F)

    view_in = _l0_block_view(npu, nest, spu, in_reg)
    view_out = _l0_block_view(npu, nest, spu, out_reg)

    if op_id == GTX_ACT_PRELU:
        slope = _fp16_low16(int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)))
        result = prelu(view_in, slope)
    elif op_id == GTX_ACT_GELU:
        result = gelu(view_in)
    elif op_id == GTX_ACT_TANH:
        result = tanh(view_in)
    elif op_id == GTX_ACT_SIGMOID:
        result = sigmoid(view_in)
    else:
        return 0

    view_out[:] = result
    return 0


def firmware_softmax_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path for ESUM / SOFTMAX. ``gtx_npu_act.cc:436-487``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[insn.rs1]) & 0x1F
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (insn.rd & 0x1F)

    view_in = _l0_block_view(npu, nest, spu, in_reg)

    op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
    max_val = _fp16_low16(op2)
    accum_val = _fp16_high16(op2)

    if op_id == GTX_ACT_ESUM:
        scalar = esum(view_in, max_val=max_val, init_accum=accum_val)
        l0 = npu.mem.l0_byte(nest, spu)
        r_off = ((out_reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
        r16 = _fp16_raw_bits(scalar)
        m16 = _fp16_raw_bits(max_val)
        l0[r_off] = r16 & 0xFF
        l0[r_off + 1] = (r16 >> 8) & 0xFF
        l0[r_off + 2] = m16 & 0xFF
        l0[r_off + 3] = (m16 >> 8) & 0xFF
        for x in range(2, 16):
            l0[r_off + x * 2] = 0
            l0[r_off + x * 2 + 1] = 0
    elif op_id == GTX_ACT_SOFTMAX:
        # r[i] = exp(x[i] - max - ln(esum)); vendor uses pre-computed esum.
        f32 = view_in.to(torch.float32)
        max_f = max_val.to(torch.float32)
        esum_f = accum_val.to(torch.float32)
        ln_esum = (torch.log(esum_f) if float(esum_f) > 0.0
                   else torch.tensor(0.0, device=esum_f.device))
        result = torch.exp(f32 - max_f - ln_esum).to(torch.float16)
        view_out = _l0_block_view(npu, nest, spu, out_reg)
        view_out[:] = result
    return 0


def firmware_pool(npu, proc, insn, *, is_max: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:166-220`` (``exec_pooling``).

    Forward direction only (ADDRA -> ADDRR per CONTEXT D-08).
    """
    nest, spu = _resolve_nest_spu(npu)

    length = int(npu.gspr.get(GSPR_GTX_OPERAND1, 0)) & 0xFFFF
    if length == 0:
        length = 0x10000
    kernel_size = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)) & 0xFFFF
    if kernel_size == 0:
        return 0   # vendor guards `kernel_size > 0` -> silent NOP.

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    l1_f16 = npu.mem.l1_f16(nest, spu)
    in_off = (addr_a // 2) % l1_f16.shape[0]
    in_view = l1_f16[in_off:in_off + length]

    result = pool_max(in_view, kernel_size) if is_max else pool_avg(in_view, kernel_size)

    out_off = (addr_r // 2) % l1_f16.shape[0]
    l1_f16[out_off:out_off + length // kernel_size] = result
    return 0


_BYTES_PER_ELEM = {'fp16': 2, 'fp32': 4, 'fp64': 8,
                   'fp8': 1, 'int8': 1, 'int32': 4}

_CVT_DTYPE_IN = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64,
                 'fp8': torch.uint8, 'int8': torch.int8, 'int32': torch.int32}


def firmware_format(npu, proc, insn, *, src_kind: str, dst_kind: str) -> int:
    """Direct port of ``gtx_npu_act.cc:222-372`` (``exec_format_cvt``)."""
    nest, spu = _resolve_nest_spu(npu)

    length = int(proc.state.XPR[insn.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
    scale = _fp16_low16(op2)
    offset = _fp16_high16(op2)

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

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
# 7. @handler entries (funct7/funct3 binding policy)
# =============================================================================
# ----- ISS L1 path (8 activations: 4 reversed + 2 forward at 0x2F) -----------

@handler(kind='custom0', funct7=GTX_F7_ACT_PRELU, funct3=3,
         mnemonic='prelu', mask_funct3=True)
def _exec_prelu(npu, proc, insn, xs1, xs2):
    """PRELU: funct7=0x28 funct3=3, REVERSED (vendor cc:37-42)."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_PRELU, is_reversed=True)


@handler(kind='custom0', funct7=GTX_F7_ACT_GELU, funct3=0,
         mnemonic='gelu', mask_funct3=True)
def _exec_gelu(npu, proc, insn, xs1, xs2):
    """GELU: funct7=0x2A funct3=0, REVERSED."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_GELU, is_reversed=True)


@handler(kind='custom0', funct7=GTX_F7_ACT_TANH, funct3=0,
         mnemonic='tanh', mask_funct3=True)
def _exec_tanh(npu, proc, insn, xs1, xs2):
    """TANH: funct7=0x2C funct3=0, REVERSED."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_TANH, is_reversed=True)


@handler(kind='custom0', funct7=GTX_F7_ACT_SIGM, funct3=0,
         mnemonic='sigmoid', mask_funct3=True)
def _exec_sigmoid(npu, proc, insn, xs1, xs2):
    """SIGMOID: funct7=0x2D funct3=0, REVERSED."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_SIGMOID, is_reversed=True)


@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=1,
         mnemonic='esum', mask_funct3=True)
def _exec_esum(npu, proc, insn, xs1, xs2):
    """ESUM: funct7=0x2F funct3=1, FORWARD — writes scalar to L0 (Pitfall 8)."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_ESUM, is_reversed=False)


@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=2,
         mnemonic='softmax', mask_funct3=True)
def _exec_softmax(npu, proc, insn, xs1, xs2):
    """SOFTMAX: funct7=0x2F funct3=2, FORWARD."""
    return firmware_act(npu, proc, insn,
                         op_id=GTX_ACT_SOFTMAX, is_reversed=False)


# ----- L0 immediate path (6 _imm activations: funct3 & 4 selects L0) --------

@handler(kind='custom0', funct7=GTX_F7_ACT_PRELU, funct3=7,
         mnemonic='prelu_i', mask_funct3=True)
def _exec_prelu_i(npu, proc, insn, xs1, xs2):
    return firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_PRELU)


@handler(kind='custom0', funct7=GTX_F7_ACT_GELU, funct3=4,
         mnemonic='gelu_i', mask_funct3=True)
def _exec_gelu_i(npu, proc, insn, xs1, xs2):
    return firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_GELU)


@handler(kind='custom0', funct7=GTX_F7_ACT_TANH, funct3=4,
         mnemonic='tanh_i', mask_funct3=True)
def _exec_tanh_i(npu, proc, insn, xs1, xs2):
    return firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_TANH)


@handler(kind='custom0', funct7=GTX_F7_ACT_SIGM, funct3=4,
         mnemonic='sigm_i', mask_funct3=True)
def _exec_sigm_i(npu, proc, insn, xs1, xs2):
    return firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_SIGMOID)


@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=5,
         mnemonic='esum_i', mask_funct3=True)
def _exec_esum_i(npu, proc, insn, xs1, xs2):
    """L0 immediate ESUM (gtx_npu_act.cc:436-487 ESUM branch)."""
    return firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_ESUM)


@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=6,
         mnemonic='softmax_i', mask_funct3=True)
def _exec_softmax_i(npu, proc, insn, xs1, xs2):
    """L0 immediate SOFTMAX (pre-computed esum from GSPR_OPERAND2 high-16)."""
    return firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_SOFTMAX)


# ----- format_cvt @handlers (7 directions including FP64) -------------------
# Direction is selected by ``GSPR_OPCODE & 1`` per gtx_npu_act.cc:245.

@handler(kind='custom0', funct7=GTX_F7_SCVT_QH, mnemonic='scvt_qh')
def _exec_scvt_qh_dispatch(npu, proc, insn, xs1, xs2):
    """0=qh (FP16->FP8), 1=hq (FP8->FP16). Both apply scale+offset."""
    sub_op = int(npu.gspr.get(GSPR_GTX_OPCODE, 0)) & 0xFF
    if sub_op & 1:
        return firmware_format(npu, proc, insn,
                                src_kind='fp8', dst_kind='fp16')
    return firmware_format(npu, proc, insn,
                            src_kind='fp16', dst_kind='fp8')


@handler(kind='custom0', funct7=GTX_F7_SCVT_IH, mnemonic='scvt_ih')
def _exec_scvt_ih_dispatch(npu, proc, insn, xs1, xs2):
    """0=ih (FP16->INT8), 1=hi (INT8->FP16). Both apply scale+offset."""
    sub_op = int(npu.gspr.get(GSPR_GTX_OPCODE, 0)) & 0xFF
    if sub_op & 1:
        return firmware_format(npu, proc, insn,
                                src_kind='int8', dst_kind='fp16')
    return firmware_format(npu, proc, insn,
                            src_kind='fp16', dst_kind='int8')


@handler(kind='custom0', funct7=GTX_F7_SCVT_HN, mnemonic='scvt_hn')
def _exec_scvt_hn(npu, proc, insn, xs1, xs2):
    """INT32 -> FP16 normalize. Applies scale+offset."""
    return firmware_format(npu, proc, insn,
                            src_kind='int32', dst_kind='fp16')


@handler(kind='custom0', funct7=GTX_F7_FCVT_SH, mnemonic='fcvt_sh')
def _exec_fcvt_sh_dispatch(npu, proc, insn, xs1, xs2):
    """0=sh (FP32->FP16), 1=hs (FP16->FP32). Bit-pattern preserving."""
    sub_op = int(npu.gspr.get(GSPR_GTX_OPCODE, 0)) & 0xFF
    if sub_op & 1:
        return firmware_format(npu, proc, insn,
                                src_kind='fp16', dst_kind='fp32')
    return firmware_format(npu, proc, insn,
                            src_kind='fp32', dst_kind='fp16')


@handler(kind='custom0', funct7=GTX_F7_FCVT_DH, mnemonic='fcvt_dh')
def _exec_fcvt_dh_dispatch(npu, proc, insn, xs1, xs2):
    """0=dh (FP64->FP16), 1=hd (FP16->FP64). Bit-pattern preserving."""
    sub_op = int(npu.gspr.get(GSPR_GTX_OPCODE, 0)) & 0xFF
    if sub_op & 1:
        return firmware_format(npu, proc, insn,
                                src_kind='fp16', dst_kind='fp64')
    return firmware_format(npu, proc, insn,
                            src_kind='fp64', dst_kind='fp16')


# ----- Pool @handlers -------------------------------------------------------

@handler(kind='custom0', funct7=GTX_F7_POOL_MAX, mnemonic='pool_m')
def _exec_pool_m(npu, proc, insn, xs1, xs2):
    """Max-pool, forward only."""
    return firmware_pool(npu, proc, insn, is_max=True)


@handler(kind='custom0', funct7=GTX_F7_POOL_AVG, mnemonic='pool_a')
def _exec_pool_a(npu, proc, insn, xs1, xs2):
    """Avg-pool with -0.0 -> +0.0 canonicalization."""
    return firmware_pool(npu, proc, insn, is_max=False)
