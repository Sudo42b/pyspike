"""Activation / softmax / pool / format-conversion dispatcher.

PyTorch-native rewrite of ``gtx_npu_act.cc`` firmware entry points.
All ``act_core`` kernels already operate on :class:`torch.Tensor`; this
module wraps the L1/L0 read+write surface for the @handler layer.
"""
from __future__ import annotations

import torch

from .act_core import (
    relu, prelu, gelu, tanh, sigmoid, softmax, esum,
    pool_max, pool_avg,
    cvt_qh, cvt_hq, cvt_ih, cvt_hi, cvt_hn, cvt_sh, cvt_hs, cvt_dh, cvt_hd,
)
from .encoding import (
    GTX_ACT_RELU, GTX_ACT_TANH, GTX_ACT_SOFTMAX, GTX_ACT_GELU,
    GTX_ACT_SIGMOID, GTX_ACT_PRELU, GTX_ACT_ESUM,
    ACT_OPS_REVERSED,
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from ...config_params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES


# ============================================================================
# Helpers — FP16 bit-pattern decode, L0 block view, warp routing.
# ============================================================================

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
    l0[l0_offset]     = u16 & 0xFF
    l0[l0_offset + 1] = (u16 >> 8) & 0xFF


# ============================================================================
# firmware_act — L1 path (RELU/TANH/GELU/SIGM/SOFTMAX/PRELU/ESUM).
# ============================================================================
def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:23-164`` (``exec_activation``).

    Direction asymmetry (lines 37-42)
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
        l0_offset = ((int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F) * 32) % GTX_L0_SIZE_BYTES
        _write_l0_fp16_scalar(npu, nest, spu, l0_offset, scalar)
        return 0
    else:
        return 0

    wr_off = (wr_addr // 2) % (l1_f16.shape[0])
    l1_f16[wr_off:wr_off + length] = result
    return 0


# ============================================================================
# firmware_act_imm — L0 immediate path (PRELU/GELU/TANH/SIGM).
# ============================================================================
def firmware_act_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path — operates on L0 SVR registers (16 FP16 each).

    Source: ``gtx_npu_act.cc:374-431``.
    """
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


# ============================================================================
# firmware_softmax_imm — L0 immediate path (ESUM/SOFTMAX).
# ============================================================================
def firmware_softmax_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path for ESUM / SOFTMAX.

    Source: ``gtx_npu_act.cc:436-487``. Each L0 reg block is 16 FP16
    elements. ESUM stores ``[result_fp16, max_fp16]`` LE at offset 0 with
    zero padding; SOFTMAX writes 16 FP16 results.
    """
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
        l0[r_off]     = r16 & 0xFF
        l0[r_off + 1] = (r16 >> 8) & 0xFF
        l0[r_off + 2] = m16 & 0xFF
        l0[r_off + 3] = (m16 >> 8) & 0xFF
        for x in range(2, 16):
            l0[r_off + x * 2]     = 0
            l0[r_off + x * 2 + 1] = 0
    elif op_id == GTX_ACT_SOFTMAX:
        # r[i] = exp(x[i] - max - ln(esum));  vendor uses pre-computed esum.
        f32 = view_in.to(torch.float32)
        max_f = max_val.to(torch.float32)
        esum_f = accum_val.to(torch.float32)
        ln_esum = (torch.log(esum_f) if float(esum_f) > 0.0
                   else torch.tensor(0.0, device=esum_f.device))
        result = torch.exp(f32 - max_f - ln_esum).to(torch.float16)
        view_out = _l0_block_view(npu, nest, spu, out_reg)
        view_out[:] = result
    return 0


# ============================================================================
# firmware_pool — max / avg pool with non-overlapping windows.
# ============================================================================
def firmware_pool(npu, proc, insn, *, is_max: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:166-220`` (``exec_pooling``).

    Forward direction only (ADDRA → ADDRR per CONTEXT D-08).
    """
    from .encoding import GSPR_GTX_OPERAND1   # local — used only here
    nest, spu = _resolve_nest_spu(npu)

    length = int(npu.gspr.get(GSPR_GTX_OPERAND1, 0)) & 0xFFFF
    if length == 0:
        length = 0x10000
    kernel_size = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)) & 0xFFFF
    if kernel_size == 0:
        return 0   # vendor guards `kernel_size > 0` → silent NOP on miss.

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    l1_f16 = npu.mem.l1_f16(nest, spu)
    in_view = l1_f16[(addr_a // 2) % l1_f16.shape[0]:
                     (addr_a // 2) % l1_f16.shape[0] + length]

    result = pool_max(in_view, kernel_size) if is_max else pool_avg(in_view, kernel_size)

    out_off = (addr_r // 2) % l1_f16.shape[0]
    l1_f16[out_off:out_off + length // kernel_size] = result
    return 0


# ============================================================================
# firmware_format — format conversion (FP16 ↔ {FP8, INT8, INT32, FP32, FP64}).
# ============================================================================
_BYTES_PER_ELEM = {'fp16': 2, 'fp32': 4, 'fp64': 8,
                   'fp8': 1, 'int8': 1, 'int32': 4}

_CVT_DTYPE_IN = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64,
                 'fp8': torch.uint8, 'int8': torch.int8, 'int32': torch.int32}


def firmware_format(npu, proc, insn, *, src_kind: str, dst_kind: str) -> int:
    """Direct port of ``gtx_npu_act.cc:222-372`` (``exec_format_cvt``).

    ``src_kind`` / ``dst_kind`` ∈ ``{'fp16','fp32','fp64','fp8','int8','int32'}``.
    Forward direction only (ADDRA → ADDRR per CONTEXT D-08).
    """
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

    # Reinterpret the output tensor as raw bytes and copy into L1.
    out_bytes = out_arr.contiguous().view(torch.uint8)
    l1[addr_r:addr_r + out_bytes.numel()] = out_bytes
    return 0
