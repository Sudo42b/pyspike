from __future__ import annotations

import torch
from torch import Tensor

from .act_core import (
    relu, prelu, gelu, tanh, sigmoid, softmax, esum,
    pool_max, pool_avg,
    cvt_qh, cvt_hq, cvt_ih, cvt_hi, cvt_hn, cvt_sh, cvt_hs, cvt_dh, cvt_hd,
)
from .encoding import (
    GTX_ACT_RELU, GTX_ACT_TANH, GTX_ACT_SOFTMAX, GTX_ACT_GELU,
    GTX_ACT_SIGMOID, GTX_ACT_PRELU, GTX_ACT_ESUM,
    ACT_OPS_REVERSED,
    GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from .params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES


# ============================================================================
# Helpers
# ============================================================================
def _fp16_low16(packed_f16: torch.Tensor) -> torch.Tensor:
    """Decode bits[15:0] of an int as FP16 (LE bit-pattern)."""
    # TODO!
    u16 = packed_f16 & 0xFFFF
    return u16[0]
    
def _fp16_high16(packed: torch.Tensor) -> torch.Tensor:
    # TODO!
    """Decode bits[31:16] of an int as FP16 (LE bit-pattern)."""
    u16 = torch.tensor([(packed >> 16) & 0xFFFF], dtype=torch.uint16)
    return u16.view(torch.float16)[0]


def _resolve_nest_spu(npu) -> tuple[int, int]:
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0
    return nest, spu


def _l0_block_view(npu, nest: int, spu: int, reg: int) -> np.ndarray:
    """Return an FP16 view of L0[(reg & 0x1F)*32 .. +32]; 16 FP16 elements."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
    return l0.view(np.float16)[off // 2:off // 2 + 16]


# ============================================================================
# firmware_act -- Plan 03 GREEN
# ============================================================================
def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of gtx_npu_act.cc:23-164 (exec_activation).

    Direction asymmetry (lines 37-42):
      reversed (TANH/GELU/SIGMOID/PRELU): rd=ADDRR, wr=ADDRA
      forward  (RELU/SOFTMAX/ESUM):       rd=ADDRA, wr=ADDRR (or L0 for ESUM)

    insn.rs1 carries `length` in low 16 bits (HW conv 0 -> 0x10000).
    Per CONTEXT D-06: ACT_OPS_REVERSED is the engine-internal consistency check
    against the @handler-entry `is_reversed` literal (D-05 source-of-truth).
    """
    # CONSISTENCY CHECK (D-06): op_id must agree with @handler is_reversed claim.
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
        length = 0x10000  # HW conv (mirrors firmware_vec_op)

    l1_f16 = npu.mem.l1_f16(nest, spu)
    rd_off = (rd_addr // 2) % (l1_f16.shape[0])
    view_in = l1_f16[rd_off:rd_off + length]

    # Per-op dispatch (gtx_npu_act.cc:59-158 switch).
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
        # Slope from GSPR_OPERAND2 low-16 (gtx_npu_act.cc:122).
        slope = _fp16_low16(int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)))
        result = prelu(view_in, slope)
    elif op_id == GTX_ACT_ESUM:
        # Pitfall 8: ESUM is forward (rd=ADDRA), but writes scalar to L0 at
        # offset (GSPR_OPERAND3 & 0x1F)*32, NOT to L1[ADDRR].
        # gtx_npu_act.cc:133-148.
        op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
        max_val = _fp16_low16(op2)
        init_accum = _fp16_high16(op2)
        scalar = esum(view_in, max_val=max_val, init_accum=init_accum)
        l0_offset = (int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F) * 32
        l0_offset = l0_offset % GTX_L0_SIZE_BYTES
        l0 = npu.mem.l0_byte(nest, spu)
        u16_val = int(np.float16(scalar).view(np.uint16))
        l0[l0_offset]     = u16_val & 0xFF
        l0[l0_offset + 1] = (u16_val >> 8) & 0xFF
        return 0  # ESUM does NOT touch L1[ADDRR]
    else:
        # Vendor default fallthrough is RELU (gtx_npu_act.cc:150-157), but in
        # pyspike unknown op_ids are dispatched only via @handler so we never
        # hit this branch from production code. Silent NOP for safety.
        return 0

    # Forward/reversed both write the array result to wr_addr (except ESUM above).
    wr_off = (wr_addr // 2) % (l1_f16.shape[0])
    l1_f16[wr_off:wr_off + length] = result
    return 0


# ============================================================================
# firmware_act_imm -- Plan 03 GREEN (PRELU/GELU/TANH/SIGM L0 immediate path)
# ============================================================================
def firmware_act_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path -- PRELU/GELU/TANH/SIGM operate on L0 SVR registers.

    Source: gtx_npu_act.cc:374-431 (exec_act_imm).

    Per RESEARCH Adjustment 3: L0 path uses explicit (input_reg, result_reg) --
    no ADDRA/ADDRR involvement. Always 16 FP16 elements per L0 register block.
    Direction is moot at byte level (rd_l0 -> wr_l0); we keep the @handler
    is_reversed=True for documentation consistency, but engine ignores it here.

    Param packing (gtx_npu_act.cc:381):
      input_reg = insn.rs1 & 0x1F
      result_reg = insn.rd & 0x1F  (or GSPR_OPERAND3 if upstream sets it)
      param (slope for PRELU) = GSPR_OPERAND2 low-16 FP16
    """
    nest, spu = _resolve_nest_spu(npu)

    # input_reg comes from XPR[insn.rs1] low-5 bits (vec_engine.cc:604 lineage:
    # `a_reg = rs1 & 0x1F` where rs1 is the value read from XPR).
    rs1_val = int(proc.state.XPR[insn.rs1])
    in_reg = rs1_val & 0x1F
    # Vendor exec_act_imm takes result_reg as a parameter; we mirror the
    # vec_engine convention: prefer GSPR_OPERAND3 if upstream set it, else
    # fall back to insn.rd (P4 D-04 / P5-02 lineage).
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    if op3_raw <= 0x1F:
        out_reg = op3_raw & 0x1F
    else:
        out_reg = insn.rd & 0x1F

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
# firmware_softmax_imm -- Plan 03 GREEN (ESUM/SOFTMAX L0 immediate path)
# ============================================================================
def firmware_softmax_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path -- ESUM/SOFTMAX. Source: gtx_npu_act.cc:436-487.

    16-element L0 reg block. ESUM writes scalar (16-bit FP16) at result_reg
    offset 0 (and max_val at offset+2 + zeros up to 16, per vendor lines
    470-474). SOFTMAX writes 16 FP16 results.
    """
    nest, spu = _resolve_nest_spu(npu)

    # input_reg from XPR[insn.rs1] low-5 (vec_engine.cc:604 lineage).
    rs1_val = int(proc.state.XPR[insn.rs1])
    in_reg = rs1_val & 0x1F
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    if op3_raw <= 0x1F:
        out_reg = op3_raw & 0x1F
    else:
        out_reg = insn.rd & 0x1F

    view_in = _l0_block_view(npu, nest, spu, in_reg)

    op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
    max_val = _fp16_low16(op2)
    accum_val = _fp16_high16(op2)

    if op_id == GTX_ACT_ESUM:
        scalar = esum(view_in, max_val=max_val, init_accum=accum_val)
        # gtx_npu_act.cc:471-474: store [r:16 | max:16] LE pair, then zero out
        # the rest of the 16-FP16 block.
        l0 = npu.mem.l0_byte(nest, spu)
        r_off = ((out_reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
        r16 = int(np.float16(scalar).view(np.uint16))
        m16 = int(np.float16(max_val).view(np.uint16))
        l0[r_off]     = r16 & 0xFF
        l0[r_off + 1] = (r16 >> 8) & 0xFF
        l0[r_off + 2] = m16 & 0xFF
        l0[r_off + 3] = (m16 >> 8) & 0xFF
        # Zero out remaining 14 FP16 slots (28 bytes).
        for x in range(2, 16):
            l0[r_off + x * 2]     = 0
            l0[r_off + x * 2 + 1] = 0
    elif op_id == GTX_ACT_SOFTMAX:
        # gtx_npu_act.cc:476-483: r[i] = exp(x[i] - max - ln(esum)).
        # Note: vendor uses pre-computed esum (passed as accum_val) here, NOT
        # a recomputed one. Different from L1 SOFTMAX (which computes its own
        # sum). We mirror the vendor exactly.
        f32 = view_in.astype(np.float32)
        max_f = np.float32(max_val)
        esum_f = np.float32(accum_val)
        ln_esum = np.log(esum_f) if esum_f > np.float32(0.0) else np.float32(0.0)
        result = np.exp(f32 - max_f - ln_esum).astype(np.float16)
        view_out = _l0_block_view(npu, nest, spu, out_reg)
        view_out[:] = result
    return 0


# ============================================================================
# firmware_pool -- Plan 04 GREEN
# ============================================================================
def firmware_pool(npu, proc, insn, *, is_max: bool) -> int:
    """Direct port of gtx_npu_act.cc:166-220 (exec_pooling) +
    gtx_npu_dispatch.cc:653-655/673-675 (firmware dispatch).

    Always forward direction (ADDRA -> ADDRR per CONTEXT D-08).
    length      = GSPR_GTX_OPERAND1 & 0xFFFF (HW conv 0 -> 0x10000)
    kernel_size = GSPR_GTX_OPERAND2 & 0xFFFF
    output_len  = length / kernel_size (integer floor; non-overlapping windows)

    Avg-pool: `avg += 0.0` canonicalises -0.0 -> +0.0 (line 211); the
    canonicalization happens inside act_core.pool_avg.
    """
    nest, spu = _resolve_nest_spu(npu)

    length = int(npu.gspr.get(GSPR_GTX_OPERAND1, 0)) & 0xFFFF
    if length == 0:
        length = 0x10000  # HW conv (mirrors firmware_vec_op / firmware_act)
    kernel_size = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0)) & 0xFFFF
    if kernel_size == 0:
        # Vendor guards `kernel_size > 0` at line 175; division by 0 is UB.
        # Match vendor by silently NOPing (no engine writeback when guard fails).
        return 0

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    l1_f16 = npu.mem.l1_f16(nest, spu)
    addra_off = (addr_a // 2) % l1_f16.shape[0]
    addrr_off = (addr_r // 2) % l1_f16.shape[0]

    in_view = l1_f16[addra_off:addra_off + length]

    if is_max:
        result = pool_max(in_view, kernel_size)
    else:
        result = pool_avg(in_view, kernel_size)

    out_len = length // kernel_size
    l1_f16[addrr_off:addrr_off + out_len] = result
    return 0


# ============================================================================
# firmware_format -- Plan 04 GREEN
# ============================================================================
# Bytes-per-element table for src/dst kinds (gtx_npu_act.cc:248-360).
_BYTES_PER_ELEM = {'fp16': 2, 'fp32': 4, 'fp64': 8,
                   'fp8': 1, 'int8': 1, 'int32': 4}


def firmware_format(npu, proc, insn, *, src_kind: str, dst_kind: str) -> int:
    """Direct port of gtx_npu_act.cc:222-372 (exec_format_cvt).

    src_kind/dst_kind in {'fp16', 'fp32', 'fp64', 'fp8', 'int8', 'int32'}.
    Always forward direction (ADDRA -> ADDRR per CONTEXT D-08).

    Scale/offset unpacked from GSPR_GTX_OPERAND2 (Pitfall 6 lock):
      scale  = OP2 & 0xFFFF        (FP16 LE, low 16 bits)
      offset = (OP2 >> 16) & 0xFFFF (FP16 LE, high 16 bits)

    scale/offset ARE applied for FP16<->{FP8, INT8, INT32};
    scale/offset are NOT applied for FP16<->{FP32, FP64} (bit-pattern preserving).

    length: read from XPR[insn.rs1] & 0xFFFF (HW conv 0 -> 0x10000).
    """
    nest, spu = _resolve_nest_spu(npu)

    # length from rs1 register value (mirrors firmware_vec_op).
    length = int(proc.state.XPR[insn.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    op2 = int(npu.gspr.get(GSPR_GTX_OPERAND2, 0))
    scale = _fp16_low16(op2)    # bits[15:0]  -- Pitfall 6
    offset = _fp16_high16(op2)  # bits[31:16]

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    l1 = npu.mem.l1_byte(nest, spu)

    in_size = length * _BYTES_PER_ELEM[src_kind]
    in_bytes = l1[addr_a:addr_a + in_size]

    if src_kind == 'fp16' and dst_kind == 'fp8':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float16)
        out_bytes = cvt_qh(in_arr, scale, offset).tobytes()
    elif src_kind == 'fp8' and dst_kind == 'fp16':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.uint8)
        out_bytes = cvt_hq(in_arr, scale, offset).tobytes()
    elif src_kind == 'fp16' and dst_kind == 'int8':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float16)
        out_bytes = cvt_ih(in_arr, scale, offset).tobytes()
    elif src_kind == 'int8' and dst_kind == 'fp16':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.int8)
        out_bytes = cvt_hi(in_arr, scale, offset).tobytes()
    elif src_kind == 'int32' and dst_kind == 'fp16':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.int32)
        out_bytes = cvt_hn(in_arr, scale, offset).tobytes()
    elif src_kind == 'fp32' and dst_kind == 'fp16':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float32)
        out_bytes = cvt_sh(in_arr).tobytes()  # NO scale/offset
    elif src_kind == 'fp16' and dst_kind == 'fp32':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float16)
        out_bytes = cvt_hs(in_arr).tobytes()
    elif src_kind == 'fp64' and dst_kind == 'fp16':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float64)
        out_bytes = cvt_dh(in_arr).tobytes()
    elif src_kind == 'fp16' and dst_kind == 'fp64':
        in_arr = np.frombuffer(bytes(in_bytes), dtype=np.float16)
        out_bytes = cvt_hd(in_arr).tobytes()
    else:
        return 0  # unsupported direction -- silent NOP

    out_arr = np.frombuffer(out_bytes, dtype=np.uint8)
    l1[addr_r:addr_r + len(out_arr)] = out_arr
    return 0
