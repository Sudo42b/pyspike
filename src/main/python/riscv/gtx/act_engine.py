#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ACT engine -- single bundled engine for activations + pool + format_cvt + L0 imm.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc. Five entry points
mirror C++ exec_activation / exec_pooling / exec_format_cvt / exec_act_imm /
exec_softmax_imm.

Per CONTEXT D-02: single bundled engine. Per D-05/D-06: `is_reversed` is
explicit at @handler entry (D-05); engine receives it as keyword.
ACT_OPS_REVERSED frozenset in encoding.py is engine-internal consistency check only.

Per RESEARCH Pitfall 4: every C++ `p->get_state()->XPR[i]` becomes Python
`proc.state.XPR[i]` (P4 04-05 lock). Do NOT use `proc.get_state()`.

Pitfall 8: ESUM is `forward` (is_reversed=False) but writes a single FP16
scalar to L0 at offset `(GSPR_OPERAND3 & 0x1F) * 32` -- NOT to L1[ADDRR].

Plan 03 GREEN-fills firmware_act + firmware_act_imm + firmware_softmax_imm.
Plan 04 GREEN-fills firmware_pool + firmware_format.
"""
from __future__ import annotations

import numpy as np

from .act_core import relu, prelu, gelu, tanh_act, sigmoid, softmax, esum
from .encoding import (
    GTX_ACT_RELU, GTX_ACT_TANH, GTX_ACT_SOFTMAX, GTX_ACT_GELU,
    GTX_ACT_SIGMOID, GTX_ACT_PRELU, GTX_ACT_ESUM,
    ACT_OPS_REVERSED,
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from .params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES


# ============================================================================
# Helpers (mirror vec_engine helpers)
# ============================================================================
def _fp16_low16(packed: int) -> np.float16:
    """Decode bits[15:0] of an int as FP16 (LE bit-pattern)."""
    u16 = np.array([packed & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]


def _fp16_high16(packed: int) -> np.float16:
    """Decode bits[31:16] of an int as FP16 (LE bit-pattern)."""
    u16 = np.array([(packed >> 16) & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]


def _resolve_nest_spu(npu) -> tuple[int, int]:
    """Mirror gtx_npu_act.cc:29 (Pitfall G). Pick (nest, spu) from warp loop
    state, falling back to 0 when out of HW range."""
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
        result = tanh_act(view_in)
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
        result = tanh_act(view_in)
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
# Plan 04 fills these
# ============================================================================
def firmware_pool(npu, proc, insn, *, is_max: bool) -> int:
    """Plan 04 GREEN-fill. Source: gtx_npu_act.cc:166-220.

    Always forward direction (ADDRA -> ADDRR per CONTEXT D-08).
    Avg-pool: `avg += 0.0` canonicalises -0.0 -> +0.0 (line 211).
    """
    return 0


def firmware_format(npu, proc, insn, *, src_kind: str, dst_kind: str) -> int:
    """Plan 04 GREEN-fill. Source: gtx_npu_act.cc:222-372.

    src_kind/dst_kind in {'fp16', 'fp32', 'fp64', 'fp8', 'int8', 'int32'}.
    Always forward direction (ADDRA -> ADDRR per CONTEXT D-08).
    Scale/offset unpacked from GSPR_GTX_OPERAND2 (low 16 = scale, high 16 = offset).
    """
    return 0
