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
"""VEC engine -- spike-bound dispatcher for firmware_vec_op.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:572-754.

Per CONTEXT D-01: spike-bound (reads npu/proc/insn). Pure VEC kernel
delegated to vec_core.py.

Per RESEARCH Pitfall 4: every C++ `p->get_state()->XPR[i]` becomes Python
`proc.state.XPR[i]` (P4 04-05 lock). Do NOT use `proc.get_state()`.

Per RESEARCH Pitfall 7: `vec_size = (rs1 & 0xFFFF) or 0x10000` (HW conv: 0 -> 65536).

Plan-extension note: in C++ funct7=0x10 SASMD scalar arith is dispatched via
`dispatch_iss_opcode` (separate path), not `firmware_vec_op`. In pyspike we
unify both behind a single Python entry point so @handler decorators at the
funct7 level can route uniformly. Behavior matches C++ semantics; only the
plumbing collapses to one function in Python.
"""
from __future__ import annotations

import numpy as np

from .vec_core import (
    sasmd_kernel,
    dot_kernel,
    vsum_kernel,
    clamp_min_kernel,
    clamp_max_kernel,
    accum_kernel,
    arange_kernel,
)
from .encoding import (
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRR,
    GTX_F7_VEC_SASMD, GTX_F7_VEC_DOT_SUM, GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP,
    GTX_VEC_ADD, GTX_VEC_SUB, GTX_VEC_MUL, GTX_VEC_DIV,
)
from .params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES


# =========================================================================
# Helpers
# =========================================================================
def _fp16_low16(packed: int) -> np.float16:
    """Decode bits[15:0] of an int as FP16 (LE bit-pattern)."""
    u16 = np.array([packed & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]


def _fp16_high16(packed: int) -> np.float16:
    """Decode bits[31:16] of an int as FP16 (LE bit-pattern)."""
    u16 = np.array([(packed >> 16) & 0xFFFF], dtype=np.uint16)
    return u16.view(np.float16)[0]


def _l1_view_addr(npu, nest: int, spu: int, addr_byte: int,
                   length: int) -> np.ndarray:
    """Return an FP16 view of L1[addr:addr+length*2] (no copy)."""
    l1_f16 = npu.mem.l1_f16(nest, spu)
    off = addr_byte // 2
    return l1_f16[off:off + length]


def _l0_block_view(npu, nest: int, spu: int, reg: int) -> np.ndarray:
    """Return an FP16 view of L0[(reg & 0x1F)*32 .. +32]; 16 FP16 elements."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
    # 32 bytes = 16 FP16. The view is a slice of l0; FP16 view via .view(np.float16)
    # only works on the full byte array, then sliced.
    return l0.view(np.float16)[off // 2:off // 2 + 16]


# =========================================================================
# Public entry: firmware_vec_op
# =========================================================================
def firmware_vec_op(npu, proc, insn) -> int:
    """Direct port of gtx_npu_vec.cc:572-754 (extended in pyspike to also
    cover the SASMD funct7=0x10 path that C++ routes via
    `dispatch_iss_opcode`).

    rs1 layout: vec_size = rs1[15:0] (single field; HW conv 0 -> 0x10000).
    funct3 = (xd<<2) | (xs1<<1) | xs2 (sub-op selector).
    rs2 -> staged into GSPR_OPERAND2 for CLAMP / ARANGE / SASMD scalar ops.
    """
    rs1 = int(proc.state.XPR[insn.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000  # Pitfall 7

    funct7 = insn.funct
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    # Pitfall G mirror (gtx_npu_vec.cc:582-585).
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0

    # Stage rs2 into GSPR_OPERAND2 (gtx_npu_vec.cc:736-737).
    rs2 = int(proc.state.XPR[insn.rs2])
    npu.gspr[GSPR_GTX_OPERAND2] = rs2

    # ---- SASMD scalar arith funct7=0x10 (VS at funct3=0..3, IS at 4..7) ----
    if funct7 == GTX_F7_VEC_SASMD:
        return _dispatch_sasmd(npu, nest, spu, funct3, rs1, rs2, insn,
                                vec_size)

    # ---- L0 II / L0 imm path (funct3 & 4) for funct7 in {0x18, 0x1C-0x1E} ----
    # gtx_npu_vec.cc:595-611 (ARITH II), :640-664 (MATH I), :674-687 (SIGN I),
    # :697-710 (ROUND I).
    if funct7 == GTX_F7_VEC_ARITH and (funct3 & 4):
        return _dispatch_arith_l0_ii(npu, nest, spu, funct3 & 3, rs1, rs2, insn)
    if funct7 in (0x1C, 0x1D, 0x1E) and (funct3 & 4):
        return _dispatch_unary_l0(npu, nest, spu, funct7, funct3 & 3, rs1, insn)

    # ---- L1 path ----
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    if funct7 == GTX_F7_VEC_ARITH:
        # SASMD VV: 4 ops add/sub/mul/div elementwise on L1.
        op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
        if (funct3 & 3) in op_map:
            view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            result = sasmd_kernel(view_a, view_b, op=op_map[funct3 & 3])
            _l1_view_addr(npu, nest, spu, addr_r, vec_size)[:] = result
            return 0

    if funct7 == GTX_F7_VEC_DOT_SUM:
        # gtx_npu_vec.cc:632-637: funct3=0 -> DOT, funct3=1 -> VSUM.
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if (funct3 & 3) == 0:
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            scalar = dot_kernel(view_a, view_b)
        else:  # funct3=1 (VSUM) -- default fallthrough also goes here per cc:636
            scalar = vsum_kernel(view_a)
        _l1_view_addr(npu, nest, spu, addr_r, 1)[0] = scalar
        # Also write L0 SVR[0] LE bytes (gtx_npu_vec.cc:108-110).
        # NB: in C++ MM_O writes BE, MM_V writes LE; for VEC the L0 dump is LE.
        u16_val = int(np.float16(scalar).view(np.uint16))
        l0 = npu.mem.l0_byte(nest, spu)
        l0[0] = u16_val & 0xFF
        l0[1] = (u16_val >> 8) & 0xFF
        return 0

    if funct7 == GTX_F7_VEC_CLAMP:
        # gtx_npu_vec.cc:719-726.
        sub = funct3 & 3
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if sub == 0:  # clamp_min_v
            scalar = _fp16_low16(rs2)
            result = clamp_min_kernel(view_a, scalar)
        elif sub == 1:  # clamp_max_v
            scalar = _fp16_low16(rs2)
            result = clamp_max_kernel(view_a, scalar)
        elif sub == 2:  # accum_v (prefix sum)
            result = accum_kernel(view_a)
        else:  # sub == 3 -> arange_v
            start = _fp16_low16(rs2)
            step = _fp16_high16(rs2)
            result = arange_kernel(vec_size, start, step)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size)[:] = result
        # funct3 4..7 (bitwise L0 ops) deferred -- not in P5 critical path.
        return 0

    # MATH/SIGN/ROUND L1 path (funct3 0..3): elementwise unary on L1.
    if funct7 in (0x1C, 0x1D, 0x1E):
        view = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = _apply_unary(funct7, funct3 & 3, view)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size)[:] = result
        return 0

    # Unknown funct7: silent NOP (mirrors C++ default fallthrough).
    return 0


# =========================================================================
# Sub-dispatchers
# =========================================================================
def _dispatch_sasmd(npu, nest: int, spu: int, funct3: int,
                     rs1: int, rs2: int, insn, vec_size: int) -> int:
    """SASMD funct7=0x10: VS (L1, funct3=0..3) and IS (L0, funct3=4..7).

    VS: read L1[ADDRA:ADDRA+vec_size*2] (FP16), broadcast scalar = rs2 low-16
        (FP16), compute, write L1[ADDRR].
    IS: read L0 reg = rs1 & 0x1F (16 FP16 = 32 bytes), broadcast scalar from
        rs2 low-16, compute, write L0 reg = (insn.rd & 0x1F) * 32.
    """
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    sub = funct3 & 3
    if sub not in op_map:
        return 0

    scalar = _fp16_low16(rs2)
    if not (funct3 & 4):
        # VS path: L1 scalar broadcast.
        addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = sasmd_kernel(view_a, scalar, op=op_map[sub])
        _l1_view_addr(npu, nest, spu, addr_r, vec_size)[:] = result
        return 0

    # IS path: L0 SVR scalar broadcast over 16 elements.
    # input reg = rs1 & 0x1F (gtx_npu_vec.cc:604), result reg = insn.rd & 0x1F
    # (or GSPR_OPERAND3 in some encodings -- here we follow exec_scalar_imm
    # signature which takes the result reg from op3 == GSPR_OPERAND3).
    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get(GSPR_GTX_OPERAND3, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    result = sasmd_kernel(view_a, scalar, op=op_map[sub])
    _l0_block_view(npu, nest, spu, r_reg)[:] = result
    return 0


def _dispatch_arith_l0_ii(npu, nest: int, spu: int, sub_op: int,
                            rs1: int, rs2: int, insn) -> int:
    """funct7=0x18, funct3 & 4: L0 II element-wise across two L0 SVR regs.

    gtx_npu_vec.cc:595-611: a_reg = rs1 & 0x1F, b_reg = rs2 & 0x1F,
    r_reg = gspr[GSPR_OPERAND3] & 0x1F.
    """
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    if sub_op not in op_map:
        return 0
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    r_reg = int(npu.gspr.get(GSPR_GTX_OPERAND3, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    view_b = _l0_block_view(npu, nest, spu, b_reg)
    result = sasmd_kernel(view_a, view_b, op=op_map[sub_op])
    _l0_block_view(npu, nest, spu, r_reg)[:] = result
    return 0


def _dispatch_unary_l0(npu, nest: int, spu: int, funct7: int, sub_op: int,
                        rs1: int, insn) -> int:
    """L0 unary path for funct7 ∈ {0x1C, 0x1D, 0x1E} when funct3 & 4 is set.

    Operates element-wise on a 16-FP16 L0 block. Input reg = rs1 & 0x1F;
    result reg = gspr[GSPR_OPERAND3] & 0x1F (or input reg when OPERAND3
    not yet set, per gtx_npu_vec.cc:659).
    """
    input_reg = rs1 & 0x1F
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub_op, view)
    _l0_block_view(npu, nest, spu, result_reg)[:] = result
    return 0


def _apply_unary(funct7: int, sub_op: int, view: np.ndarray) -> np.ndarray:
    """Element-wise unary kernels for funct7 0x1C/0x1D/0x1E (FP32 internal)."""
    f32 = view.astype(np.float32)
    if funct7 == 0x1C:  # MATH: sqrt/exp/log
        if sub_op == 0:
            return np.sqrt(f32).astype(np.float16)
        if sub_op == 1:
            return np.exp(f32).astype(np.float16)
        if sub_op == 2:
            # gtx_npu_vec.cc:142 -- log(a) only when a > 0; else 0.
            with np.errstate(invalid='ignore', divide='ignore'):
                out = np.where(f32 > 0.0, np.log(f32), 0.0)
            return out.astype(np.float16)
    if funct7 == 0x1D:  # SIGN: abs/neg/sign/step
        if sub_op == 0:
            return np.abs(f32).astype(np.float16)
        if sub_op == 1:
            return (-f32).astype(np.float16)
        if sub_op == 2:
            return np.sign(f32).astype(np.float16)
        if sub_op == 3:
            return (f32 > 0.0).astype(np.float16)  # step (Heaviside)
    if funct7 == 0x1E:  # ROUND: ceil/trunc/floor/rne
        if sub_op == 0:
            return np.ceil(f32).astype(np.float16)
        if sub_op == 1:
            return np.trunc(f32).astype(np.float16)
        if sub_op == 2:
            return np.floor(f32).astype(np.float16)
        if sub_op == 3:
            return np.rint(f32).astype(np.float16)
    # Unknown: pass-through.
    return view.copy()
