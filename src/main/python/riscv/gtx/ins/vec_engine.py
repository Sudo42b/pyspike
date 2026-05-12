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

Migrated 2026-05-11: numpy → torch. Memory storage is torch CPU tensors;
compute kernels use torch ops (CUDA-optional via vec_core).

Per RESEARCH Pitfall 7: `vec_size = (rs1 & 0xFFFF) or 0x10000` (HW conv: 0 -> 65536).
"""
from __future__ import annotations

import torch

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
def _fp16_low16(packed: int) -> torch.Tensor:
    """Decode bits[15:0] of an int as FP16 (LE bit-pattern), 0-d tensor."""
    u16 = torch.tensor([packed & 0xFFFF], dtype=torch.uint16)
    return u16.view(torch.float16)[0]


def _fp16_high16(packed: int) -> torch.Tensor:
    """Decode bits[31:16] of an int as FP16 (LE bit-pattern), 0-d tensor."""
    u16 = torch.tensor([(packed >> 16) & 0xFFFF], dtype=torch.uint16)
    return u16.view(torch.float16)[0]


def _l1_view_addr(npu, nest: int, spu: int, addr_byte: int,
                   length: int) -> torch.Tensor:
    """Return an FP16 view of L1[addr:addr+length*2] (no copy)."""
    l1_f16 = npu.mem.l1_f16(nest, spu)
    off = addr_byte // 2
    return l1_f16[off:off + length]


def _l0_block_view(npu, nest: int, spu: int, reg: int) -> torch.Tensor:
    """Return an FP16 view of L0[(reg & 0x1F)*32 .. +32]; 16 FP16 elements."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
    return l0.view(torch.float16)[off // 2:off // 2 + 16]


# =========================================================================
# Public entry: exec_vec_op
# =========================================================================
def exec_vec_op(npu, proc, insn) -> int:
    """Direct port of gtx_npu_vec.cc:572-754."""
    rs1 = int(proc.state.XPR[insn.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000

    funct7 = insn.funct
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    if spu >= GTX_SPU_NUM:
        spu = 0

    rs2 = int(proc.state.XPR[insn.rs2])
    npu.gspr[GSPR_GTX_OPERAND2] = rs2

    if funct7 == GTX_F7_VEC_SASMD:
        return _dispatch_sasmd(npu, nest, spu, funct3, rs1, rs2, insn,
                                vec_size)

    if funct7 == GTX_F7_VEC_ARITH and (funct3 & 4):
        return _dispatch_arith_l0_ii(npu, nest, spu, funct3 & 3, rs1, rs2, insn)
    if funct7 in (0x1C, 0x1D, 0x1E) and (funct3 & 4):
        return _dispatch_unary_l0(npu, nest, spu, funct7, funct3 & 3, rs1, insn)

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR_SPM_ADDRB, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)

    if funct7 == GTX_F7_VEC_ARITH:
        op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
        if (funct3 & 3) in op_map:
            view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            result = sasmd_kernel(view_a, view_b, op=op_map[funct3 & 3])
            _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
            return 0

    if funct7 == GTX_F7_VEC_DOT_SUM:
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if (funct3 & 3) == 0:
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            scalar = dot_kernel(view_a, view_b)
        else:
            scalar = vsum_kernel(view_a)
        _l1_view_addr(npu, nest, spu, addr_r, 1)[0] = scalar
        u16_val = int(scalar.to(torch.float16).view(torch.uint16))
        l0 = npu.mem.l0_byte(nest, spu)
        l0[0] = u16_val & 0xFF
        l0[1] = (u16_val >> 8) & 0xFF
        return 0

    if funct7 == GTX_F7_VEC_CLAMP:
        sub = funct3 & 3
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if sub == 0:
            scalar = _fp16_low16(rs2)
            result = clamp_min_kernel(view_a, scalar)
        elif sub == 1:
            scalar = _fp16_low16(rs2)
            result = clamp_max_kernel(view_a, scalar)
        elif sub == 2:
            result = accum_kernel(view_a)
        else:
            start = _fp16_low16(rs2)
            step = _fp16_high16(rs2)
            result = arange_kernel(vec_size, start, step)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    if funct7 in (0x1C, 0x1D, 0x1E):
        view = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = _apply_unary(funct7, funct3 & 3, view)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    return 0


# =========================================================================
# Sub-dispatchers
# =========================================================================
def _dispatch_sasmd(npu, nest: int, spu: int, funct3: int,
                     rs1: int, rs2: int, insn, vec_size: int) -> int:
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    sub = funct3 & 3
    if sub not in op_map:
        return 0

    scalar = _fp16_low16(rs2)
    if not (funct3 & 4):
        addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = sasmd_kernel(view_a, scalar, op=op_map[sub])
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get(GSPR_GTX_OPERAND3, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    result = sasmd_kernel(view_a, scalar, op=op_map[sub])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def _dispatch_arith_l0_ii(npu, nest: int, spu: int, sub_op: int,
                            rs1: int, rs2: int, insn) -> int:
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    if sub_op not in op_map:
        return 0
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    r_reg = int(npu.gspr.get(GSPR_GTX_OPERAND3, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    view_b = _l0_block_view(npu, nest, spu, b_reg)
    result = sasmd_kernel(view_a, view_b, op=op_map[sub_op])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def _dispatch_unary_l0(npu, nest: int, spu: int, funct7: int, sub_op: int,
                        rs1: int, insn) -> int:
    input_reg = rs1 & 0x1F
    op3_raw = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0xFFFFFFFF))
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub_op, view)
    _l0_block_view(npu, nest, spu, result_reg).copy_(result)
    return 0


def _apply_unary(funct7: int, sub_op: int, view: torch.Tensor) -> torch.Tensor:
    """Element-wise unary kernels for funct7 0x1C/0x1D/0x1E (FP32 internal)."""
    f32 = view.to(torch.float32)
    if funct7 == 0x1C:  # MATH: sqrt/exp/log
        if sub_op == 0:
            return torch.sqrt(f32).to(torch.float16)
        if sub_op == 1:
            return torch.exp(f32).to(torch.float16)
        if sub_op == 2:
            tiny = torch.finfo(torch.float32).tiny
            return torch.where(f32 > 0.0,
                               torch.log(f32.clamp(min=tiny)),
                               torch.zeros_like(f32)).to(torch.float16)
    if funct7 == 0x1D:  # SIGN: abs/neg/sign/step
        if sub_op == 0:
            return torch.abs(f32).to(torch.float16)
        if sub_op == 1:
            return (-f32).to(torch.float16)
        if sub_op == 2:
            return torch.sign(f32).to(torch.float16)
        if sub_op == 3:
            return (f32 > 0.0).to(torch.float16)
    if funct7 == 0x1E:  # ROUND
        if sub_op == 0:
            return torch.ceil(f32).to(torch.float16)
        if sub_op == 1:
            return torch.trunc(f32).to(torch.float16)
        if sub_op == 2:
            return torch.floor(f32).to(torch.float16)
        if sub_op == 3:
            return torch.round(f32).to(torch.float16)
    return view.clone()
