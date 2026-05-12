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
"""VEC op @handler entries -- D-04 thin shim layer (mirror of ops/mm.py).

Each handler delegates the full decode/dispatch work to
`vec_engine.firmware_vec_op`. The disasm layer is fed by
`@handler(mnemonic=..., mask_funct3=True)` -- one mnemonic per
(funct7, funct3) tuple, matching `gtx_npu_disasm.inc:67-142`.

Phase 5 plan 02 task 3.
"""
from ..._registry import handler
from ... import vec_engine
from ...encoding import (
    GTX_F7_VEC_SASMD, GTX_F7_VEC_DOT_SUM, GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP,
    GTX_F7_VEC_MATH, GTX_F7_VEC_SIGN, GTX_F7_VEC_ROUND,
)


# =========================================================================
# SASMD scalar arith funct7=0x10 (8 variants: VS funct3=0..3, IS funct3=4..7)
# disasm.inc:67-74
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=0,
         mnemonic='add_vs', mask_funct3=True)
def _exec_add_vs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=1,
         mnemonic='sub_vs', mask_funct3=True)
def _exec_sub_vs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=2,
         mnemonic='mul_vs', mask_funct3=True)
def _exec_mul_vs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=3,
         mnemonic='div_vs', mask_funct3=True)
def _exec_div_vs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=4,
         mnemonic='add_is', mask_funct3=True)
def _exec_add_is(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=5,
         mnemonic='sub_is', mask_funct3=True)
def _exec_sub_is(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=6,
         mnemonic='mul_is', mask_funct3=True)
def _exec_mul_is(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=7,
         mnemonic='div_is', mask_funct3=True)
def _exec_div_is(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


# =========================================================================
# VSUM/DOT funct7=0x1A (DOT at funct3=0, SUM at funct3=1)
# disasm.inc:101-104 -- vendor authoritative; Plan 01 seeded 0x13 incorrectly.
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_VEC_DOT_SUM, funct3=0,
         mnemonic='dot_vvs', mask_funct3=True)
def _exec_dot_vvs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_DOT_SUM, funct3=1,
         mnemonic='sum_vs', mask_funct3=True)
def _exec_sum_vs(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


# =========================================================================
# SASMD vector arith funct7=0x18 (8 variants: VV funct3=0..3, II funct3=4..7)
# disasm.inc:87-94
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=0,
         mnemonic='add_vv', mask_funct3=True)
def _exec_add_vv(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=1,
         mnemonic='sub_vv', mask_funct3=True)
def _exec_sub_vv(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=2,
         mnemonic='mul_vv', mask_funct3=True)
def _exec_mul_vv(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=3,
         mnemonic='div_vv', mask_funct3=True)
def _exec_div_vv(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=4,
         mnemonic='add_ii', mask_funct3=True)
def _exec_add_ii(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=5,
         mnemonic='sub_ii', mask_funct3=True)
def _exec_sub_ii(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=6,
         mnemonic='mul_ii', mask_funct3=True)
def _exec_mul_ii(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=7,
         mnemonic='div_ii', mask_funct3=True)
def _exec_div_ii(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


# =========================================================================
# CLAMP family funct7=0x1F (4 variants funct3=0..3; bitwise funct3=4..7
# deferred for v1 critical path)
# disasm.inc:135-138
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=0,
         mnemonic='clamp_min_v', mask_funct3=True)
def _exec_clamp_min_v(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=1,
         mnemonic='clamp_max_v', mask_funct3=True)
def _exec_clamp_max_v(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=2,
         mnemonic='accum_v', mask_funct3=True)
def _exec_accum_v(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=3,
         mnemonic='arange_v', mask_funct3=True)
def _exec_arange_v(npu, proc, insn, xs1, xs2):
    return vec_engine.firmware_vec_op(npu, proc, insn)


# =========================================================================
# MATH / SIGN / ROUND families (funct7 0x1C / 0x1D / 0x1E)
# Sub-op selected by funct3:
#   0x1C: 0=sqrt 1=exp 2=log
#   0x1D: 0=abs  1=neg 2=sign 3=step
#   0x1E: 0=ceil 1=trunc 2=floor 3=rne
# All variants delegate to vec_engine.firmware_vec_op which routes
# (funct7, funct3 & 3) through _apply_unary.
# P8 NEG fix (2026-05-11): previous code registered only funct7=0x1D funct3=0
# (incorrectly mnemonic'd 'sign_v' — that slot is actually abs.v). neg.v
# emits funct3=1, sign.v funct3=2, step.v funct3=3 — all silent-NOP'd
# without a handler. Same gap for the entire 0x1C MATH family and 0x1E ROUND
# family. Disasm precision (one mnemonic per funct3) preserved.
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=0,
         mnemonic='sqrt_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=1,
         mnemonic='exp_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=2,
         mnemonic='log_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=0,
         mnemonic='abs_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=1,
         mnemonic='neg_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=2,
         mnemonic='sign_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=3,
         mnemonic='step_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=0,
         mnemonic='ceil_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=1,
         mnemonic='trunc_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=2,
         mnemonic='floor_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=3,
         mnemonic='rne_v', mask_funct3=True)
def _exec_unary_family(npu, proc, insn, xs1, xs2):
    """Element-wise unary entry (MATH/SIGN/ROUND). Sub-op decoded from
    (funct7, funct3) inside vec_engine.firmware_vec_op._apply_unary.
    """
    return vec_engine.firmware_vec_op(npu, proc, insn)
