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
`proc.state.XPR[i]` (P4 04-05 lock).

Phase 5 plans 03/04 Wave 1b GREEN-fill. Plan 01 ships stubs.
"""
from __future__ import annotations


def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Plan 03 GREEN-fill. Source: gtx_npu_act.cc:23-164.

    Direction routing per CONTEXT D-05:
      - is_reversed=False: read ADDRA, write ADDRR (RELU/SOFTMAX/ESUM)
      - is_reversed=True:  read ADDRR, write ADDRA (PRELU/GELU/TANH/SIGM)
    """
    return 0


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


def firmware_act_imm(npu, proc, insn, *, op_id: int) -> int:
    """Plan 03 GREEN-fill. Source: gtx_npu_act.cc:374-431 (PRELU/GELU/TANH/SIGM L0)."""
    return 0


def firmware_softmax_imm(npu, proc, insn, *, op_id: int) -> int:
    """Plan 03 GREEN-fill. Source: gtx_npu_act.cc:436-487 (ESUM/SOFTMAX L0)."""
    return 0
