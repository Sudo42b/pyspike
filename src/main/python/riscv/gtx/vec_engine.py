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

Phase 5 plan 02 (vec) Wave 1b GREEN-fills. Plan 01 ships stub.
"""
from __future__ import annotations


def firmware_vec_op(npu, proc, insn) -> int:
    """Plan 02 GREEN-fill (rs1 = vec_size, funct3 = (xd<<2)|(xs1<<1)|xs2).

    Source: gtx_npu_vec.cc:572-754. Decodes rs1 (low 16 = vec_size, 0->65536),
    stages rs2 into GSPR_GTX_OPERAND2, branches L0 (funct3 & 4) vs L1 path,
    delegates to vec_core kernels.
    """
    return 0  # stub; Plan 02 fills body
