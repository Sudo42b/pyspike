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
"""ACT op @handler entries -- thin shim layer for activations.

Plan 03 lands 12 ISS activation @handlers (8 ISS L1 path: prelu/gelu/tanh/
sigmoid + 4 _imm at funct3 & 4 = L0 immediate path; 4 forward at funct7=0x2F:
esum/softmax/esum_i/softmax_i).

Plan 04 will append 7 cvt @handlers (funct7=0x20/0x21/0x22/0x24/0x25) +
2 pool @handlers (funct7=0x30/0x31).

Per CONTEXT D-05/D-06: each @handler is the source-of-truth for `is_reversed`.
The act_engine asserts consistency vs ACT_OPS_REVERSED frozenset (engine
internal check; @handler literal is the policy).

Vendor authority for funct7/funct3 values:
  vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:152-157 (verbatim).
"""
from .._registry import handler
from .. import act_engine
from ..encoding import (
    GTX_F7_ACT_PRELU, GTX_F7_ACT_GELU, GTX_F7_ACT_TANH,
    GTX_F7_ACT_SIGM, GTX_F7_ACT_SOFTMAX,
    GTX_ACT_PRELU, GTX_ACT_GELU, GTX_ACT_TANH,
    GTX_ACT_SIGMOID, GTX_ACT_SOFTMAX, GTX_ACT_ESUM,
)


# ============================================================================
# ISS L1 path (8 activations -- 4 reversed + 2 forward at 0x2F)
# disasm.inc:152-156
# ============================================================================

# PRELU: funct7=0x28 funct3=3, REVERSED (vendor cc:37-42)
@handler(kind='custom0', funct7=GTX_F7_ACT_PRELU, funct3=3,
         mnemonic='prelu', mask_funct3=True)
def _exec_prelu(npu, proc, insn, xs1, xs2):
    """Reversed direction: ADDRR -> ADDRA. Slope from GSPR_OPERAND2 low-16."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_PRELU, is_reversed=True)


# GELU: funct7=0x2A funct3=0, REVERSED
@handler(kind='custom0', funct7=GTX_F7_ACT_GELU, funct3=0,
         mnemonic='gelu', mask_funct3=True)
def _exec_gelu(npu, proc, insn, xs1, xs2):
    """Reversed direction: ADDRR -> ADDRA."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_GELU, is_reversed=True)


# TANH: funct7=0x2C funct3=0, REVERSED
@handler(kind='custom0', funct7=GTX_F7_ACT_TANH, funct3=0,
         mnemonic='tanh', mask_funct3=True)
def _exec_tanh(npu, proc, insn, xs1, xs2):
    """Reversed direction: ADDRR -> ADDRA."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_TANH, is_reversed=True)


# SIGMOID: funct7=0x2D funct3=0, REVERSED
@handler(kind='custom0', funct7=GTX_F7_ACT_SIGM, funct3=0,
         mnemonic='sigmoid', mask_funct3=True)
def _exec_sigmoid(npu, proc, insn, xs1, xs2):
    """Reversed direction: ADDRR -> ADDRA."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_SIGMOID, is_reversed=True)


# ESUM: funct7=0x2F funct3=1, FORWARD -- writes scalar to L0 (Pitfall 8)
@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=1,
         mnemonic='esum', mask_funct3=True)
def _exec_esum(npu, proc, insn, xs1, xs2):
    """Forward direction -- but writes scalar to L0[(GSPR_OPERAND3 & 0x1F)*32],
    NOT to L1[ADDRR] (Pitfall 8)."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_ESUM, is_reversed=False)


# SOFTMAX: funct7=0x2F funct3=2, FORWARD
@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=2,
         mnemonic='softmax', mask_funct3=True)
def _exec_softmax(npu, proc, insn, xs1, xs2):
    """Forward direction: ADDRA -> ADDRR."""
    return act_engine.firmware_act(npu, proc, insn,
                                    op_id=GTX_ACT_SOFTMAX, is_reversed=False)


# ============================================================================
# L0 immediate path (6 _imm activations -- funct3 & 4 selects L0)
# disasm.inc:152-157 (the funct3=4..7 column)
# ============================================================================
# Per RESEARCH Adjustment 3: L0 path uses (input_reg, result_reg). Direction
# is moot at byte level; engine ignores `is_reversed` in firmware_act_imm.
# ============================================================================

# prelu_i: funct7=0x28 funct3=7
@handler(kind='custom0', funct7=GTX_F7_ACT_PRELU, funct3=7,
         mnemonic='prelu_i', mask_funct3=True)
def _exec_prelu_i(npu, proc, insn, xs1, xs2):
    """L0 immediate (RESEARCH Adjustment 3). Slope from GSPR_OPERAND2 low-16."""
    return act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_PRELU)


# gelu_i: funct7=0x2A funct3=4
@handler(kind='custom0', funct7=GTX_F7_ACT_GELU, funct3=4,
         mnemonic='gelu_i', mask_funct3=True)
def _exec_gelu_i(npu, proc, insn, xs1, xs2):
    """L0 immediate."""
    return act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_GELU)


# tanh_i: funct7=0x2C funct3=4
@handler(kind='custom0', funct7=GTX_F7_ACT_TANH, funct3=4,
         mnemonic='tanh_i', mask_funct3=True)
def _exec_tanh_i(npu, proc, insn, xs1, xs2):
    """L0 immediate."""
    return act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_TANH)


# sigm_i: funct7=0x2D funct3=4
@handler(kind='custom0', funct7=GTX_F7_ACT_SIGM, funct3=4,
         mnemonic='sigm_i', mask_funct3=True)
def _exec_sigm_i(npu, proc, insn, xs1, xs2):
    """L0 immediate."""
    return act_engine.firmware_act_imm(npu, proc, insn, op_id=GTX_ACT_SIGMOID)


# esum_i: funct7=0x2F funct3=5 (L0 path -- exec_softmax_imm ESUM branch)
@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=5,
         mnemonic='esum_i', mask_funct3=True)
def _exec_esum_i(npu, proc, insn, xs1, xs2):
    """L0 immediate ESUM (gtx_npu_act.cc:436-487 ESUM branch)."""
    return act_engine.firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_ESUM)


# softmax_i: funct7=0x2F funct3=6
@handler(kind='custom0', funct7=GTX_F7_ACT_SOFTMAX, funct3=6,
         mnemonic='softmax_i', mask_funct3=True)
def _exec_softmax_i(npu, proc, insn, xs1, xs2):
    """L0 immediate SOFTMAX (uses pre-computed esum from GSPR_OPERAND2 high-16)."""
    return act_engine.firmware_softmax_imm(npu, proc, insn, op_id=GTX_ACT_SOFTMAX)
