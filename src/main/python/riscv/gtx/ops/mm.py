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
"""MM/MMC op @handler entry points -- spike-bound shim layer (D-04).

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:333-389 dispatch
surface. The decode + variant logic lives in mm_engine.py (Plan 03);
GEMM kernel lives in gemm_core.py (Plan 02). This file ONLY:
  1. Registers 10 @handler entries (5 MM at funct7=0x00, 5 MMC at funct7=0x01)
  2. Applies Pitfall F: rs1==0 -> NOP (gem5-simplified WRSPR collision safety)
  3. Delegates to mm_engine.firmware_mm with explicit is_accumulate + variant.

Per RESEARCH Pitfall 5 + Pitfall F: funct7=0x00 collides with gem5-simplified
WRSPR. We DO NOT install a None-key (mask_funct3=False) handler at funct7=0x00,
because the 2-level dispatch (npu.py:125-144) would make None-key win over
funct3-keyed handlers, hiding all 5 MM variants. Instead: each per-funct3
handler checks `insn.rs1 == 0` at entry and NOPs if true.

Phase 4 plan 04 Task 2.
"""
from .._registry import handler
from .. import mm_engine
from ..encoding import (
    GTX_F7_WRSPR, GTX_F7_RDSPR,    # 0x00 (MM family) and 0x01 (MMC family)
    GTX_F3_MM_S, GTX_F3_MM_O, GTX_F3_MM, GTX_F3_MM_V, GTX_F3_MM_T,
)


# =========================================================================
# MM family -- funct7=0x00. Pitfall F: rs1==0 collides with gem5 WRSPR -> NOP.
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_S,
         mnemonic='mm_s', mask_funct3=True)
def _exec_mm_s(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=0 -> mm_s (FP32 result to ADDRC)."""
    if insn.rs1 == 0:
        return 0  # Pitfall F: gem5 WRSPR collision -- NOP for safety
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=False, variant='mm_s')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_O,
         mnemonic='mm_o', mask_funct3=True)
def _exec_mm_o(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=1 -> mm_o (scalar sum(A) to L0 BE + mxe_accum)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=False, variant='mm_o')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM,
         mnemonic='mm', mask_funct3=True)
def _exec_mm(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=2 -> mm (basic GEMM, FP16 result to ADDRR)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=False, variant='mm')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_V,
         mnemonic='mm_v', mask_funct3=True)
def _exec_mm_v(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=3 -> mm_v (scalar dot(A,B) to L0 LE + mxe_accum)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=False, variant='mm_v')


@handler(kind='custom0', funct7=GTX_F7_WRSPR, funct3=GTX_F3_MM_T,
         mnemonic='mm_t', mask_funct3=True)
def _exec_mm_t(npu, proc, insn, xs1, xs2):
    """funct7=0x00 funct3=7 -> mm_t (transposed C^T to ADDRR, NxM layout)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=False, variant='mm_t')


# =========================================================================
# MMC family -- funct7=0x01. is_accumulate=True. RDSPR collision is similar
# but RDSPR is registered at funct7=0x48 (ISS-full) in P2; gem5-simplified
# RDSPR funct7=0x01 + rs1!=0 is the only valid MMC route. Apply same rs1
# safety just for symmetry (it's also a NOP if rs1==0).
# =========================================================================
@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_S,
         mnemonic='mmc_s', mask_funct3=True)
def _exec_mmc_s(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=0 -> mmc_s (FP32 result to ADDRC, accumulate)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=True, variant='mmc_s')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_O,
         mnemonic='mmc_o', mask_funct3=True)
def _exec_mmc_o(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=1 -> mmc_o (mxe_accum chain: prior + sum(A))."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=True, variant='mmc_o')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM,
         mnemonic='mmc', mask_funct3=True)
def _exec_mmc(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=2 -> mmc (basic GEMM with ADDRC FP32 bias)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=True, variant='mmc')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_V,
         mnemonic='mmc_v', mask_funct3=True)
def _exec_mmc_v(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=3 -> mmc_v (mxe_accum chain: prior + dot(A,B))."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=True, variant='mmc_v')


@handler(kind='custom0', funct7=GTX_F7_RDSPR, funct3=GTX_F3_MM_T,
         mnemonic='mmc_t', mask_funct3=True)
def _exec_mmc_t(npu, proc, insn, xs1, xs2):
    """funct7=0x01 funct3=7 -> mmc_t (transposed C^T to ADDRR, accumulate)."""
    if insn.rs1 == 0:
        return 0
    return mm_engine.firmware_mm(npu, proc, insn,
                                  is_accumulate=True, variant='mmc_t')
