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
"""4-mode warp dispatch router + ISS opcode stub.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:25-200.

Lives in its own module (NOT dispatch.py) to keep Plan 04 file ownership
distinct from Plan 02's dispatch.py table-builder upgrades, and to honor
03-CONTEXT.md "Defer to user follow-up" §dispatch.py vs dispatch_4mode.py
split. dispatch.py re-exports both functions so the public import surface
(`from riscv.gtx.dispatch import dispatch_4mode`) remains unchanged.

Plan 02 owns dispatch.py body; Plan 04 owns this file + a 1-line import-only
append in dispatch.py.
"""
from .params import GTX_NEST_NUM, GTX_SPU_NUM
from .encoding import (
    GTX_OP_DMA,
    GTX_ISS_F7_DMA_LD_SVR_L1, GTX_ISS_F7_DMA_ST_SVR_L1,
    GTX_ISS_F7_CREDIT_ST_CHK,
)
from . import dma_engine


def dispatch_iss_opcode(npu, nest_id: int, spu_id: int, funct7: int,
                         op1: int, op2: int, op3: int) -> int:
    """Unified opcode router. Direct port of gtx_npu_dispatch.cc:151-1100+ --
    P3 stubs only DMA-relevant funct7s; P4 fills MM (funct7=GTX_OP_MM=0),
    P5 fills VEC/ACT.

    In P3 the only firmware paths that reach this are:
      - dispatch_4mode Mode 1/2/4 (broadcasting MM/VEC/ACT -- all NOP in P3)
      - load_svr_l1 (funct7=0x43) -- disasm-only stub here
      - store_svr_l1 (funct7=0x45) -- disasm-only stub here
      - credit_st_chk (funct7=0x53) -- Plan 05 wires the flush trigger here

    Returns 0 (cycles vestigial). NEVER raises -- invalid funct7 silently NOPs.
    Out-of-range nest_id / spu_id silently NOP (matches C++ guard pattern).
    """
    if nest_id < 0 or nest_id >= GTX_NEST_NUM:
        return 0
    if spu_id < 0 or spu_id >= GTX_SPU_NUM:
        return 0
    # P3 NOPs for everything. P4/P5 will dispatch to op modules here.
    # Plan 05 will replace this body with a credit_st_chk flush trigger:
    #   if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop:
    #       npu.flush_deferred_ddr_stores()
    # Reference funct7 constants (used by P4/P5 fillers):
    _ = (funct7, op1, op2, op3)  # silence linters; future entry point
    _ = (GTX_ISS_F7_DMA_LD_SVR_L1, GTX_ISS_F7_DMA_ST_SVR_L1,
         GTX_ISS_F7_CREDIT_ST_CHK)
    return 0


def dispatch_4mode(npu, *, opcode: int, op1: int, op2: int, op3: int,
                    sub_op: int = 0) -> int:
    """4-mode warp router. Direct port of gtx_npu_dispatch.cc:79-139.

    Mode 1: !is_ploop                            -> broadcast all NEST x SPU
    Mode 2: is_ploop && !is_sloop && !is_tloop   -> broadcast SPU within tmu_id
    Mode 3: is_ploop && is_sloop                 -> DDR<->L2 via dma_engine.exec_dma_2d
    Mode 4: is_ploop && is_tloop                 -> single (tmu_id, curr_id)

    Args:
      npu: GtxNpu instance (for .warp, .mem)
      opcode: GTX_OP_MM | GTX_OP_VECTOR | GTX_OP_ACTIVATION | GTX_OP_DMA
      op1, op2, op3: read by caller from npu.gspr[GSPR_GTX_OPERAND1/2/3]
      sub_op: low byte of npu.gspr[GSPR_GTX_OPCODE]

    Returns: vestigial cycle count (0 in functional model).

    Notes:
      - firmware_dma (funct7=0x40) bypasses this router entirely; only the
        gem5-simplified dispatch_dma (funct7=0x07) reaches Mode 3 here.
      - Pitfall 8 (Mode 3): is_load = (sub_op == 0) OR (opcode == GTX_OP_DMA).
    """
    w = npu.warp
    if not w.is_ploop:
        # Mode 1: broadcast all
        for n in range(GTX_NEST_NUM):
            for s in range(GTX_SPU_NUM):
                dispatch_iss_opcode(npu, n, s, opcode, op1, op2, op3)
        return 0
    if w.is_ploop and not w.is_sloop and not w.is_tloop:
        # Mode 2: broadcast within tmu_id
        for s in range(GTX_SPU_NUM):
            dispatch_iss_opcode(npu, w.tmu_id, s, opcode, op1, op2, op3)
        return 0
    if w.is_ploop and w.is_sloop:
        # Mode 3: DDR<->L2 single-NEST DMA via dma_engine.exec_dma_2d.
        # Pitfall 8: is_load = (sub_op == 0) || (opcode == GTX_OP_DMA).
        is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)
        return dma_engine.exec_dma_2d(
            npu.mem,
            nest_id=w.tmu_id,
            l2_addr=op1 & 0xFFFFFFFF,
            l1_addr=op2 & 0xFFFFFFFF,
            width=op3 & 0xFFFF,
            height=(op3 >> 16) & 0xFFFF,
            is_load=is_load,
        )
    if w.is_ploop and w.is_tloop:
        # Mode 4: single (tmu_id, curr_id)
        return dispatch_iss_opcode(
            npu, w.tmu_id, w.curr_id, opcode, op1, op2, op3
        )
    return 0
