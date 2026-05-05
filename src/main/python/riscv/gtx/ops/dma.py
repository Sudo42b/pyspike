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
"""DMA op @handler entry points -- spike-bound shim layer (CONTEXT D-01).

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc dispatch entry points.
The pure-function bodies live in dma_engine.py (Plan 01); this file ONLY reads
from proc/insn/npu and delegates.

Phase 3 plan 02 Task 2a: 9 active @handler entry points
  - firmware_dma load/store/copy (funct7=0x40, mask_funct3=True, funct3=0/1/2)
  - load_svr/store_svr           (funct7=0x41, mask_funct3=True, funct3=0/1)
  - load_svr_l1/store_svr_l1     (funct7=0x43/0x45, mask_funct3=False)
  - tpose                        (funct7=0x38, mask_funct3=False)
  - fill                         (funct7=0x39, mask_funct3=False)

Phase 3 plan 02 Task 2b: 5 disasm-only stubs + credit_st_chk stub.
"""
from .._registry import handler
from .. import dma_engine
from ..encoding import (
    GSPR_GTX_OPERAND3,                      # 0x003 -- gtx_params.h:40
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,         # 0x900 / 0x903 -- gtx_params.h:64,67
    GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
    GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
    GTX_ISS_F7_DMA_MCAST_S2L, GTX_ISS_F7_DMA_LD_SVR_L1,
    GTX_ISS_F7_DMA_MCAST_GS, GTX_ISS_F7_DMA_ST_SVR_L1,
    GTX_ISS_F7_CREDIT_ST_CHK,
)
from ..params import GTX_NEST_NUM, GTX_SPU_NUM


# ============================================================================
# Helpers
# ============================================================================
def _select_nest(npu) -> int:
    """Select NEST id per gtx_npu_dma.cc:289-291.

    is_ploop -> use warp.tmu_id; else default to 0. Out-of-range clamps to 0.
    """
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    if nest >= GTX_NEST_NUM:
        nest = 0
    return nest


def _select_spu(npu) -> int:
    """Select SPU id from warp.curr_id, clamped to GTX_SPU_NUM."""
    spu = npu.warp.curr_id
    if spu >= GTX_SPU_NUM:
        spu = 0
    return spu


# ============================================================================
# firmware_dma load/store/copy (funct7=0x40, mask_funct3=True)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=0,
         mnemonic='load', mask_funct3=True)
def _firmware_dma_load(npu, proc, insn, xs1, xs2):
    """firmware_dma LOAD (funct7=0x40 funct3=0).

    Pitfall 3 (CORE-04): xs1/xs2 args are unreliable when the encoding flag is 0
    (Spike marshals -1). Read XPR[insn.rs1] / XPR[insn.rs2] directly.
    """
    state = proc.get_state()
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR_GTX_OPERAND3, 0)   # 0x003 per gtx_params.h:40
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        return dma_engine.firmware_dma_sloop_load(
            npu.mem, nest=nest,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_load_store(
            npu.mem, nest=nest, spu=spu, is_store=False,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=1,
         mnemonic='store', mask_funct3=True)
def _firmware_dma_store(npu, proc, insn, xs1, xs2):
    """firmware_dma STORE (funct7=0x40 funct3=1).

    is_sloop branch passes `npu` (not `npu.mem`) so firmware_dma_sloop_store can
    push DeferredDdrStore onto npu.deferred_ddr_stores (Plan 05 flushes).
    """
    state = proc.get_state()
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR_GTX_OPERAND3, 0)
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_sloop:
        return dma_engine.firmware_dma_sloop_store(
            npu, nest=nest,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_load_store(
            npu.mem, nest=nest, spu=spu, is_store=True,
            addr_hi=args['addr_hi'], addr_lo=args['addr_lo'],
            length=args['length'], height=args['height'],
            rd_stride=args['rd_stride'], wr_stride=args['wr_stride'])
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_ST, funct3=2,
         mnemonic='copy', mask_funct3=True)
def _firmware_dma_copy(npu, proc, insn, xs1, xs2):
    """firmware_dma COPY (funct7=0x40 funct3=2). T-loop L1->L1 only.

    Pitfall 1: COPY decodes addr_hi from rs1>>32 (32-bit dst), NOT (rs1>>27)&0x1F..
    addr_hi is the dst, addr_lo is the src.
    """
    state = proc.get_state()
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR_GTX_OPERAND3, 0)
    args = dma_engine.decode_firmware_dma_args(
        rs1, rs2, rs3, xd=insn.xd, xs1=insn.xs1, xs2=insn.xs2)
    nest = _select_nest(npu)
    if npu.warp.is_tloop:
        spu = _select_spu(npu)
        return dma_engine.firmware_dma_tloop_copy(
            npu.mem, nest=nest, spu=spu,
            src_addr=args['addr_lo'], dst_addr=args['addr_hi'],
            length=args['length'], height=args['height'])
    return 0


# ============================================================================
# load_svr/store_svr (funct7=0x41, mask_funct3=True)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_3D, funct3=0,
         mnemonic='load_svr', mask_funct3=True)
def _load_svr(npu, proc, insn, xs1, xs2):
    """load_svr (funct7=0x41 funct3=0): L1 -> L0 SVR transfer (32 bytes)."""
    state = proc.get_state()
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_load_svr(npu.mem, nest_id=nest, spu_id=spu,
                              l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_3D, funct3=1,
         mnemonic='store_svr', mask_funct3=True)
def _store_svr(npu, proc, insn, xs1, xs2):
    """store_svr (funct7=0x41 funct3=1): L0 -> L1 SVR transfer (32 bytes)."""
    state = proc.get_state()
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_store_svr(npu.mem, nest_id=nest, spu_id=spu,
                               l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


# ============================================================================
# load_svr_l1 / store_svr_l1 (funct7=0x43 / 0x45, mask_funct3=False) -- aliases
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_LD_SVR_L1, mnemonic='load_svr_l1')
def _load_svr_l1(npu, proc, insn, xs1, xs2):
    """load_svr_l1 (funct7=0x43): L1-bound load_svr alias."""
    state = proc.get_state()
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_load_svr(npu.mem, nest_id=nest, spu_id=spu,
                              l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_ST_SVR_L1, mnemonic='store_svr_l1')
def _store_svr_l1(npu, proc, insn, xs1, xs2):
    """store_svr_l1 (funct7=0x45): L1-bound store_svr alias."""
    state = proc.get_state()
    l1_addr = state.XPR[insn.rs1] & 0x7FFFFFF
    l0_reg = state.XPR[insn.rs2] & 0x1F
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    dma_engine.exec_store_svr(npu.mem, nest_id=nest, spu_id=spu,
                               l1_addr=l1_addr, l0_reg=l0_reg)
    return 0


# ============================================================================
# tpose / fill (funct7=0x38 / 0x39, mask_funct3=False)
# ============================================================================
@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_TPOSE, mnemonic='tpose')
def _tpose(npu, proc, insn, xs1, xs2):
    """tpose (funct7=0x38): matrix transpose in L1 (FP16, 2 bytes per elem).

    Source matrix base: LSPR_SPM_ADDRA (0x900) -- gtx_params.h:64
    Result matrix base: LSPR_SPM_ADDRR (0x903) -- gtx_params.h:67
    AUTHORITATIVE values; no magic numbers in handler body.
    """
    state = proc.get_state()
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rows = rs1 & 0xFFFF
    cols = rs2 & 0xFFFF
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0) & 0xFFFFFFFF
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0) & 0xFFFFFFFF
    return dma_engine.exec_transpose(
        npu.mem, nest_id=nest, spu_id=spu, rows=rows, cols=cols,
        addr_a=addr_a, addr_r=addr_r)


@handler(kind='custom0', funct7=GTX_ISS_F7_DMA_FILL, mnemonic='fill')
def _fill(npu, proc, insn, xs1, xs2):
    """fill (funct7=0x39): fill L1 region at addr_r with constant FP16 value.

    Result address: LSPR_SPM_ADDRR (0x903) -- gtx_params.h:67. AUTHORITATIVE
    constant; no magic number in handler body (LSPR_SPM_ADDRB is NOT used here).
    """
    state = proc.get_state()
    rs1 = state.XPR[insn.rs1]
    length = rs1 & 0xFFFF
    fill_val = (rs1 >> 16) & 0xFFFF
    nest = _select_nest(npu)
    spu = _select_spu(npu)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0) & 0xFFFFFFFF
    return dma_engine.exec_fill(
        npu.mem, nest_id=nest, spu_id=spu,
        length=length, fill_val=fill_val, addr_r=addr_r)
