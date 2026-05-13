"""SPR ops -- WRSPR/RDSPR handlers (SPR-02).

Ports vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:56-113 verbatim.

Four handlers:
  funct7 = 0x49 (ISS-full WRSPR)        -- full encoding with ALL register reads via XPR
  funct7 = 0x48 (ISS-full RDSPR)        -- same, plus force-write to rd
  funct7 = 0x00 (fully implemented WRSPR) -- collision-aware: insn.rs1!=0 -> P4 MM stub
  funct7 = 0x01 (fully implemented RDSPR) -- collision-aware: insn.rs1!=0 -> P4 MMC stub

All four read register values directly via proc.state.XPR[insn.rs1] to bypass
Spike's xs1=0 -> -1 marshalling (CORE-04 / D-05).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...._registry import handler
from ...context.spr_router import rd_spr, wr_spr
from ..encoding import (
    GTX_F7_RDSPR,
    GTX_F7_WRSPR,
    GTX_ISS_F7_CPSVR,
    GTX_ISS_F7_MVSVR,
    GTX_ISS_F7_OPSET,
    GTX_ISS_F7_RDSPR_ISS,
    GTX_ISS_F7_WRSPR_ISS,
)

if TYPE_CHECKING:
    from ....npu import GtxNpu

# =============================================================================
# WRSPR / RDSPR (gem5 simplified encoding, funct7=0x00/0x01)
# =============================================================================

@handler(kind='custom0', funct7=GTX_F7_WRSPR, mnemonic='wrspr_gem5')
def wrspr_gem5(npu: GtxNpu, proc, insn, xs1, xs2):
    """gem5 WRSPR with funct7=0x00 collision (D-02): insn.rs1!=0 → MM stub.
    
    Port of ``custom0.cc:63-72``. When rs1==0, it writes to SPR 0
    (GSPR_GTX_RUN). When rs1!=0, it's actually an MM instruction which
    is handled by mm.py (each mm handler has a rs1==0 -> NOP guard).
    """
    if insn.rs1 != 0:
        return 0  # Should be unreachable if mm.py is loaded correctly
    
    state = proc.state
    # Even for rs1==0, we read XPR[0] (0) to match C++ verbatim
    addr = state.XPR[insn.rs1]
    val = state.XPR[insn.rs2]
    wr_spr(npu, addr & 0xFFFF, val)
    return 0


@handler(kind='custom0', funct7=GTX_F7_RDSPR, mnemonic='rdspr_gem5')
def rdspr_gem5(npu: GtxNpu, proc, insn, xs1, xs2):
    """gem5 RDSPR with funct7=0x01 collision (D-02): insn.rs1!=0 → MMC stub."""
    if insn.rs1 != 0:
        return 0
    
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = rd_spr(npu, addr & 0xFFFF)
    if insn.rd != 0:
        state.XPR.write(insn.rd, val)
    return val


# =============================================================================
# ISS Full Encoding (funct7=0x48-0x4C)
# =============================================================================

@handler(kind='custom0', funct7=GTX_ISS_F7_RDSPR_ISS, mnemonic='rdspr')
def rdspr_full(npu: GtxNpu, proc, insn, xs1, xs2):
    """Full-encoding RDSPR (funct7=0x48): addr from XPR[rs1], return rd_spr(addr)."""
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = rd_spr(npu, addr & 0xFFFF)
    if insn.rd != 0:
        state.XPR.write(insn.rd, val)
    return val


@handler(kind='custom0', funct7=GTX_ISS_F7_WRSPR_ISS, mnemonic='wrspr')
def wrspr_full(npu: GtxNpu, proc, insn, xs1, xs2):
    """Full-encoding WRSPR (funct7=0x49): addr from XPR[rs1], val from XPR[rs2]."""
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = state.XPR[insn.rs2]
    wr_spr(npu, addr & 0xFFFF, val)
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_OPSET, mnemonic='opset')
def opset(npu: GtxNpu, proc, insn, xs1, xs2):
    """ISS-full OPSET: stage rs3/rs4 for the next instruction."""
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    slot = rs1_val & 1
    if slot == 0:
        npu.gspr.tensor[0x003] = rs2_val  # Direct tensor access for performance
    else:
        npu.gspr.tensor[0x005] = rs2_val
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_CPSVR, mnemonic='cpsvr')
def cpsvr(npu: GtxNpu, proc, insn, xs1, xs2):
    """CPSVR (funct7=0x4B): replicate L0 SVR register pattern.
    
    rs1 = SVR address/index, rs2 = pattern size [1:0].
    Port of ``custom0.cc:138-164`` verbatim.
    """
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    
    # base is LSPR-relative (4 words per SVR)
    base = ((rs1_val & 0x1F) // 4) * 32
    bsz = rs2_val & 3
    
    l0 = npu.mem.l0_byte(nest, spu)
    # Build 8B pattern from L0[base]
    if bsz == 0: # 1B -> 8x
        p = l0[base].repeat(8)
    elif bsz == 1: # 2B -> 4x
        p = l0[base:base+2].repeat(4)
    elif bsz == 2: # 4B -> 2x
        p = l0[base:base+4].repeat(2)
    else: # 8B -> 1x
        p = l0[base:base+8]
        
    # Fill the 32B SVR with the 8B pattern replicated 4 times
    l0[base:base+32] = p.repeat(4)
    return 0


@handler(kind='custom0', funct7=GTX_ISS_F7_MVSVR, mnemonic='mvsvr')
def mvsvr(npu: GtxNpu, proc, insn, xs1, xs2):
    """MVSVR (funct7=0x4C): Move 32B L0 SVR register (copy + clear source)."""
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    
    # MVSVR uses direct register indexing (0-31)
    src_idx = rs1_val & 0x1F
    dst_idx = rs2_val & 0x1F
    
    if src_idx == dst_idx:
        return 0
        
    l0 = npu.mem.l0_byte(nest, spu)
    src_off = src_idx * 32
    dst_off = dst_idx * 32
    
    l0[dst_off:dst_off+32] = l0[src_off:src_off+32].clone()
    l0[src_off:src_off+32].zero_()
    return 0
