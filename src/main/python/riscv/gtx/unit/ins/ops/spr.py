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
from ..encoding import (
    GTX_ISS_F7_CPSVR,
    GTX_ISS_F7_MVSVR,
    GTX_ISS_F7_OPSET,
    GTX_ISS_F7_RDSPR_ISS,
    GTX_ISS_F7_WRSPR_ISS,
)

if TYPE_CHECKING:
    from ....npu import GtxNpu
"""SPR routing -- port of vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc.

GSPR (0x000-0x3FF) flat single-instance.
NSPR (0x400-0x7FF) per-NEST -- routed by tmu_id when is_ploop, else NEST 0.
LSPR (0x800-0xBFF) per-(NEST,SPU) -- routed by (tmu_id, curr_id) when is_tloop,
broadcast across SPUs when is_ploop, fallback to (0,0) otherwise.

Loop-control GSPR addresses 0x100..0x105 trigger startp/endp/starts/ends/startt/endt
side-effect handlers from ops.control (lazy imported to avoid plan 02 -> plan 03
circular import; plan 03 provides the _do_* helpers).
"""
from ....config_params import GTX_NEST_NUM, GTX_SPU_NUM
from ...csr import (GSPR_BASE, GSPR_END, NSPR_BASE, NSPR_END,
                   LSPR_BASE, LSPR_END)
# from ..ins.encoding import (GSPR_STARTP, GSPR_ENDP, GSPR_STARTS,
#                        GSPR_ENDS, GSPR_STARTT, GSPR_ENDT)


def _in_range(addr: int, base: int, end: int) -> bool:
    return base <= addr <= end

# =============================================================================
# ISS Full Encoding (funct7=0x48-0x4C)
# =============================================================================


def rd_spr(npu, addr: int) -> int:
    """
        Read SPR. Port of gtx_npu_t::rd_spr (gtx_npu_spr.cc:83-107).
        
    operand1: nest_id[29:24],spu_id[21:16],spr_addr[11:0]  
    result: spr_data[63:0]  
    """
    addr &= 0xFFFF

    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM
                and npu.warp.curr_id < GTX_SPU_NUM):
            return npu.lspr[npu.warp.tmu_id][npu.warp.curr_id].get(addr, 0)
        return npu.lspr[0][0].get(addr, 0)

    if _in_range(addr, NSPR_BASE, NSPR_END):
        nid = npu.warp.tmu_id if (npu.warp.is_ploop and
                                  npu.warp.tmu_id < GTX_NEST_NUM) else 0
        return npu.nspr[nid].get(addr, 0)

    if _in_range(addr, GSPR_BASE, GSPR_END):
        return npu.gspr.get(addr, 0)

    return 0

@handler(kind='custom0', funct7=GTX_ISS_F7_RDSPR_ISS, mnemonic='rdspr')
def rdspr_full(npu: GtxNpu, proc, insn, xs1, xs2):
    """Full-encoding RDSPR (funct7=0x48): addr from XPR[rs1], return rd_spr(addr)."""
    state = proc.state
    addr = state.XPR[insn.rs1]
    val = rd_spr(npu, addr & 0xFFFF)
    if insn.rd != 0:
        state.XPR.write(insn.rd, val)
    return val




def wr_spr(npu, addr: int, value: int) -> None:
    """Write SPR.
    Port of gtx_npu_t::wr_spr (gtx_npu_spr.cc:16-78).
    rs3 is for masking broadcast only, spu or nest is selected  by target address
    operand1: spr_addr[11:0], wrstb_n[23:16]
    operand2: spr_data[63:0]
    operand3: *target_mask[63:0]
    """
    addr &= 0xFFFF

    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM
                and npu.warp.curr_id < GTX_SPU_NUM):
            npu.lspr[npu.warp.tmu_id][npu.warp.curr_id][addr] = value
        elif npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            # P-loop: same value into every SPU's LSPR within the active
            # nest. C++ vendor writes each SPU RF separately
            # (gtx_npu_spr.cc:24-25) — semantically equivalent to the
            # docstring's "broadcast across SPUs in the NEST", so a
            # tight per-SPU loop matches both.
            nest_lsprs = npu.lspr[npu.warp.tmu_id]
            for spu_rf in nest_lsprs:
                spu_rf[addr] = value
        else:
            npu.lspr[0][0][addr] = value   # fallback NEST 0, SPU 0
        return

    if _in_range(addr, NSPR_BASE, NSPR_END):
        if npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            npu.nspr[npu.warp.tmu_id][addr] = value
        else:
            npu.nspr[0][addr] = value
        return

    if _in_range(addr, GSPR_BASE, GSPR_END):
        npu.gspr[addr] = value
        return

    # Out-of-range: silently drop (matches C++ behavior -- log only).


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
    """ISS-full OPSET: stage rs3/rs4 for the next instruction.
    #!TODO: 제대로 했는지 확인.
    set operand3(target==0) or operrand_sel(target==1)
    operand1: target 
    operand2: *data[63:0]
    """
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
    #!TODO: 제대로 했는지 확인.
        - copy svr low bytes to upper bytes
    operand1: svr_addr[4:0]
        - rs1 = SVR address/index, rs2 = pattern size [1:0].
    operand2: byte_size[1:0]
        - byte size decoding (0:1byte, 1:2byte, 2:4byte 3:8byte)
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
    """MVSVR (funct7=0x4C): Move 32B L0 SVR register (copy + clear source).
        - move srd to nvr
    #!TODO: 제대로 했는지 확인.
    operand1: src_svr_addr[4:0]
    operand2: dst_svr_addr[4:0]
    operand3: wrstrb_n[31:0]
    result: result[255:0]
    
    """
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
