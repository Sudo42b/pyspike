"""SPR routing + SVR ops — port of vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc.

GSPR (0x000-0x3FF) flat single-instance.
NSPR (0x400-0x7FF) per-NEST  — routed by current_nest when in P-loop, else NEST 0.
LSPR (0x800-0xBFF) per-(NEST,SPU) — routed by (current_nest, current_spu) when
in T-loop, broadcast across SPUs when in P-loop, else (0, 0).

Loop flags (is_ploop/is_tloop) derive from npu.CONTEXT — see context.WarpState.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...inst_handler import inst_register

from ....config_params import NEST_NUM, SPU_NUM
from ... import _resolve_nest_spu
from ....csr import (GSPR_BASE, GSPR_END, NSPR_BASE, NSPR_END,
                     LSPR_BASE, LSPR_END, GSPR)

if TYPE_CHECKING:
    from ....npu import GtxNpu

_OPERAND3_ADDR = GSPR['GSPR_GTX_OPERAND3'].address & 0x3FF   # 0x003
_OPERAND5_ADDR = GSPR['GSPR_GTX_OPERAND5'].address & 0x3FF   # 0x005


def _in_range(addr: int, base: int, end: int) -> bool:
    return base <= addr <= end


# ─── SPR read/write helpers (context-routed) ──────────────────────────────
def rd_spr(npu: "GtxNpu", addr: int) -> int:
    addr &= 0xFFFF
    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.current_nest < NEST_NUM
                and npu.warp.current_spu < SPU_NUM):
            return npu.lspr[npu.warp.current_nest][npu.warp.current_spu].get(addr, 0)
        return npu.lspr[0][0].get(addr, 0)
    if _in_range(addr, NSPR_BASE, NSPR_END):
        nid = npu.warp.current_nest if (npu.warp.is_ploop
                                        and npu.warp.current_nest < NEST_NUM) else 0
        return npu.nspr[nid].get(addr, 0)
    if _in_range(addr, GSPR_BASE, GSPR_END):
        return npu.gspr.get(addr, 0)
    return 0


def wr_spr(npu: "GtxNpu", addr: int, value: int) -> None:
    """Write SPR. Port of gtx_npu_t::wr_spr (gtx_npu_spr.cc:16-78)."""
    addr &= 0xFFFF
    if _in_range(addr, LSPR_BASE, LSPR_END):
        if (npu.warp.is_tloop and npu.warp.current_nest < NEST_NUM
                and npu.warp.current_spu < SPU_NUM):
            npu.lspr[npu.warp.current_nest][npu.warp.current_spu][addr] = value
        elif npu.warp.is_ploop and npu.warp.current_nest < NEST_NUM:
            # P-loop: broadcast the same value into every SPU's LSPR in the nest.
            for spu_rf in npu.lspr[npu.warp.current_nest]:
                spu_rf[addr] = value
        else:
            npu.lspr[0][0][addr] = value
        return
    if _in_range(addr, NSPR_BASE, NSPR_END):
        if npu.warp.is_ploop and npu.warp.current_nest < NEST_NUM:
            npu.nspr[npu.warp.current_nest][addr] = value
        else:
            npu.nspr[0][addr] = value
        return
    if _in_range(addr, GSPR_BASE, GSPR_END):
        npu.gspr[addr] = value
        return
    # Out-of-range: silently drop (vendor logs only).


# ─── ISS-full SPR handlers ────────────────────────────────────────────────
@inst_register.custom0(funct7=0b1001000, funct3=0b000, name='rdspr')
def rdspr(npu: "GtxNpu", proc, inst, cxt) -> int:
    # rs1    """nest_id[29:24],spu_id[21:16],spr_addr[11:0]"""
    # rsult(gpr) spr_data[63:0]
    addr = proc.state.XPR[inst.rs1]
    val = rd_spr(npu, addr & 0xFFFF)
    if inst.rd != 0:
        proc.state.XPR.write(inst.rd, val)
    return val


@inst_register.custom0(funct7=0b1001001, funct3=0b000, name='wrspr')
def wrspr(npu: "GtxNpu", proc, inst, cxt) -> int:
    """WRSPR (funct7=0x49): addr from XPR[rs1], value from XPR[rs2]."""
    addr = proc.state.XPR[inst.rs1]
    val = proc.state.XPR[inst.rs2]
    wr_spr(npu, addr & 0xFFFF, val)
    return 0


@inst_register.custom0(funct7=0b1001010, funct3=0b000, name='opset')
def opset(npu: "GtxNpu", proc, inst, cxt) -> int:
    """OPSET (funct7=0x4A): stage operand for the next instruction.

    rs1 LSB selects the slot: 0 → GSPR_OPERAND3, 1 → GSPR_OPERAND5.
    Verified against gtx_npu_custom0.cc:115-131.
    
    set operand3(target==0) or operrand_sel(target==1)
    
    """
    slot = proc.state.XPR[inst.rs1] & 1
    val = proc.state.XPR[inst.rs2]
    npu.gspr.tensor[_OPERAND5_ADDR if slot else _OPERAND3_ADDR] = val
    return 0


@inst_register.custom0(funct7=0b1001011, funct3=0b000, name='cpsvr')
def cpsvr(npu: "GtxNpu", proc, inst, cxt) -> int:
    """CPSVR (funct7=0x4B): replicate an L0 SVR byte pattern across 32 B.

    base = ((rs1 & 0x1F) / 4) * 32 (4 words per SVR); rs2[1:0] = byte size
    (0:1B 1:2B 2:4B 3:8B). The 8 B pattern fills the 32 B SVR.
    Verified against gtx_npu_custom0.cc:133-172.
    """
    rs1_val = proc.state.XPR[inst.rs1]
    rs2_val = proc.state.XPR[inst.rs2]
    nest, spu = _resolve_nest_spu(npu)
    base = ((rs1_val & 0x1F) // 4) * 32
    bsz = rs2_val & 3
    l0 = npu.mem.l0_byte(nest, spu)
    if bsz == 0:
        p = l0[base].repeat(8)
    elif bsz == 1:
        p = l0[base:base + 2].repeat(4)
    elif bsz == 2:
        p = l0[base:base + 4].repeat(2)
    else:
        p = l0[base:base + 8]
    l0[base:base + 32] = p.repeat(4)
    return 0


@inst_register.custom0(funct7=0b1001100, funct3=0b000, name='mvsvr')
def mvsvr(npu: "GtxNpu", proc, inst, cxt) -> int:
    """MVSVR (funct7=0x4C): move a 32 B L0 SVR register (copy + clear source).

    src_off = (rs1 & 0x1F) * 32, dst_off = (rs2 & 0x1F) * 32.
    Verified against gtx_npu_custom0.cc:174-190. Self-move short-circuits.
    """
    rs1_val = proc.state.XPR[inst.rs1]
    rs2_val = proc.state.XPR[inst.rs2]
    nest, spu = _resolve_nest_spu(npu)
    src_idx = rs1_val & 0x1F
    dst_idx = rs2_val & 0x1F
    if src_idx == dst_idx:
        return 0
    l0 = npu.mem.l0_byte(nest, spu)
    src_off = src_idx * 32
    dst_off = dst_idx * 32
    l0[dst_off:dst_off + 32] = l0[src_off:src_off + 32].clone()
    l0[src_off:src_off + 32].zero_()
    return 0
