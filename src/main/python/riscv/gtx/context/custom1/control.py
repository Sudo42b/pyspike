"""Warp / loop control — custom1 funct3 dispatch (8 variants).

Each marker drives the persistent NPU context (``npu.CONTEXT``) and the routed
IDs (``npu.warp.current_nest`` / ``current_spu``); the loop flags
``is_ploop`` / ``is_sloop`` / ``is_tloop`` are derived from CONTEXT.

    start.p : C1 → C4   (current_nest = NEST id)
    end.p   : C4 → C1   (flush deferred stores when !wsplit_seen)
    start.s : C4 → C2   (current_spu = GDMAC id)
    end.s   : C2 → C4
    start.t : C4 → C3   (current_spu = SPU id)
    end.t   : C3 → C4
    split   : set wsplit_seen sentinel
    join    : flush deferred stores (no exit — multi-tile firmware joins per tile)

Buffering (T/S-loop instruction replay) is intentionally disabled — execution
is eager. References:
  vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:21-142,
  vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc:29-137.
"""
import sys
from typing import TYPE_CHECKING

from ..inst_handler import inst_register
from ..exec_st import CXT
from ...config_params import NEST_NUM, SPU_NUM

if TYPE_CHECKING:
    from ...npu import GtxNpu

def _extract_id(rs1: int, rs2: int) -> int:
    """Dual-mode addressing — rs2 marker bit (0x400) selects rs2 low6 vs rs1 low32.

    Verbatim port of gtx_npu_loop.cc:21-23.
    """
    if rs2 & 0x400:
        return rs2 & 0x3F
    return rs1 & 0xFFFFFFFF


@inst_register.custom1(name='start.p', funct3=0b110)
def startp(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C1 → C4. Select the active NEST."""
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    nest_id = _extract_id(rs1, rs2)
    assert 0 <= nest_id < NEST_NUM, f"Invalid NEST id {nest_id} in start.p"
    npu.warp.current_nest = nest_id
    npu.CONTEXT = CXT.C4
    return 0


@inst_register.custom1(name='end.p', funct3=0b111)
def endp(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C4 → C1. Simple (non-WSPLIT) firmware flushes deferred stores here."""
    npu.CONTEXT = CXT.C1
    if not npu.warp.wsplit_seen:
        npu.flush_deferred_ddr_stores()
    return 0


@inst_register.custom1(name='start.s', funct3=0b010)
def starts(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C4 → C2. Select the active GDMAC (clamped to NEST count, vendor parity)."""
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    gdmac_id = _extract_id(rs1, rs2)
    assert 0 <= gdmac_id < NEST_NUM, f"Invalid GDMAC id {gdmac_id} in start.s"
    npu.warp.current_spu = gdmac_id
    npu.CONTEXT = CXT.C2
    return 0


@inst_register.custom1(name='end.s', funct3=0b011)
def ends(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C2 → C4."""
    npu.CONTEXT = CXT.C4
    return 0


@inst_register.custom1(name='start.t', funct3=0b000)
def startt(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C4 → C3. Select the active SPU and arm the T-loop buffer.

    Arming lets GtxNpu.custom0 capture the (load, vec, store) inner
    cadence so end.t can flush it as fused bulk torch ops.
    """
    rs1 = proc.state.XPR[inst.rs1]
    rs2 = proc.state.XPR[inst.rs2]
    spu_id = _extract_id(rs1, rs2)
    assert 0 <= spu_id < SPU_NUM, f"Invalid SPU id {spu_id} in start.t"
    npu.warp.current_spu = spu_id
    npu.CONTEXT = CXT.C3
    npu._tloop_buf = [] if npu._fusion_enabled else None
    return 0


@inst_register.custom1(name='end.t', funct3=0b001)
def endt(npu: "GtxNpu", proc, inst, cxt) -> int:
    """C3 → C4. The pending buffer was already drained by GtxNpu.custom1
    on entry to this marker; just disarm so post-loop ops run eagerly."""
    npu.CONTEXT = CXT.C4
    npu._tloop_buf = None
    return 0


@inst_register.custom1(name='split', funct3=0b100)
def split(npu: "GtxNpu", proc, inst, cxt) -> int:
    """WSPLIT — set the process-lifetime sentinel that suppresses end.p flush
    (plan-style firmware flushes mid-execution via credit.st.chk instead)."""
    npu.warp.wsplit_seen = True
    return 0


@inst_register.custom1(name='join', funct3=0b101)
def join(npu: "GtxNpu", proc, inst, cxt) -> int:
    """WJOIN — flush deferred stores; no exit (multi-tile firmware joins per
    tile and exits naturally; DDR dump runs via the atexit hook).

    Strict credit (vendor ISS ``NSU::wjoin``, NSU.cpp:451): every load/store
    credit must be balanced (== 0) at WJOIN. A remaining credit means firmware
    left a DMA↔compute hand-off unconsumed — report it (non-fatal, like the ISS)."""
    npu.flush_deferred_ddr_stores()
    if npu._credit_ld.any() or npu._credit_st.any():
        for nest in range(NEST_NUM):
            for spu in range(SPU_NUM):
                ld = int(npu._credit_ld[nest, spu])
                st = int(npu._credit_st[nest, spu])
                if ld or st:
                    print(f"[GTX_CREDIT_REMAINED] nest{nest} spu{spu}: "
                          f"ld={ld} st={st} (plz check firmware)",
                          file=sys.stderr, flush=True)
    return 0
