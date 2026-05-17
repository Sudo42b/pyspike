"""Warp / control ops -- DISP-02 + CORE-03.

Handlers:
  custom1 funct3 = 0b000..0b111 (8 warp-control variants -- port gtx_npu_custom1.cc)
  custom0 funct7 = 0x02 (wsplit), 0x03 (wjoin firmware variant, NO exit per
                         research §439), 0x04..0x07 (dispatch_* P3+ stubs)

Internal _do_* helpers shared with spr_router (loop-control GSPR addresses
0x100..0x105). Helpers take a (rs1, rs2) value pair so they are reusable
both from custom1 handlers (which read XPR first) and from wr_spr side-effects.

References:
  - vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:21-142 (verbatim port of
    extract_id + startp/endp/startt/endt/starts/ends).
  - vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc:29-137 (8 funct3 dispatch +
    WJOIN env-var branch).
"""
import os
import sys

from ..._registry import handler
from .warp_state import WarpState   # noqa: F401  -- type hint reference
from ...config_params import GTX_NEST_NUM, GTX_SPU_NUM
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...npu import GtxNpu

# Per-tile progress marker for the pytest regression harness. Off by default;
# the harness sets GTX_PROGRESS=1 and reads each emitted line to drive a tqdm
# bar. One line per wjoin call (≈ one line per firmware tile).
_PROGRESS_TAG = "[GTX_PROGRESS] wjoin"


def _emit_progress() -> None:
    if os.environ.get("GTX_PROGRESS") == "1":
        print(_PROGRESS_TAG, file=sys.stderr, flush=True)


# ============================================================================
# extract_id -- gtx_npu_loop.cc:21-23 dual-mode addressing
# ============================================================================
def _extract_id(rs1: int, rs2: int) -> int:
    """Dual-mode addressing: rs2 marker bit selects rs2 low6 vs rs1 low32.

    Verbatim port of gtx_npu_loop.cc:21-23 (the marker-bit ternary).
    """
    if rs2 & 0x400:
        return rs2 & 0x3F
    return rs1 & 0xFFFFFFFF


# ============================================================================
# _do_* helpers -- value-level loop transitions, callable from custom1 handler
# AND from spr_router.wr_spr (loop-control GSPR addresses 0x100..0x105).
# ============================================================================
def _do_startp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startp. Sets is_ploop, tmu_id."""
    nest_id = _extract_id(rs1, rs2)
    assert 0 <= nest_id < GTX_NEST_NUM, f"Invalid NEST ID {nest_id} in startp (is_ploop={npu.warp.is_ploop})"
    npu.warp.tmu_id = nest_id
    npu.warp.is_ploop = True


def _do_endp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::endp. Clears is_ploop. P3 (Plan 05): flushes the
    deferred-store queue when !wsplit_seen.

    RESEARCH "Deferred-Store Flush Trigger" #1: simple firmware (no WSPLIT)
    flushes here at end_p. Plan-style firmware (with WSPLIT) flushes via
    credit_st_chk mid-execution instead -- see ops/dma.py:_credit_st_chk.
    The wsplit_seen sentinel chooses the path. ROADMAP P3 success #4 path.
    """
    npu.warp.is_ploop = False
    if not npu.warp.wsplit_seen:
        npu.flush_deferred_ddr_stores()


def _do_startt(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startt. Sets is_tloop, curr_id.

    Also opens the T-loop instruction buffer (see :mod:`gtx.tloop_buffer`)
    so subsequent bufferable mnemonics are captured for replay-at-endt
    instead of executing immediately.
    """
    spu_id = _extract_id(rs1, rs2)
    assert spu_id < GTX_SPU_NUM, f"Invalid SPU ID {spu_id} in startt (is_tloop={npu.warp.is_tloop})"
    npu.warp.curr_id = spu_id
    npu.warp.is_tloop = True
    # Hard kill-switch: ``GTX_TLOOP_DISABLE=1`` keeps the FSM on the eager
    # path while leaving the buffer wiring in place, so we can A/B against
    # the in-order replay path without reverting the patch.
    if not os.environ.get("GTX_TLOOP_DISABLE"):
        npu._tloop_buf = []


def _do_endt(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::endt. Clears is_tloop.

    Drains any buffered T-loop instructions BEFORE clearing ``is_tloop``
    so replayed handlers see the warp state they were captured under.
    """
    if npu._tloop_buf:
        from ...tloop_buffer import flush as _flush_tloop_buf
        _flush_tloop_buf(npu)
    npu._tloop_buf = None
    npu.warp.is_tloop = False


def _do_starts(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::starts. P3 DMA: sets is_sloop, curr_id (GDMAC).

    GTX_GDMAC_NUM == GTX_NUM_NESTS == 4 in the C++ reference, so we clamp
    against GTX_NEST_NUM.

    Also opens the S-loop instruction buffer (see :mod:`gtx.sloop_buffer`)
    so subsequent bufferable SMU mnemonics are captured for credit-gated
    dequeue at ``credit_st_chk`` (or full drain at ``_do_ends``) instead
    of executing immediately. Mirror of ``_do_startt`` for the SMU side.
    """
    gdmac_id = _extract_id(rs1, rs2)
    assert 0 <= gdmac_id < GTX_NEST_NUM, f"Invalid GDMAC ID {gdmac_id} in starts (is_sloop={npu.warp.is_sloop})"
    npu.warp.curr_id = gdmac_id
    npu.warp.is_sloop = True
    # Hard kill-switch: ``GTX_SLOOP_DISABLE=1`` keeps the FSM on the eager
    # path while leaving the buffer wiring in place, parity with the
    # ``GTX_TLOOP_DISABLE`` escape hatch used by tloop_buffer A/B tests.
    if not os.environ.get("GTX_SLOOP_DISABLE"):
        npu._sloop_buf = []


def _do_ends(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::ends. Clears is_sloop.

    Drains any buffered S-loop instructions BEFORE clearing ``is_sloop``
    so replayed handlers see the warp state they were captured under
    (mirror of ``_do_endt`` for the SMU side).
    """
    if npu._sloop_buf:
        from ...sloop_buffer import flush as _flush_sloop_buf
        _flush_sloop_buf(npu)
    npu._sloop_buf = None
    npu.warp.is_sloop = False


# ============================================================================
# custom1 funct3 handlers -- DISP-02
# (each reads rs1/rs2 directly via proc.state.XPR per CORE-04, then
#  delegates to the matching _do_* helper.)
# ============================================================================
@handler(kind='custom1', funct3=0b000, mnemonic='start.t')
def startt(npu: "GtxNpu", proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_startt(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b001, mnemonic='end.t')
def endt(npu: "GtxNpu", proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_endt(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b010, mnemonic='start.s')
def starts(npu: "GtxNpu", proc, insn, xs1, xs2):
    """P3 DMA: full implementation in phase 03. P2 still wires _do_starts so
    spr_router.wr_spr(GSPR_STARTS, ..) flag-only side-effect works."""
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_starts(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b011, mnemonic='end.s')
def ends(npu: "GtxNpu", proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_ends(npu, rs1_val, rs2_val)
    return 0

@handler(kind='custom1', funct3=0b110, mnemonic='start.p')
def startp(npu: "GtxNpu", proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_startp(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b111, mnemonic='end.p')
def endp(npu: "GtxNpu", proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_endp(npu, rs1_val, rs2_val)
    return 0

@handler(kind='custom1', funct3=0b100, mnemonic='split')
def wsplit(npu: "GtxNpu", proc, insn, xs1, xs2):
    """WSPLIT -- start timing section. P3 (Plan 05): sets wsplit_seen sentinel.

    The wsplit_seen flag determines which deferred-store flush trigger fires:
    when set, end_p suppresses its flush so plan-style firmware (which flushes
    mid-execution via credit_st_chk) doesn't double-flush. See 03-RESEARCH
    "Deferred-Store Flush Trigger".
    """
    npu.warp.wsplit_seen = True
    return 0


@handler(kind='custom1', funct3=0b101, mnemonic='join')
def wjoin_with_exit(npu: "GtxNpu", proc, insn, xs1, xs2):
    """WJOIN — flush deferred stores and emit progress, no exit.

    Multi split-join firmware (e.g. abs.elf with 97 tiles) calls WJOIN
    once per tile, so this handler no longer raises ``SystemExit`` —
    spike continues into the next tile's instructions and exits
    naturally when the firmware's ``main`` returns. The DDR dump
    happens once via :class:`GtxNpu`'s ``atexit`` hook instead of
    every join.
    """
    npu.flush_deferred_ddr_stores()
    _emit_progress()
    return 0

