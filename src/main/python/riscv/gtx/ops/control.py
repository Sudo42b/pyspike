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

from .._registry import handler
from ..warp_state import WarpState   # noqa: F401  -- type hint reference
from ..params import GTX_NEST_NUM, GTX_SPU_NUM


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
def _do_startp(npu, rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startp. Sets is_ploop, tmu_id."""
    nest_id = _extract_id(rs1, rs2)
    if nest_id >= GTX_NEST_NUM:
        nest_id = 0
    npu.warp.tmu_id = nest_id
    npu.warp.is_ploop = True


def _do_endp(npu, rs1: int, rs2: int) -> None:
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


def _do_startt(npu, rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startt. Sets is_tloop, curr_id."""
    spu_id = _extract_id(rs1, rs2)
    if spu_id >= GTX_SPU_NUM:
        spu_id = 0
    npu.warp.curr_id = spu_id
    npu.warp.is_tloop = True


def _do_endt(npu, rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::endt. Clears is_tloop."""
    npu.warp.is_tloop = False


def _do_starts(npu, rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::starts. P3 DMA: sets is_sloop, curr_id (GDMAC).

    GTX_GDMAC_NUM == GTX_NUM_NESTS == 4 in the C++ reference, so we clamp
    against GTX_NEST_NUM.
    """
    gdmac_id = _extract_id(rs1, rs2)
    if gdmac_id >= GTX_NEST_NUM:   # GDMAC_NUM == NEST_NUM
        gdmac_id = 0
    npu.warp.curr_id = gdmac_id
    npu.warp.is_sloop = True


def _do_ends(npu, rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::ends. Clears is_sloop."""
    npu.warp.is_sloop = False


# ============================================================================
# custom1 funct3 handlers -- DISP-02
# (each reads rs1/rs2 directly via proc.state.XPR per CORE-04, then
#  delegates to the matching _do_* helper.)
# ============================================================================
@handler(kind='custom1', funct3=0b000, mnemonic='warp_start_t')
def startt(npu, proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_startt(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b001, mnemonic='warp_end_t')
def endt(npu, proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_endt(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b010, mnemonic='warp_start_s')
def starts(npu, proc, insn, xs1, xs2):
    """P3 DMA: full implementation in phase 03. P2 still wires _do_starts so
    spr_router.wr_spr(GSPR_STARTS, ..) flag-only side-effect works."""
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_starts(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b011, mnemonic='warp_end_s')
def ends(npu, proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_ends(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b100, mnemonic='warp_split')
def wsplit(npu, proc, insn, xs1, xs2):
    """WSPLIT -- start timing section. P3 (Plan 05): sets wsplit_seen sentinel.

    The wsplit_seen flag determines which deferred-store flush trigger fires:
    when set, end_p suppresses its flush so plan-style firmware (which flushes
    mid-execution via credit_st_chk) doesn't double-flush. See 03-RESEARCH
    "Deferred-Store Flush Trigger".
    """
    npu.warp.wsplit_seen = True
    return 0


@handler(kind='custom1', funct3=0b101, mnemonic='warp_join')
def wjoin_with_exit(npu, proc, insn, xs1, xs2):
    """WJOIN -- CORE-03 / D-07 / D-08.

    Per D-07: read GTX_NO_EXIT every call (no caching).

    Truthiness rule (matches Python bool()):
        unset / empty string  -> raises SystemExit (testable via pytest.raises)
        any non-empty value   -> returns 0 (firmware loop continues)

    Note: '0' literally is non-empty, so it is treated as truthy -> return 0.
    Tests in tests/gtx/test_wjoin.py exercise both branches (D-08).
    """
    if os.environ.get('GTX_NO_EXIT'):
        return 0
    raise SystemExit(0)


@handler(kind='custom1', funct3=0b110, mnemonic='warp_start_p')
def startp(npu, proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_startp(npu, rs1_val, rs2_val)
    return 0


@handler(kind='custom1', funct3=0b111, mnemonic='warp_end_p')
def endp(npu, proc, insn, xs1, xs2):
    state = proc.state
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_endp(npu, rs1_val, rs2_val)
    return 0


# ============================================================================
# custom0 funct7 stubs -- skeleton phase only (P3+ replaces with real handlers)
# ============================================================================
@handler(kind='custom0', funct7=0x02, mnemonic='wsplit_c0')
def wsplit_custom0(npu, proc, insn, xs1, xs2):
    """custom0 funct7=0x02 WSPLIT (firmware variant). P3 (Plan 05): sets
    wsplit_seen sentinel -- mirror of custom1 funct3=0b100 wsplit handler.
    See 03-RESEARCH "Deferred-Store Flush Trigger".
    """
    npu.warp.wsplit_seen = True
    return 0


@handler(kind='custom0', funct7=0x03, mnemonic='wjoin_c0')
def wjoin_custom0_no_exit(npu, proc, insn, xs1, xs2):
    """custom0 funct7=0x03 WJOIN firmware variant -- NEVER raises SystemExit
    (research §439: only custom1 funct3=0b101 has exit semantics).

    Returns 0 (placeholder for P3+ "elapsed cycles" return value).
    """
    return 0


@handler(kind='custom0', funct7=0x04, mnemonic='dispatch_mm')
def dispatch_mm_stub(npu, proc, insn, xs1, xs2):
    """P3+: dispatch_mm. P2: NOP."""
    return 0


@handler(kind='custom0', funct7=0x05, mnemonic='dispatch_vec')
def dispatch_vec_stub(npu, proc, insn, xs1, xs2):
    """P3+: dispatch_vec. P2: NOP."""
    return 0


@handler(kind='custom0', funct7=0x06, mnemonic='dispatch_act')
def dispatch_act_stub(npu, proc, insn, xs1, xs2):
    """P3+: dispatch_act. P2: NOP."""
    return 0


@handler(kind='custom0', funct7=0x07, mnemonic='dispatch_dma')
def dispatch_dma_stub(npu, proc, insn, xs1, xs2):
    """P3+: dispatch_dma. P2: NOP."""
    return 0
