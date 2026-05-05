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
"""Tests for DISP-02 (custom1 funct3 dispatch + warp loop state machine).

All tests run on mocks (no _riscv.so required) -- call handler functions
directly with a SimpleNamespace fake npu.
"""
from types import SimpleNamespace

from riscv.gtx.ops.control import (
    _extract_id, _do_startp, _do_endp, _do_startt, _do_endt,
    _do_starts, _do_ends,
    startp, endp, startt, endt, starts, ends, wsplit,
)
from riscv.gtx.warp_state import WarpState


def _fake_npu():
    """Stub npu shim with WarpState + a no-op flush_deferred_ddr_stores.

    Plan 05 wired _do_endp to call npu.flush_deferred_ddr_stores() when
    !wsplit_seen, so even a SimpleNamespace fake needs to expose the method
    (default WarpState has wsplit_seen=False so the flush path is exercised).
    """
    return SimpleNamespace(
        warp=WarpState(),
        deferred_ddr_stores=[],
        flush_deferred_ddr_stores=lambda: None,
    )


class _FakeProc:
    def __init__(self):
        from tests.gtx._mocks import MockProcessor
        self._mp = MockProcessor()

    def get_state(self):
        return self._mp.get_state()


# ----------------- _extract_id (dual-mode addressing) -----------------

def test_extract_id_rs1_path():
    """rs2 & 0x400 == 0 -> use rs1 & 0xFFFFFFFF."""
    assert _extract_id(rs1=2, rs2=0) == 2
    assert _extract_id(rs1=15, rs2=0x100) == 15  # bit 0x400 not set


def test_extract_id_rs2_marker_path():
    """rs2 & 0x400 != 0 -> use rs2 & 0x3F (low 6 bits)."""
    assert _extract_id(rs1=99, rs2=0x405) == 0x405 & 0x3F   # = 5
    assert _extract_id(rs1=99, rs2=0x43F) == 0x3F


def test_extract_id_rs1_truncated_to_32_bits():
    """rs1 path is rs1 & 0xFFFFFFFF (no upper-bit leak)."""
    assert _extract_id(rs1=0x100000003, rs2=0) == 3


# ----------------- _do_startp / _do_endp -----------------

def test_do_startp_sets_is_ploop_and_tmu_id():
    npu = _fake_npu()
    _do_startp(npu, rs1=2, rs2=0)
    assert npu.warp.is_ploop is True
    assert npu.warp.tmu_id == 2


def test_do_startp_clamps_out_of_range_nest():
    """nest_id >= GTX_NEST_NUM (4) -> clamp to 0."""
    npu = _fake_npu()
    _do_startp(npu, rs1=99, rs2=0)
    assert npu.warp.is_ploop is True
    assert npu.warp.tmu_id == 0


def test_do_endp_clears_is_ploop():
    """end_p clears the flag only -- tmu_id is NOT zeroed (matches C++)."""
    npu = _fake_npu()
    npu.warp.is_ploop = True
    npu.warp.tmu_id = 3
    _do_endp(npu, rs1=0, rs2=0)
    assert npu.warp.is_ploop is False


# ----------------- _do_startt / _do_endt -----------------

def test_do_startt_sets_is_tloop_and_curr_id():
    npu = _fake_npu()
    _do_startt(npu, rs1=7, rs2=0)
    assert npu.warp.is_tloop is True
    assert npu.warp.curr_id == 7


def test_do_startt_clamps_out_of_range_spu():
    """spu_id >= GTX_SPU_NUM (16) -> clamp to 0."""
    npu = _fake_npu()
    _do_startt(npu, rs1=99, rs2=0)
    assert npu.warp.is_tloop is True
    assert npu.warp.curr_id == 0


def test_do_endt_clears_is_tloop():
    npu = _fake_npu()
    npu.warp.is_tloop = True
    _do_endt(npu, rs1=0, rs2=0)
    assert npu.warp.is_tloop is False


# ----------------- _do_starts / _do_ends (P3 DMA stub but flag works in P2) -----------------

def test_do_starts_sets_is_sloop():
    npu = _fake_npu()
    _do_starts(npu, rs1=1, rs2=0)
    assert npu.warp.is_sloop is True
    assert npu.warp.curr_id == 1


def test_do_ends_clears_is_sloop():
    npu = _fake_npu()
    npu.warp.is_sloop = True
    _do_ends(npu, rs1=0, rs2=0)
    assert npu.warp.is_sloop is False


# ----------------- custom1 handlers (read XPR, then call _do_*) -----------------

def test_startp_handler_reads_xpr_and_sets_state():
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(2, 1)        # rs1 = x2 -> nest_id 1
    insn = MockInsn(rs1=2, rs2=0)
    ret = startp(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.warp.is_ploop is True
    assert npu.warp.tmu_id == 1


def test_startt_handler_full_path():
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(3, 5)
    insn = MockInsn(rs1=3, rs2=0)
    ret = startt(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.warp.is_tloop is True
    assert npu.warp.curr_id == 5


def test_wsplit_handler_is_nop():
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    snapshot = (npu.warp.is_ploop, npu.warp.is_tloop, npu.warp.is_sloop)
    ret = wsplit(npu, proc, MockInsn(), xs1=0, xs2=0)
    assert ret == 0
    # No state changed
    assert (npu.warp.is_ploop, npu.warp.is_tloop, npu.warp.is_sloop) == snapshot


# ----------------- ROADMAP P2 success criterion 4 -----------------

def test_loop_state_machine_full_sequence():
    """ROADMAP P2 #4: start_p -> start_t -> end_t -> end_p must end in
    (is_ploop=False, is_tloop=False) -- no flag leak.

    NB: tmu_id/curr_id are NOT reset by end_p/end_t (only by full
    WarpState.reset()) -- this matches verbatim C++ semantics.
    """
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # Pre-load registers: x2=NEST 2, x3=SPU 5
    proc.get_state().XPR.write(2, 2)
    proc.get_state().XPR.write(3, 5)

    # 1. start_p (funct3=0b110, rs1=x2)
    startp(npu, proc, MockInsn(rs1=2, rs2=0), xs1=0, xs2=0)
    assert npu.warp.is_ploop is True and npu.warp.tmu_id == 2

    # 2. start_t (funct3=0b000, rs1=x3)
    startt(npu, proc, MockInsn(rs1=3, rs2=0), xs1=0, xs2=0)
    assert npu.warp.is_tloop is True and npu.warp.curr_id == 5
    assert npu.warp.is_ploop is True   # P-loop still active

    # 3. end_t (funct3=0b001)
    endt(npu, proc, MockInsn(rs1=0, rs2=0), xs1=0, xs2=0)
    assert npu.warp.is_tloop is False
    assert npu.warp.is_ploop is True   # P-loop still active

    # 4. end_p (funct3=0b111)
    endp(npu, proc, MockInsn(rs1=0, rs2=0), xs1=0, xs2=0)
    assert npu.warp.is_ploop is False
    assert npu.warp.is_tloop is False


def test_warp_state_reset_clears_all():
    """WarpState.reset() returns to fresh state (used by GtxNpu.reset())."""
    npu = _fake_npu()
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 3
    npu.warp.curr_id = 7
    npu.warp.reset()
    assert npu.warp.is_ploop is False
    assert npu.warp.is_tloop is False
    assert npu.warp.is_sloop is False
    assert npu.warp.tmu_id == 0
    assert npu.warp.curr_id == 0
