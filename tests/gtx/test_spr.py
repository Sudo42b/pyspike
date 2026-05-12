"""Tests for SPR-01 (routing) and SPR-02 (WRSPR/RDSPR handlers).

All tests run on mocks (no _riscv.so required) since spr_router and op
handlers are pure Python operating on dict storage.
"""
from types import SimpleNamespace

from riscv.gtx.spr_router import wr_spr, rd_spr
from riscv.gtx.unit.ins.ops.spr import (wrspr_iss, rdspr_iss,
                               wrspr_gem5, rdspr_gem5)
from riscv.gtx.unit.context.warp_state import WarpState
from riscv.gtx.config_params import GTX_NEST_NUM, GTX_SPU_NUM


def _fake_npu():
    """Minimal npu surface that wr_spr / rd_spr / handlers need.

    P4 plan 04 expansion: wrspr_gem5/rdspr_gem5 re-dispatch to MM/MMC funct3-keyed
    handlers via npu._custom0.get(funct7, {}).get(funct3); shim grew _custom0={}
    so the rs1!=0 path can NOP-fallback when no funct3 entry exists (matches the
    test's "rs1!=0 -> stub returns 0" expectation since no MM handler is wired
    in this lightweight shim context).
    """
    return SimpleNamespace(
        gspr={},
        nspr=[{} for _ in range(GTX_NEST_NUM)],
        lspr=[[{} for _ in range(GTX_SPU_NUM)] for _ in range(GTX_NEST_NUM)],
        warp=WarpState(),
        _custom0={},  # P4 plan 04: stub for re-dispatch fallback
    )


# ------------- SPR-01: routing -----------------

def test_routing_gspr_roundtrip():
    npu = _fake_npu()
    wr_spr(npu, 0x010, 0xDEAD)
    assert rd_spr(npu, 0x010) == 0xDEAD
    assert npu.gspr[0x010] == 0xDEAD


def test_routing_nspr_with_ploop():
    npu = _fake_npu()
    npu.warp.is_ploop = True
    npu.warp.tmu_id = 2
    wr_spr(npu, 0x500, 0x1234)
    assert npu.nspr[2][0x500] == 0x1234
    assert rd_spr(npu, 0x500) == 0x1234


def test_routing_nspr_no_ploop_falls_back_to_nest_0():
    npu = _fake_npu()
    wr_spr(npu, 0x500, 0xCAFE)
    assert npu.nspr[0][0x500] == 0xCAFE


def test_routing_lspr_with_tloop_targets_specific_spu():
    npu = _fake_npu()
    npu.warp.is_tloop = True
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 7
    wr_spr(npu, 0x900, 0xCAFE)
    assert npu.lspr[1][7][0x900] == 0xCAFE
    # Other SPUs untouched
    assert 0x900 not in npu.lspr[1][6]
    assert 0x900 not in npu.lspr[2][7]


def test_routing_lspr_ploop_broadcasts_across_spus():
    npu = _fake_npu()
    npu.warp.is_ploop = True
    npu.warp.tmu_id = 3
    wr_spr(npu, 0x901, 0xBEEF)
    for s in range(GTX_SPU_NUM):
        assert npu.lspr[3][s][0x901] == 0xBEEF, f"SPU {s} not broadcast"
    # Other NESTs untouched
    for s in range(GTX_SPU_NUM):
        assert 0x901 not in npu.lspr[0][s]


def test_routing_lspr_no_loop_fallback_to_0_0():
    npu = _fake_npu()
    wr_spr(npu, 0x902, 0xABBA)
    assert npu.lspr[0][0][0x902] == 0xABBA


def test_routing_addr_masked_to_16bits():
    npu = _fake_npu()
    wr_spr(npu, 0x10010, 0xFEED)   # bits beyond 0xFFFF must be ignored
    assert rd_spr(npu, 0x010) == 0xFEED


# ------------- SPR-02: WRSPR/RDSPR via handlers -----------------

class _FakeProc:
    """Minimal proc surface used by handlers -- wraps MockProcessor for tests
    that don't use the conftest fixture."""
    def __init__(self):
        from tests.gtx._mocks import MockProcessor
        self._mp = MockProcessor()

    # Plan 04-05 fix: real pybind11 processor_t exposes `state` as property.
    @property
    def state(self):
        return self._mp.state

    def get_state(self):
        return self._mp.get_state()


def test_wrspr_iss_writes_via_handler():
    """ISS WRSPR (funct7=0x49): addr from rs1, val from rs2."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # Pre-load registers: x10 -> addr=0x010, x11 -> val=0xCAFE
    proc.get_state().XPR.write(10, 0x010)
    proc.get_state().XPR.write(11, 0xCAFE)
    insn = MockInsn(rs1=10, rs2=11)
    ret = wrspr_iss(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.gspr[0x010] == 0xCAFE


def test_rdspr_iss_force_writes_to_rd():
    """ISS RDSPR (funct7=0x48): force-write to rd even if xd=0 (custom0.cc:101-103)."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # Pre-load LSPR[0][0][0x900] = 0xCAFE; expect rd to receive it.
    npu.lspr[0][0][0x900] = 0xCAFE
    proc.get_state().XPR.write(2, 0x900)   # rs1 = x2 -> addr
    insn = MockInsn(rs1=2, rd=5)
    ret = rdspr_iss(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0xCAFE
    assert proc.get_state().XPR[5] == 0xCAFE


def test_rdspr_iss_does_not_write_when_rd_is_x0():
    """rd==0 means writeback to x0 -- must NOT happen (x0 is hardwired)."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    npu.gspr[0x100] = 0xDEAD   # any value (note: 0x100 is GSPR_STARTP, not used here)
    proc.get_state().XPR.write(2, 0x010)
    insn = MockInsn(rs1=2, rd=0)
    ret = rdspr_iss(npu, proc, insn, xs1=0, xs2=0)
    # Return value still propagates (Spike's macro will handle xd=0 separately)
    assert ret == 0   # gspr[0x010] is 0 (not written)
    # XPR[0] stays 0 (hardwired in MockXPR.write)
    assert proc.get_state().XPR[0] == 0


def test_wrspr_gem5_collision_rs1_nonzero_returns_0_no_write():
    """gem5 funct7=0x00 collision (D-02): insn.rs1 != 0 -> P4 MM stub return 0."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(3, 0x010)
    proc.get_state().XPR.write(4, 0xDEAD)
    insn = MockInsn(rs1=3, rs2=4)   # rs1 != 0 -> P4 stub
    snapshot = dict(npu.gspr)
    ret = wrspr_gem5(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.gspr == snapshot   # no write happened


def test_wrspr_gem5_rs1_zero_writes_to_addr_xpr0():
    """D-02 verbatim port: insn.rs1==0 -> addr=XPR[0]=0 -> writes GSPR_GTX_RUN (0x000)."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(4, 0xFEED)   # rs2 = x4
    insn = MockInsn(rs1=0, rs2=4)
    ret = wrspr_gem5(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.gspr[0x000] == 0xFEED   # GSPR_GTX_RUN


def test_rdspr_gem5_rs1_nonzero_returns_0_stub():
    """gem5 funct7=0x01 collision: rs1!=0 -> P4 MMC stub."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(3, 0x100)
    insn = MockInsn(rs1=3)
    ret = rdspr_gem5(npu, proc, insn, xs1=0, xs2=0)
    assert ret == 0


def test_xs1_zero_workaround_proof():
    """CORE-04: handler reads XPR[insn.rs1], NOT the xs1 arg.

    Simulate Spike's marshalling: pass xs1 = 0xFFFFFFFFFFFFFFFF (the unsigned
    reg_t representation of -1) AND populate XPR[2] with the real address.
    Handler must use XPR, ignoring the xs1 marshalled junk.
    """
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    proc.get_state().XPR.write(2, 0x010)       # the real addr
    proc.get_state().XPR.write(3, 0xCAFE)      # the real val
    insn = MockInsn(rs1=2, rs2=3)
    # Pretend Spike marshalled -1 for xs1 (xs1 flag was 0 in the encoding)
    ret = wrspr_iss(npu, proc, insn,
                    xs1=0xFFFFFFFFFFFFFFFF, xs2=0xFFFFFFFFFFFFFFFF)
    assert ret == 0
    # If handler had used the xs1 arg, addr would have been -1 & 0xFFFF = 0xFFFF
    # and gspr[0xFFFF] would be set. Instead, XPR[2]=0x010 is the source.
    assert npu.gspr[0x010] == 0xCAFE
    assert 0xFFFF not in npu.gspr


# ------------- ROADMAP P2 success criterion 3 -----------------

def test_roadmap_p2_3_wrspr_rdspr_lspr_roundtrip_iss_encoding():
    """ROADMAP success criterion 3 (ISS-full encoding path)."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # Step 1: WRSPR(LSPR_SPM_ADDRA=0x900, 0xCAFE)
    proc.get_state().XPR.write(2, 0x900)
    proc.get_state().XPR.write(3, 0xCAFE)
    wrspr_iss(npu, proc, MockInsn(rs1=2, rs2=3), xs1=0, xs2=0)
    # Step 2: RDSPR(LSPR_SPM_ADDRA) -> rd
    proc.get_state().XPR.write(2, 0x900)
    ret = rdspr_iss(npu, proc, MockInsn(rs1=2, rd=5), xs1=0, xs2=0)
    # Step 3: assert XPR[rd] == 0xCAFE
    assert ret == 0xCAFE
    assert proc.get_state().XPR[5] == 0xCAFE


def test_roadmap_p2_3_wrspr_rdspr_gem5_encoding():
    """ROADMAP success criterion 3 (gem5-simplified encoding path)."""
    from tests.gtx._mocks import MockInsn
    npu = _fake_npu()
    proc = _FakeProc()
    # gem5: insn.rs1 must be 0 (xs1=xs2=1 marker but addr comes from XPR[0]=0)
    # so the addressed SPR is GSPR 0x000. Test that path explicitly.
    proc.get_state().XPR.write(4, 0xBABE)   # rs2 holds the value
    wrspr_gem5(npu, proc, MockInsn(rs1=0, rs2=4), xs1=1, xs2=1)
    assert npu.gspr[0x000] == 0xBABE
    ret = rdspr_gem5(npu, proc, MockInsn(rs1=0), xs1=1, xs2=0)
    assert ret == 0xBABE
