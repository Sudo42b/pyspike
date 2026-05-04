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
"""Tests for DISP-01 -- custom0 funct7 dispatch dict + collision heuristic.

All tests require _riscv.so (since the dispatch dict is built by GtxNpu.__init__).
Skipif-gated via module-level detection so the suite works under
`pytest ... --noconftest -o "addopts="`.
"""
import pytest


# Module-level detection -- self-contained so --noconftest still works.
try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="_riscv.so not built -- GtxNpu requires rocc_t base",
)


def _make_npu():
    from riscv.gtx import GtxNpu
    return GtxNpu()


def _make_proc():
    from tests.gtx._mocks import MockProcessor
    return MockProcessor()


def _make_insn(**kwargs):
    from tests.gtx._mocks import MockInsn
    return MockInsn(**kwargs)


# ---------------------------------------------------------------------------
# Dispatch table coverage
# ---------------------------------------------------------------------------

def test_custom0_table_has_p2_handlers():
    """All 10 P2 custom0 funct7 keys present (4 SPR + 6 stubs)."""
    npu = _make_npu()
    keys = set(npu._custom0.keys())
    expected = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x48, 0x49}
    missing = expected - keys
    assert not missing, f"missing custom0 funct7 keys: {[hex(k) for k in missing]}"


def test_custom1_table_has_8_funct3_handlers():
    """All 8 custom1 funct3 keys present (warp control)."""
    npu = _make_npu()
    keys = set(npu._custom1.keys())
    expected = {0, 1, 2, 3, 4, 5, 6, 7}
    missing = expected - keys
    assert not missing, f"missing custom1 funct3 keys: {missing}"


# ---------------------------------------------------------------------------
# D-02 collision heuristic (funct7=0x00)
# ---------------------------------------------------------------------------

def test_custom0_funct7_collision_rs1_zero_writes_spr():
    """DISP-01 / D-02: funct7=0x00, insn.rs1=0 -> wrspr_gem5 writes addr=XPR[0]=0."""
    npu = _make_npu()
    proc = _make_proc()
    proc.get_state().XPR.write(4, 0xCAFE)
    insn = _make_insn(funct=0x00, rs1=0, rs2=4)
    ret = npu.custom0(proc, insn, xs1=0, xs2=0)
    assert ret == 0
    # gem5 WRSPR with rs1==0 writes to GSPR address 0 (research §649 verbatim port).
    # reset() pre-seeds gspr[0x000]=0; the call overwrites it with 0xCAFE.
    assert npu.gspr[0x000] == 0xCAFE


def test_custom0_funct7_collision_rs1_nonzero_returns_zero():
    """DISP-01 / D-02: funct7=0x00, insn.rs1!=0 -> P4 MM stub (return 0, no SPR write)."""
    npu = _make_npu()
    proc = _make_proc()
    proc.get_state().XPR.write(3, 0x900)
    proc.get_state().XPR.write(4, 0xCAFE)
    snapshot_gspr = dict(npu.gspr)
    snapshot_lspr00 = dict(npu.lspr[0][0])
    insn = _make_insn(funct=0x00, rs1=3, rs2=4)
    ret = npu.custom0(proc, insn, xs1=0, xs2=0)
    assert ret == 0
    # P4 MM stub: NO SPR mutation
    assert npu.gspr == snapshot_gspr
    assert npu.lspr[0][0] == snapshot_lspr00


# ---------------------------------------------------------------------------
# ISS-full encodings (funct7=0x48 RDSPR, 0x49 WRSPR)
# ---------------------------------------------------------------------------

def test_custom0_iss_wrspr_writes_spr():
    """ISS-full WRSPR (funct7=0x49) -- direct register read, full encoding."""
    npu = _make_npu()
    proc = _make_proc()
    proc.get_state().XPR.write(2, 0x900)   # addr (LSPR_SPM_ADDRA)
    proc.get_state().XPR.write(3, 0xBEEF)  # value
    insn = _make_insn(funct=0x49, rs1=2, rs2=3)
    ret = npu.custom0(proc, insn, xs1=0, xs2=0)
    assert ret == 0
    # No loop state -> fallback (NEST 0, SPU 0).
    assert npu.lspr[0][0][0x900] == 0xBEEF


def test_custom0_iss_rdspr_writes_to_rd():
    """ISS-full RDSPR (funct7=0x48) -- force-write to rd."""
    npu = _make_npu()
    proc = _make_proc()
    npu.gspr[0x010] = 0xFEED
    proc.get_state().XPR.write(2, 0x010)
    insn = _make_insn(funct=0x48, rs1=2, rd=5)
    ret = npu.custom0(proc, insn, xs1=0, xs2=0)
    assert ret == 0xFEED
    assert proc.get_state().XPR[5] == 0xFEED


# ---------------------------------------------------------------------------
# Unmapped funct7
# ---------------------------------------------------------------------------

def test_custom0_unmapped_funct7_returns_zero():
    """Unmapped funct7 -> silent NOP (return 0). P5/P6 may upgrade to illegal_instruction."""
    npu = _make_npu()
    proc = _make_proc()
    insn = _make_insn(funct=0x7C, rs1=0, rs2=0)   # 0x7C not in P2 set
    ret = npu.custom0(proc, insn, xs1=0, xs2=0)
    assert ret == 0


# ---------------------------------------------------------------------------
# custom1 funct3 dispatch (full sweep including WJOIN SystemExit)
# ---------------------------------------------------------------------------

def test_custom1_funct3_dispatch_routes_correctly():
    """custom1 reconstructs funct3 = (xd<<2)|(xs1<<1)|xs2 and dispatches."""
    npu = _make_npu()
    proc = _make_proc()
    proc.get_state().XPR.write(2, 1)   # nest_id 1
    # funct3=0b110 = (xd=1, xs1=1, xs2=0) -> start_p
    insn = _make_insn(xd=1, xs1=1, xs2=0, rs1=2, rs2=0)
    ret = npu.custom1(proc, insn, xs1=0, xs2=0)
    assert ret == 0
    assert npu.warp.is_ploop is True
    assert npu.warp.tmu_id == 1


def test_custom1_unmapped_funct3_returns_zero():
    """All 8 funct3 keys are mapped in P2; this test verifies all 8 dispatch
    via a parametrize-style sweep. WJOIN raises SystemExit -- handle separately."""
    npu = _make_npu()
    proc = _make_proc()
    # Sanity-check that calling with each funct3 doesn't raise (except WJOIN).
    for f3 in range(8):
        xd = (f3 >> 2) & 1
        xs1_bit = (f3 >> 1) & 1
        xs2_bit = f3 & 1
        insn = _make_insn(xd=xd, xs1=xs1_bit, xs2=xs2_bit)
        if f3 == 0b101:   # WJOIN -- raises SystemExit when GTX_NO_EXIT unset
            import os
            os.environ.pop('GTX_NO_EXIT', None)
            with pytest.raises(SystemExit):
                npu.custom1(proc, insn, xs1=0, xs2=0)
        else:
            ret = npu.custom1(proc, insn, xs1=0, xs2=0)
            assert ret == 0
