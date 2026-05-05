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
"""DMA-03 unit tests: deferred-store dual-trigger flush (Plan 05).

Covers the two flush-trigger paths from 03-RESEARCH "Deferred-Store Flush
Trigger" lock-in:
  1. end_p (custom1 funct3=0b111) when !wsplit_seen  -- simple firmware path.
  2. credit_st_chk (custom0 funct7=0x53) when is_sloop  -- WSPLIT plan-style
     firmware path. Reachable via TWO entry points: ops/dma.py _credit_st_chk
     handler AND dispatch_4mode.dispatch_iss_opcode (RESEARCH "3 call sites"
     lock-in).

Also verifies that wsplit_seen suppresses end_p flushing (so plan-style
firmware doesn't double-flush) and that wsplit_seen survives reset()
(Pitfall 7).

Module-level _RISCV_AVAILABLE detection so --noconftest acceptance command
still selects correctly.
"""
import numpy as np
import pytest

# Module-level detection -- self-contained for --noconftest.
try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="deferred_store tests use GtxNpu; require _riscv.so; see Plan 05",
)


# ============================================================================
# Helpers / fixtures
# ============================================================================
def _make_npu():
    from riscv.gtx import GtxNpu
    return GtxNpu()


def _make_proc():
    from tests.gtx._mocks import MockProcessor
    return MockProcessor()


def _make_insn(**kwargs):
    from tests.gtx._mocks import MockInsn
    return MockInsn(**kwargs)


@pytest.fixture
def npu_with_pattern():
    """NPU pre-populated with a known L2 byte pattern in NEST 0 [100..200)."""
    npu = _make_npu()
    npu.mem.l2_byte(0)[100:200] = np.arange(100, dtype=np.uint8)
    return npu


def _push_deferred_store(npu, *, nest=0, l2_off=100, ddr_off=0x1000,
                          length=100, height=1, l2_stride=100, ddr_stride=100):
    """Synthesize a single S-loop store push via dma_engine helper."""
    from riscv.gtx.dma_engine import firmware_dma_sloop_store
    firmware_dma_sloop_store(
        npu, nest=nest,
        addr_hi=ddr_off, addr_lo=l2_off,
        length=length, height=height,
        rd_stride=l2_stride, wr_stride=ddr_stride,
    )


# ============================================================================
# Queue push + flush diff (DMA-03 base)
# ============================================================================
def test_deferred_store_queue_push_shape(npu_with_pattern):
    """firmware_dma_sloop_store pushes one DeferredDdrStore with locked fields."""
    npu = npu_with_pattern
    assert npu.deferred_ddr_stores == []
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x1000, length=100)
    assert len(npu.deferred_ddr_stores) == 1
    req = npu.deferred_ddr_stores[0]
    # Locked field order per Plan 01 D-2 (DeferredDdrStore frozen dataclass).
    assert req.nest == 0
    assert req.l2_off == 100
    assert req.ddr_off == 0x1000
    assert req.length == 100
    assert req.height == 1
    assert req.l2_stride == 100
    assert req.ddr_stride == 100


def test_deferred_store_flush_diff(npu_with_pattern):
    """Pre-flush DDR is zero, post-flush DDR matches L2 source bytes."""
    from riscv.gtx.ddr import ensure_ddr
    npu = npu_with_pattern
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x1000, length=100)

    # Pre-flush snapshot -- DDR untouched (deferred queue is a no-op until flush).
    ensure_ddr(npu.mem, 0x1000 + 100)
    pre_flush = bytes(npu.mem._ddr_bytes[0x1000:0x1000 + 100])
    assert pre_flush == bytes(100)   # all zeros

    npu.flush_deferred_ddr_stores()

    post_flush = bytes(npu.mem._ddr_bytes[0x1000:0x1000 + 100])
    expected = bytes(np.arange(100, dtype=np.uint8))
    assert post_flush == expected
    assert pre_flush != post_flush  # explicit divergence assertion
    assert npu.deferred_ddr_stores == []


# ============================================================================
# end_p flush trigger (custom1 funct3=0b111)
# ============================================================================
def _invoke_endp(npu, proc):
    """Invoke custom1 warp_end_p (funct3=0b111) through the full handler path.

    funct3 = (xd<<2)|(xs1<<1)|xs2 = (1<<2)|(1<<1)|1 = 0b111 = 7
    """
    insn = _make_insn(funct=0, xd=1, xs1=1, xs2=1, rs1=0, rs2=0)
    return npu.custom1(proc, insn, 0, 0)


def test_endp_flushes_when_no_wsplit_seen(npu_with_pattern):
    """end_p path: !wsplit_seen -> flush. ROADMAP P3 success #4 path."""
    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0
    npu.warp.wsplit_seen = False
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x1000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    proc = _make_proc()
    rc = _invoke_endp(npu, proc)
    assert rc == 0
    # Flushed: queue empty + DDR holds the bytes
    assert npu.deferred_ddr_stores == []
    assert (
        bytes(npu.mem._ddr_bytes[0x1000:0x1000 + 100])
        == bytes(np.arange(100, dtype=np.uint8))
    )
    # is_ploop cleared as before
    assert npu.warp.is_ploop is False


def test_endp_does_not_flush_when_wsplit_seen(npu_with_pattern):
    """end_p path: wsplit_seen=True -> NO flush (plan-style firmware uses
    credit_st_chk to flush mid-execution instead).
    """
    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0
    npu.warp.wsplit_seen = True
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x2000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    proc = _make_proc()
    rc = _invoke_endp(npu, proc)
    assert rc == 0
    # NOT flushed -- queue still has the entry
    assert len(npu.deferred_ddr_stores) == 1
    # is_ploop still cleared (matches C++ end_p semantics)
    assert npu.warp.is_ploop is False


# ============================================================================
# WSPLIT handlers set wsplit_seen
# ============================================================================
def test_wsplit_custom1_sets_wsplit_seen():
    """custom1 funct3=0b100 (warp_split) sets npu.warp.wsplit_seen=True."""
    npu = _make_npu()
    assert npu.warp.wsplit_seen is False
    proc = _make_proc()
    # synthesize funct3=0b100 = 4: xd=1, xs1=0, xs2=0
    insn = _make_insn(funct=0, xd=1, xs1=0, xs2=0, rs1=0, rs2=0)
    rc = npu.custom1(proc, insn, 0, 0)
    assert rc == 0
    assert npu.warp.wsplit_seen is True


def test_wsplit_custom0_sets_wsplit_seen():
    """custom0 funct7=0x02 (wsplit_c0 firmware variant) sets wsplit_seen=True."""
    npu = _make_npu()
    assert npu.warp.wsplit_seen is False
    proc = _make_proc()
    insn = _make_insn(funct=0x02, xd=0, xs1=0, xs2=0, rs1=0, rs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert npu.warp.wsplit_seen is True


# ============================================================================
# credit_st_chk via custom0 entry path (ops/dma.py _credit_st_chk)
# ============================================================================
def test_credit_st_chk_flushes_when_is_sloop(npu_with_pattern):
    """custom0 funct7=0x53 with is_sloop=True -> flush."""
    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x2000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    proc = _make_proc()
    insn = _make_insn(funct=0x53, xd=0, xs1=0, xs2=0, rs1=0, rs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert npu.deferred_ddr_stores == []
    assert (
        bytes(npu.mem._ddr_bytes[0x2000:0x2000 + 100])
        == bytes(np.arange(100, dtype=np.uint8))
    )


def test_credit_st_chk_no_flush_when_not_sloop(npu_with_pattern):
    """custom0 funct7=0x53 with is_sloop=False -> NO flush. Queue retained."""
    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = False  # critical: NOT in sloop
    npu.warp.tmu_id = 0
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x3000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    proc = _make_proc()
    insn = _make_insn(funct=0x53, xd=0, xs1=0, xs2=0, rs1=0, rs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    # NOT flushed -- queue still has the entry
    assert len(npu.deferred_ddr_stores) == 1


# ============================================================================
# credit_st_chk via dispatch_iss_opcode entry path (dispatch_4mode.py)
# ============================================================================
def test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop(npu_with_pattern):
    """dispatch_iss_opcode funct7=GTX_ISS_F7_CREDIT_ST_CHK with is_sloop -> flush.

    RESEARCH 3 call sites lock-in: BOTH ops/dma.py _credit_st_chk AND
    dispatch_4mode.dispatch_iss_opcode trigger the flush. (The dispatch path
    is reachable via Mode 3+ dispatch_4mode wiring; both paths converge on
    the same flush API.)
    """
    from riscv.gtx.dispatch_4mode import dispatch_iss_opcode
    from riscv.gtx.encoding import GTX_ISS_F7_CREDIT_ST_CHK

    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x4000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    rc = dispatch_iss_opcode(npu, 0, 0, GTX_ISS_F7_CREDIT_ST_CHK, 0, 0, 0)
    assert rc == 0
    assert npu.deferred_ddr_stores == []
    assert (
        bytes(npu.mem._ddr_bytes[0x4000:0x4000 + 100])
        == bytes(np.arange(100, dtype=np.uint8))
    )


def test_dispatch_iss_opcode_credit_st_chk_no_flush_when_not_sloop(npu_with_pattern):
    """dispatch_iss_opcode funct7=GTX_ISS_F7_CREDIT_ST_CHK with !is_sloop -> NO flush."""
    from riscv.gtx.dispatch_4mode import dispatch_iss_opcode
    from riscv.gtx.encoding import GTX_ISS_F7_CREDIT_ST_CHK

    npu = npu_with_pattern
    npu.warp.is_ploop = True
    npu.warp.is_sloop = False
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x4000, length=100)
    assert len(npu.deferred_ddr_stores) == 1

    rc = dispatch_iss_opcode(npu, 0, 0, GTX_ISS_F7_CREDIT_ST_CHK, 0, 0, 0)
    assert rc == 0
    # NOT flushed
    assert len(npu.deferred_ddr_stores) == 1


# ============================================================================
# Pitfall 7 -- reset clears queue but NOT wsplit_seen
# ============================================================================
def test_reset_clears_deferred_queue_but_not_wsplit_seen(npu_with_pattern):
    """Pitfall 7 e2e: reset() clears deferred_ddr_stores but does NOT clear
    wsplit_seen (the sentinel is process-lifetime).
    """
    npu = npu_with_pattern
    _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x5000, length=64)
    npu.warp.wsplit_seen = True
    assert len(npu.deferred_ddr_stores) == 1

    proc = _make_proc()
    npu.reset(proc)

    # Queue cleared
    assert npu.deferred_ddr_stores == []
    # wsplit_seen NOT cleared (Pitfall 7)
    assert npu.warp.wsplit_seen is True
