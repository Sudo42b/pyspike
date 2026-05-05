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
"""DISP-03 unit tests for dispatch_4mode + dispatch_iss_opcode.

Tests the 4-mode warp router (Mode 1/2/3/4) plus the DMA-only stub for
dispatch_iss_opcode (P3: handles funct7 0x43 / 0x45 / 0x53 as NOPs).

Self-contained `_RISCV_AVAILABLE` detection so the planner's `--noconftest`
acceptance command (`pytest ... --noconftest -o "addopts="`) selects
correctly without the conftest fixture.
"""
from typing import List, Tuple
import pytest

try:
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="dispatch_4mode tests use GtxNpu; require _riscv.so; see Plan 04",
)


def _make_npu():
    from riscv.gtx.npu import GtxNpu
    return GtxNpu()


# ----------------------------------------------------------------------------
# Public surface preservation -- re-export through dispatch.py
# ----------------------------------------------------------------------------
def test_dispatch_4mode_reexport_via_dispatch_module():
    """`from riscv.gtx.dispatch import dispatch_4mode` still resolves to the
    sibling module's symbol after the Plan 04 split."""
    from riscv.gtx.dispatch import dispatch_4mode as via_dispatch
    from riscv.gtx.dispatch_4mode import dispatch_4mode as via_dispatch_4mode
    assert via_dispatch is via_dispatch_4mode


def test_dispatch_iss_opcode_reexport_via_dispatch_module():
    """Same re-export check for dispatch_iss_opcode."""
    from riscv.gtx.dispatch import dispatch_iss_opcode as via_dispatch
    from riscv.gtx.dispatch_4mode import dispatch_iss_opcode as via_module
    assert via_dispatch is via_module


# ----------------------------------------------------------------------------
# Mode 1 / 2 / 4 broadcast counts (Mode 3 verified separately -- DMA path)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("loop_state,expected_count", [
    ((False, False, False), 64),  # Mode 1: broadcast 4*16
    ((True,  False, False), 16),  # Mode 2: broadcast 16 in tmu_id
    ((True,  True,  False), 1),   # Mode 4 (P+T, !S): single (tmu_id, curr_id)
])
def test_dispatch_4mode_routing_count(loop_state, expected_count, monkeypatch):
    """Mode 1/2/4 broadcast counts."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_VECTOR
    npu = _make_npu()
    is_ploop, is_tloop, is_sloop = loop_state
    npu.warp.is_ploop = is_ploop
    npu.warp.is_tloop = is_tloop
    npu.warp.is_sloop = is_sloop
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 5

    seen: List[Tuple[int, int]] = []
    monkeypatch.setattr(
        d4, "dispatch_iss_opcode",
        lambda npu, n, s, opc, o1, o2, o3: seen.append((n, s)) or 0,
    )

    d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR, op1=0, op2=0, op3=0)

    if expected_count == 64:
        assert sorted(seen) == [(n, s) for n in range(4) for s in range(16)]
    elif expected_count == 16:
        assert sorted(seen) == [(1, s) for s in range(16)]
    elif expected_count == 1:
        assert seen == [(1, 5)]


# ----------------------------------------------------------------------------
# Mode 3 (P+S) -- DMA path
# ----------------------------------------------------------------------------
def test_dispatch_4mode_mode3_calls_exec_dma_2d_load_when_sub_op_zero(monkeypatch):
    """Mode 3 sub_op=0 -> is_load=True regardless of opcode.

    Also asserts width = op3 & 0xFFFF, height = (op3 >> 16) & 0xFFFF and
    that l2_addr / l1_addr / nest_id wiring lines up with op1 / op2 / tmu_id.
    """
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_VECTOR
    calls = []
    monkeypatch.setattr(
        d4.dma_engine, "exec_dma_2d",
        lambda mem, **kw: calls.append(kw) or 0,
    )
    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 2

    d4.dispatch_4mode(
        npu, opcode=GTX_OP_VECTOR,
        op1=0x100, op2=0x200,
        op3=(0x10 << 16) | 0x40,  # height=0x10, width=0x40
        sub_op=0,
    )
    assert len(calls) == 1
    assert calls[0]["nest_id"] == 2
    assert calls[0]["l2_addr"] == 0x100
    assert calls[0]["l1_addr"] == 0x200
    assert calls[0]["width"] == 0x40
    assert calls[0]["height"] == 0x10
    assert calls[0]["is_load"] is True


def test_dispatch_4mode_mode3_or_rule_opcode_dma(monkeypatch):
    """Mode 3 opcode=GTX_OP_DMA -> is_load=True even when sub_op != 0
    (the OR-rule, Pitfall 8). Also verifies width / height extraction."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_DMA
    calls = []
    monkeypatch.setattr(
        d4.dma_engine, "exec_dma_2d",
        lambda mem, **kw: calls.append(kw) or 0,
    )
    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True

    # op3 = 0x10001 -> width=0x0001, height=0x0001
    d4.dispatch_4mode(npu, opcode=GTX_OP_DMA,
                      op1=0, op2=0, op3=0x10001, sub_op=1)
    assert len(calls) == 1
    assert calls[0]["is_load"] is True
    assert calls[0]["width"] == 1
    assert calls[0]["height"] == 1


def test_dispatch_4mode_mode3_store_when_sub_op_nonzero_non_dma(monkeypatch):
    """Mode 3 sub_op != 0 AND opcode != GTX_OP_DMA -> is_load=False."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_VECTOR
    calls = []
    monkeypatch.setattr(
        d4.dma_engine, "exec_dma_2d",
        lambda mem, **kw: calls.append(kw) or 0,
    )
    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True

    d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR,
                      op1=0, op2=0, op3=0x10001, sub_op=1)
    assert len(calls) == 1
    assert calls[0]["is_load"] is False


def test_dispatch_4mode_mode3_does_not_call_iss_opcode(monkeypatch):
    """Mode 3 routes EXCLUSIVELY through dma_engine.exec_dma_2d, NOT through
    dispatch_iss_opcode."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_DMA
    iss_calls = []
    monkeypatch.setattr(
        d4, "dispatch_iss_opcode",
        lambda *a, **k: iss_calls.append(a) or 0,
    )
    monkeypatch.setattr(
        d4.dma_engine, "exec_dma_2d", lambda mem, **kw: 0,
    )
    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    d4.dispatch_4mode(npu, opcode=GTX_OP_DMA,
                      op1=0, op2=0, op3=0x10001)
    assert iss_calls == []


# ----------------------------------------------------------------------------
# Mode 2 explicit (uses tmu_id, not nest 0)
# ----------------------------------------------------------------------------
def test_dispatch_4mode_mode2_uses_tmu_id_not_zero(monkeypatch):
    """Mode 2 broadcasts within tmu_id, NOT nest 0."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import GTX_OP_VECTOR
    seen: List[Tuple[int, int]] = []
    monkeypatch.setattr(
        d4, "dispatch_iss_opcode",
        lambda npu, n, s, opc, o1, o2, o3: seen.append((n, s)) or 0,
    )
    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.tmu_id = 3
    d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR, op1=0, op2=0, op3=0)
    assert all(n == 3 for n, _ in seen)
    assert sorted(s for _, s in seen) == list(range(16))


# ----------------------------------------------------------------------------
# dispatch_iss_opcode stub coverage
# ----------------------------------------------------------------------------
def test_dispatch_iss_opcode_oob_nest_returns_zero():
    """Out-of-range nest_id / spu_id silently NOPs (returns 0)."""
    from riscv.gtx import dispatch_4mode as d4
    npu = _make_npu()
    assert d4.dispatch_iss_opcode(npu, 99, 0, 0x40, 0, 0, 0) == 0
    assert d4.dispatch_iss_opcode(npu, -1, 0, 0x40, 0, 0, 0) == 0
    assert d4.dispatch_iss_opcode(npu, 0, 99, 0x40, 0, 0, 0) == 0
    assert d4.dispatch_iss_opcode(npu, 0, -1, 0x40, 0, 0, 0) == 0


def test_dispatch_iss_opcode_dma_funct7_returns_zero():
    """P3 stub: load_svr_l1 (0x43), store_svr_l1 (0x45), credit_st_chk (0x53)
    all NOP and return 0. Plan 05 will wire credit_st_chk's flush trigger."""
    from riscv.gtx import dispatch_4mode as d4
    from riscv.gtx.encoding import (
        GTX_ISS_F7_DMA_LD_SVR_L1, GTX_ISS_F7_DMA_ST_SVR_L1,
        GTX_ISS_F7_CREDIT_ST_CHK,
    )
    npu = _make_npu()
    assert d4.dispatch_iss_opcode(
        npu, 0, 0, GTX_ISS_F7_DMA_LD_SVR_L1, 0, 0, 0
    ) == 0
    assert d4.dispatch_iss_opcode(
        npu, 0, 0, GTX_ISS_F7_DMA_ST_SVR_L1, 0, 0, 0
    ) == 0
    assert d4.dispatch_iss_opcode(
        npu, 0, 0, GTX_ISS_F7_CREDIT_ST_CHK, 0, 0, 0
    ) == 0


def test_dispatch_iss_opcode_unknown_funct7_returns_zero():
    """Every unknown funct7 NOPs in P3 -- never raises."""
    from riscv.gtx import dispatch_4mode as d4
    npu = _make_npu()
    assert d4.dispatch_iss_opcode(npu, 0, 0, 0xFF, 0, 0, 0) == 0
    assert d4.dispatch_iss_opcode(npu, 0, 0, 0x00, 0, 0, 0) == 0
