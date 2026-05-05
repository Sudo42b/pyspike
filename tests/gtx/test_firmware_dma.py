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
"""DMA-02 unit tests: firmware_dma rs1/rs2/rs3 decode + branch routing tests
plus tpose/fill LSPR_SPM_ADDR* assertion tests.

Phase 3 plan 02 Task 2a populates this from Wave 0 placeholder. Module-level
_RISCV_AVAILABLE detection so --noconftest acceptance command still selects
correctly.
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
    reason="firmware_dma tests require _riscv.so (built by setup.py); see Plan 02",
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


# ============================================================================
# Task 2a: firmware_dma routing tests
# ============================================================================

def test_firmware_dma_load_sloop_calls_sloop_load(monkeypatch):
    """is_sloop branch -> dma_engine.firmware_dma_sloop_load receives mem + decoded args."""
    from riscv.gtx import dma_engine
    from riscv.gtx.encoding import GSPR_GTX_OPERAND3

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 1

    captured = {}

    def fake_sloop_load(mem, **kw):
        captured["mem"] = mem
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "firmware_dma_sloop_load", fake_sloop_load)

    proc = _make_proc()
    # rs1: addr_hi at bits [63:27], addr_lo at bits [26:0]
    # Pick distinguishable values: addr_hi = 0x100, addr_lo = 0x200
    rs1_val = (0x100 << 27) | 0x200
    proc.get_state().XPR.write(1, rs1_val)
    # rs2: height=2 [63:48], length=256 [47:32], rs2_low=128 [31:0]
    rs2_val = (2 << 48) | (256 << 32) | 128
    proc.get_state().XPR.write(2, rs2_val)
    # rs3 from gspr[GSPR_GTX_OPERAND3]
    npu.gspr[GSPR_GTX_OPERAND3] = 64

    # synthesize funct3=0 for LOAD: xd=0, xs1=0, xs2=0
    insn = _make_insn(funct=0x40, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    # mem must be npu.mem (not npu)
    assert captured["mem"] is npu.mem
    assert captured["nest"] == 1   # tmu_id
    assert captured["addr_hi"] == 0x100
    assert captured["addr_lo"] == 0x200
    assert captured["length"] == 256
    assert captured["height"] == 2
    assert captured["rd_stride"] == 128   # rs2_low (LOAD: rs2_low->rd)
    assert captured["wr_stride"] == 64    # rs3_low (LOAD: rs3_low->wr)


def test_firmware_dma_store_sloop_pushes_deferred(monkeypatch):
    """is_sloop STORE branch -> dma_engine.firmware_dma_sloop_store(npu, ...) push."""
    from riscv.gtx import dma_engine
    from riscv.gtx.encoding import GSPR_GTX_OPERAND3

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0

    captured = {}

    def fake_sloop_store(npu_arg, **kw):
        captured["npu"] = npu_arg
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "firmware_dma_sloop_store", fake_sloop_store)

    proc = _make_proc()
    rs1_val = (0x300 << 27) | 0x400
    proc.get_state().XPR.write(1, rs1_val)
    rs2_val = (3 << 48) | (1024 << 32) | 256
    proc.get_state().XPR.write(2, rs2_val)
    npu.gspr[GSPR_GTX_OPERAND3] = 512

    # synthesize funct3=1 for STORE: xd=0, xs1=0, xs2=1
    insn = _make_insn(funct=0x40, xd=0, xs1=0, xs2=1, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    # store branch: first arg is npu (not mem) so push to deferred queue works
    assert captured["npu"] is npu
    assert captured["addr_hi"] == 0x300
    assert captured["addr_lo"] == 0x400
    assert captured["length"] == 1024
    assert captured["height"] == 3
    # STORE: wr_stride = rs2_low, rd_stride = rs3_low
    assert captured["wr_stride"] == 256
    assert captured["rd_stride"] == 512


def test_firmware_dma_copy_tloop_uses_high_32_bit_dst(monkeypatch):
    """Pitfall 1 e2e: COPY (funct3=2) decodes addr_hi as (rs1>>32) NOT (rs1>>27)&0x1F.."""
    from riscv.gtx import dma_engine

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = 0
    npu.warp.curr_id = 5

    captured = {}

    def fake_tloop_copy(mem, **kw):
        captured["mem"] = mem
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "firmware_dma_tloop_copy", fake_tloop_copy)

    proc = _make_proc()
    # Pick distinguishable hi/lo values that differ in bits 27-31.
    # If decoder used (rs1>>27)&0x1FFFFFFFFF: addr_hi would have bits 27-31 leak.
    # Set rs1 = (0xCAFE << 32) | 0x1234
    # COPY decode: addr_hi = rs1 >> 32 = 0xCAFE (32-bit)
    #              addr_lo = rs1 & 0x7FFFFFF = 0x1234
    rs1_val = (0xCAFE << 32) | 0x1234
    proc.get_state().XPR.write(1, rs1_val)
    # rs2: height=4, length=64, stride=64
    rs2_val = (4 << 48) | (64 << 32) | 64
    proc.get_state().XPR.write(2, rs2_val)

    # synthesize funct3=2 for COPY: xd=0, xs1=1, xs2=0
    insn = _make_insn(funct=0x40, xd=0, xs1=1, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert captured["mem"] is npu.mem
    # COPY: dst_addr is addr_hi (high 32 bits)
    assert captured["dst_addr"] == 0xCAFE
    # COPY: src_addr is addr_lo (low 27 bits)
    assert captured["src_addr"] == 0x1234
    assert captured["spu"] == 5
    assert captured["nest"] == 0
    assert captured["length"] == 64
    assert captured["height"] == 4


def test_firmware_dma_no_loop_returns_zero():
    """Outside is_sloop and is_tloop: firmware_dma is a NOP (returns 0)."""
    npu = _make_npu()
    # No loop flags set
    assert npu.warp.is_sloop is False
    assert npu.warp.is_tloop is False

    proc = _make_proc()
    proc.get_state().XPR.write(1, 0)
    proc.get_state().XPR.write(2, 0)

    # funct3=0 (LOAD) but no loop active
    insn = _make_insn(funct=0x40, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    # Queue must NOT be touched
    assert npu.deferred_ddr_stores == []


def test_firmware_dma_xs1_zero_uses_proc_xpr(monkeypatch):
    """Pitfall 3 (CORE-04): handler reads XPR[insn.rs1] directly, not the xs1 arg.

    Simulate Spike marshalling -1 for xs1/xs2 when the encoded xs1/xs2 flags
    are 0 -- the handler must ignore this and use proc.get_state().XPR[insn.rs1].
    """
    from riscv.gtx import dma_engine
    from riscv.gtx.encoding import GSPR_GTX_OPERAND3

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0

    captured = {}

    def fake_sloop_load(mem, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "firmware_dma_sloop_load", fake_sloop_load)

    proc = _make_proc()
    rs1_val = (0x111 << 27) | 0x222
    proc.get_state().XPR.write(1, rs1_val)
    proc.get_state().XPR.write(2, (1 << 48) | (32 << 32))   # height=1, length=32
    npu.gspr[GSPR_GTX_OPERAND3] = 0

    # funct3=0 for LOAD; pass xs1=0xFFFFFFFFFFFFFFFF to simulate -1 marshalling
    insn = _make_insn(funct=0x40, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn,
                     xs1=0xFFFFFFFFFFFFFFFF, xs2=0xFFFFFFFFFFFFFFFF)
    assert rc == 0
    # If handler had used xs1 arg: addr_lo would be (-1 & 0x7FFFFFF) = 0x7FFFFFF
    # Instead we expect 0x222 (from XPR[1])
    assert captured["addr_lo"] == 0x222
    assert captured["addr_hi"] == 0x111


def test_firmware_dma_length_zero_means_65536_e2e(monkeypatch):
    """Pitfall 2: length_raw=0 in encoding -> resolved length=0x10000 (65536)."""
    from riscv.gtx import dma_engine

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0

    captured = {}

    def fake_sloop_load(mem, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "firmware_dma_sloop_load", fake_sloop_load)

    proc = _make_proc()
    proc.get_state().XPR.write(1, 0)
    # height_raw=0 (HW conv: 1), length_raw=0 (HW conv: 65536)
    proc.get_state().XPR.write(2, 0)

    insn = _make_insn(funct=0x40, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert captured["length"] == 0x10000   # 65536, NOT 0
    assert captured["height"] == 1         # NOT 0


def test_firmware_dma_funct7_0x41_load_svr_dispatch(monkeypatch):
    """funct7=0x41 funct3=0 -> exec_load_svr (NOT firmware_dma)."""
    from riscv.gtx import dma_engine

    npu = _make_npu()
    npu.warp.is_ploop = True
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 7

    captured = {}

    def fake_load_svr(mem, **kw):
        captured["mem"] = mem
        captured.update(kw)

    monkeypatch.setattr(dma_engine, "exec_load_svr", fake_load_svr)

    proc = _make_proc()
    proc.get_state().XPR.write(1, 0xABCDEF12)   # rs1 -> l1_addr & 0x7FFFFFF
    proc.get_state().XPR.write(2, 0xFFFFFFFF)   # rs2 -> l0_reg & 0x1F

    # synthesize funct3=0: xd=0, xs1=0, xs2=0
    insn = _make_insn(funct=0x41, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert captured["mem"] is npu.mem
    assert captured["nest_id"] == 1
    assert captured["spu_id"] == 7
    assert captured["l1_addr"] == 0xABCDEF12 & 0x7FFFFFF
    assert captured["l0_reg"] == 0xFFFFFFFF & 0x1F


def test_tpose_reads_lspr_spm_addrr_at_0x903(monkeypatch):
    """tpose handler reads addr_a from LSPR_SPM_ADDRA (0x900) and addr_r from
    LSPR_SPM_ADDRR (0x903) -- NOT 0x901 (LSPR_SPM_ADDRB).

    Critical regression guard: an earlier draft used 0x901; orchestrator-verified
    gtx_params.h:67 confirms 0x903 is the result address.
    """
    from riscv.gtx import dma_engine
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRR

    # Sanity: the constants we test against ARE the gtx_params.h authoritative values
    assert LSPR_SPM_ADDRA == 0x900
    assert LSPR_SPM_ADDRR == 0x903

    npu = _make_npu()
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0xCAFEBABE
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0xDEADBEEF

    captured = {}

    def fake_transpose(mem, **kw):
        captured["mem"] = mem
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "exec_transpose", fake_transpose)

    proc = _make_proc()
    proc.get_state().XPR.write(1, 4)   # rs1: rows = 4
    proc.get_state().XPR.write(2, 8)   # rs2: cols = 8

    insn = _make_insn(funct=0x38, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert captured["addr_a"] == 0xCAFEBABE
    # CRITICAL: addr_r MUST be 0xDEADBEEF (LSPR_SPM_ADDRR=0x903),
    # NOT 0 (which would be the case if 0x901 / LSPR_SPM_ADDRB was used by mistake).
    assert captured["addr_r"] == 0xDEADBEEF
    assert captured["rows"] == 4
    assert captured["cols"] == 8


def test_fill_reads_lspr_spm_addrr_at_0x903(monkeypatch):
    """fill handler reads addr_r from LSPR_SPM_ADDRR (0x903) -- NOT 0x901."""
    from riscv.gtx import dma_engine
    from riscv.gtx.encoding import LSPR_SPM_ADDRR

    assert LSPR_SPM_ADDRR == 0x903

    npu = _make_npu()
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0xCAFE0000

    captured = {}

    def fake_fill(mem, **kw):
        captured["mem"] = mem
        captured.update(kw)
        return 0

    monkeypatch.setattr(dma_engine, "exec_fill", fake_fill)

    proc = _make_proc()
    # rs1: length=64 [15:0], fill_val=0xBEEF [31:16]
    proc.get_state().XPR.write(1, (0xBEEF << 16) | 64)

    insn = _make_insn(funct=0x39, xd=0, xs1=0, xs2=0, rs1=1, rs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert captured["mem"] is npu.mem
    assert captured["length"] == 64
    assert captured["fill_val"] == 0xBEEF
    assert captured["addr_r"] == 0xCAFE0000


# ============================================================================
# Task 2b: disasm-only stubs (load_3d/store_3d/mcast_*/copy_mem) + credit_st_chk
# ============================================================================

def test_firmware_dma_funct7_0x41_funct3_4_load_3d_is_nop():
    """funct7=0x41 funct3=4 (load_3d) is a v2 deferral disasm-only stub:
    returns 0 and does NOT touch deferred_ddr_stores."""
    npu = _make_npu()
    proc = _make_proc()
    # synthesize funct3=4: xd=1, xs1=0, xs2=0 -> (1<<2)|(0<<1)|0 = 4
    insn = _make_insn(funct=0x41, xd=1, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0
    assert npu.deferred_ddr_stores == []


def test_firmware_dma_funct7_0x41_funct3_5_store_3d_is_nop():
    """funct7=0x41 funct3=5 (store_3d) v2 deferral stub returns 0."""
    npu = _make_npu()
    proc = _make_proc()
    # synthesize funct3=5: xd=1, xs1=0, xs2=1 -> 5
    insn = _make_insn(funct=0x41, xd=1, xs1=0, xs2=1, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0


def test_firmware_dma_mcast_funct7_0x42_is_nop():
    """funct7=0x42 (mcast_s2l) v2 deferral stub returns 0."""
    npu = _make_npu()
    proc = _make_proc()
    insn = _make_insn(funct=0x42, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0


def test_firmware_dma_mcast_funct7_0x44_funct3_branches_are_nop():
    """funct7=0x44 funct3=0/2/3 (mcast_g2s/mcast_s2s/copy_mem) v2 deferral stubs."""
    npu = _make_npu()
    proc = _make_proc()
    for f3 in (0, 2, 3):
        xd = (f3 >> 2) & 1
        xs1_bit = (f3 >> 1) & 1
        xs2_bit = f3 & 1
        insn = _make_insn(funct=0x44, xd=xd, xs1=xs1_bit, xs2=xs2_bit,
                          rs1=1, rs2=2)
        rc = npu.custom0(proc, insn, 0, 0)
        assert rc == 0, f"funct3={f3} stub did not return 0"


def test_credit_st_chk_p3_stub_returns_zero():
    """funct7=0x53 (credit_st_chk) stub returns 0 in Plan 02; Plan 05 will
    replace body with `if npu.warp.is_sloop: npu.flush_deferred_ddr_stores()`."""
    npu = _make_npu()
    proc = _make_proc()
    insn = _make_insn(funct=0x53, xd=0, xs1=0, xs2=0, rs1=0, rs2=0)
    rc = npu.custom0(proc, insn, 0, 0)
    assert rc == 0


def test_disasm_includes_all_dma_mnemonics():
    """All 16 DMA mnemonics (9 active + 5 v2 stubs + 1 alias load_svr_l1 +
    1 alias store_svr_l1 + credit_st_chk = total mix) registered in disasm.

    Best-effort: walks the registry directly to confirm @handler decorators
    fired for every expected mnemonic. Bypasses any disasm_insn_t introspection
    quirks under different bindings.
    """
    from riscv.gtx import _registry as reg
    seen_mnemonics = {
        e["mnemonic"] for e in reg._HANDLER_REGISTRY
        if e.get("mnemonic")
    }
    required = {
        # active firmware_dma + load/store_svr + tpose/fill (9)
        'load', 'store', 'copy',
        'load_svr', 'store_svr',
        'load_svr_l1', 'store_svr_l1',
        'tpose', 'fill',
        # v2 deferral stubs (5)
        'load_3d', 'store_3d',
        'mcast_s2l', 'mcast_g2s', 'mcast_s2s', 'copy_mem',
        # credit_st_chk (1)
        'credit_st_chk',
    }
    missing = required - seen_mnemonics
    assert not missing, f"missing DMA mnemonics from @handler registry: {missing}"
