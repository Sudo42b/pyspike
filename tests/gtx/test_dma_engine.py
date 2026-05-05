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
"""Tests for DMA-01 -- dma_engine helpers + constants + WarpState.wsplit_seen.

Phase 3 plan 01 Task 1 (constants/state) and Task 2 (dma_engine helpers).

Module has NO `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)` --
all helpers in dma_engine.py are spike-independent (CONTEXT D-01) and the
constants live in pure-Python modules. Tests run with --noconftest.
"""
import pytest


def test_gtx_ddr_base_constant():
    from riscv.gtx.params import GTX_DDR_BASE
    assert GTX_DDR_BASE == 0x370000000


def test_gspr_operand_addresses():
    # AUTHORITATIVE values from gtx_params.h:38-41 (orchestrator-verified).
    from riscv.gtx.encoding import (
        GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3, GSPR_GTX_OPCODE)
    assert GSPR_GTX_OPERAND1 == 0x001
    assert GSPR_GTX_OPERAND2 == 0x002
    assert GSPR_GTX_OPERAND3 == 0x003
    assert GSPR_GTX_OPCODE   == 0x004


def test_lspr_spm_addresses():
    # AUTHORITATIVE values from gtx_params.h:64-67.
    from riscv.gtx.encoding import (
        LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR)
    assert LSPR_SPM_ADDRA == 0x900
    assert LSPR_SPM_ADDRB == 0x901
    assert LSPR_SPM_ADDRC == 0x902
    assert LSPR_SPM_ADDRR == 0x903


def test_iss_dma_funct7_constants():
    from riscv.gtx.encoding import (
        GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
        GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
        GTX_ISS_F7_CREDIT_ST_CHK)
    assert (GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
            GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
            GTX_ISS_F7_CREDIT_ST_CHK) == (0x38, 0x39, 0x40, 0x41, 0x53)


def test_warp_wsplit_seen_persists_through_reset():
    from riscv.gtx.warp_state import WarpState
    w = WarpState()
    assert w.wsplit_seen is False
    w.wsplit_seen = True
    w.is_ploop = True
    w.reset()
    assert w.is_ploop is False        # normal field reset
    assert w.wsplit_seen is True      # process-lifetime sentinel persists


def test_wave0_scaffolds_exist():
    import pathlib
    td = pathlib.Path(__file__).parent
    for name in ("test_firmware_dma.py", "test_deferred_store.py",
                 "test_ddr_modes.py", "test_dma_roundtrip.py",
                 "test_dispatch_4mode.py"):
        assert (td / name).exists(), f"missing wave 0 scaffold: {name}"


def test_dma_engine_module_imports():
    from riscv.gtx import dma_engine
    assert hasattr(dma_engine, 'exec_dma_2d')
    assert hasattr(dma_engine, 'decode_firmware_dma_args')
    assert hasattr(dma_engine, 'DeferredDdrStore')


# ============================================================================
# Task 2 tests -- DeferredDdrStore + decode_firmware_dma_args + helpers
# ============================================================================

import dataclasses
import numpy as np


def test_deferred_ddr_store_has_seven_fields_in_order():
    """Pitfall 4: DeferredDdrStore must have exactly 7 fields in spec order."""
    from riscv.gtx.dma_engine import DeferredDdrStore
    fields = dataclasses.fields(DeferredDdrStore)
    names = tuple(f.name for f in fields)
    assert len(fields) == 7
    assert names == ('nest', 'l2_off', 'ddr_off', 'length', 'height',
                     'l2_stride', 'ddr_stride')


def test_deferred_ddr_store_is_frozen():
    from riscv.gtx.dma_engine import DeferredDdrStore
    req = DeferredDdrStore(nest=0, l2_off=0, ddr_off=0, length=64,
                            height=1, l2_stride=64, ddr_stride=64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.nest = 99  # type: ignore


def test_decode_load_basic():
    """LOAD funct3=0 (xs2=0,xs1=0,xd=0): rs1 split as addr_hi[63:27]|addr_lo[26:0]."""
    from riscv.gtx.dma_engine import decode_firmware_dma_args
    rs1 = 0x123456789ABCDEF
    rs2 = (4 << 48) | (256 << 32) | 0x100  # height=4, length=256, stride=0x100
    rs3 = 0x200
    d = decode_firmware_dma_args(rs1, rs2, rs3, xd=0, xs1=0, xs2=0)
    assert d['is_store'] is False
    assert d['is_copy'] is False
    assert d['funct3'] == 0
    assert d['addr_hi'] == ((rs1 >> 27) & 0x1FFFFFFFFF)
    assert d['addr_lo'] == (rs1 & 0x7FFFFFF)
    assert d['height'] == 4
    assert d['length'] == 256
    assert d['rd_stride'] == 0x100  # LOAD: rs2_low -> rd_stride
    assert d['wr_stride'] == 0x200  # LOAD: rs3_low -> wr_stride


def test_decode_store_swaps_strides():
    """STORE funct3=1 (xs2=1): rs2_low -> wr_stride, rs3_low -> rd_stride."""
    from riscv.gtx.dma_engine import decode_firmware_dma_args
    rs1 = 0x123456789ABCDEF
    rs2 = (2 << 48) | (128 << 32) | 0xAA
    rs3 = 0xBB
    d = decode_firmware_dma_args(rs1, rs2, rs3, xd=0, xs1=0, xs2=1)
    assert d['is_store'] is True
    assert d['is_copy'] is False
    assert d['funct3'] == 1
    assert d['wr_stride'] == 0xAA
    assert d['rd_stride'] == 0xBB


def test_decode_copy_uses_high_32_bits():
    """Pitfall 1: COPY funct3=010 (xs2=0,xs1=1) must use rs1>>32 not rs1>>27."""
    from riscv.gtx.dma_engine import decode_firmware_dma_args
    # Set rs1 with distinguishable hi (bits 32-63) and lo (bits 0-26)
    rs1 = (0xABCDEF12 << 32) | 0x07654321  # bit 31 is 0
    rs2 = 0
    rs3 = 0
    d = decode_firmware_dma_args(rs1, rs2, rs3, xd=0, xs1=1, xs2=0)
    assert d['is_store'] is False
    assert d['is_copy'] is True
    assert d['funct3'] == 2
    # is_copy carve-out: addr_hi = rs1 >> 32
    assert d['addr_hi'] == 0xABCDEF12
    # NOT (rs1 >> 27) & 0x1FFFFFFFFF
    assert d['addr_hi'] != ((rs1 >> 27) & 0x1FFFFFFFFF)
    assert d['addr_lo'] == (rs1 & 0x7FFFFFF)


def test_decode_length_zero_means_65536():
    """Pitfall 2: length_raw=0 means 65536 (0x10000)."""
    from riscv.gtx.dma_engine import decode_firmware_dma_args
    rs2 = (1 << 48) | (0 << 32)  # height=1, length=0
    d = decode_firmware_dma_args(0, rs2, 0, xd=0, xs1=0, xs2=0)
    assert d['length'] == 0x10000


def test_decode_height_zero_means_one():
    """Pitfall 2: height_raw=0 means 1."""
    from riscv.gtx.dma_engine import decode_firmware_dma_args
    rs2 = (0 << 48) | (256 << 32)  # height=0, length=256
    d = decode_firmware_dma_args(0, rs2, 0, xd=0, xs1=0, xs2=0)
    assert d['height'] == 1


def test_exec_dma_2d_l2_to_l1_load():
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(64, dtype=np.uint8) + 50
    mem.l2_byte(0)[0:64] = pattern

    cy = dma_engine.exec_dma_2d(
        mem, nest_id=0, l2_addr=0, l1_addr=0,
        width=64, height=1, is_load=True, l2_stride=64, spu_id=0)
    assert cy == 0
    assert (mem.l1_byte(0, 0)[0:64] == pattern).all()


def test_exec_dma_2d_l1_to_l2_store():
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(64, dtype=np.uint8) + 200
    mem.l1_byte(0, 0)[0:64] = pattern

    dma_engine.exec_dma_2d(
        mem, nest_id=0, l2_addr=0, l1_addr=0,
        width=64, height=1, is_load=False, l2_stride=64, spu_id=0)
    assert (mem.l2_byte(0)[0:64] == pattern).all()


def test_exec_dma_2d_strided():
    """Strided LOAD: l2_stride=128 means rows are 128 bytes apart in L2,
    but contiguous (width=64) in L1."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    # Row 0 at L2[0:64], row 1 at L2[128:192], row 2 at L2[256:320], row 3 at L2[384:448]
    for row in range(4):
        mem.l2_byte(0)[row * 128 : row * 128 + 64] = (
            np.arange(64, dtype=np.uint8) + row * 10
        )

    dma_engine.exec_dma_2d(
        mem, nest_id=0, l2_addr=0, l1_addr=0,
        width=64, height=4, is_load=True, l2_stride=128, spu_id=0)

    for row in range(4):
        expected = np.arange(64, dtype=np.uint8) + row * 10
        assert (mem.l1_byte(0, 0)[row * 64 : row * 64 + 64] == expected).all()


def test_exec_dma_2d_zero_height_returns_zero():
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    cy = dma_engine.exec_dma_2d(
        mem, nest_id=0, l2_addr=0, l1_addr=0,
        width=64, height=0, is_load=True)
    assert cy == 0


def test_exec_load_svr_32_bytes():
    """L1[l1_addr:l1_addr+32] -> L0[l0_reg*32:l0_reg*32+32] in 8 x 4-byte words."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(32, dtype=np.uint8) + 100
    mem.l1_byte(0, 0)[0:32] = pattern

    dma_engine.exec_load_svr(mem, nest_id=0, spu_id=0, l1_addr=0, l0_reg=2)
    # l0_reg=2 -> l0_off = 2*32 = 64
    assert (mem.l0_byte(0, 0)[64:96] == pattern).all()


def test_exec_store_svr_round_trip():
    """L0 -> L1 reverse of load_svr."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(32, dtype=np.uint8) + 200
    # write pattern to L0[0:32]
    mem.l0_byte(0, 0)[0:32] = pattern

    dma_engine.exec_store_svr(mem, nest_id=0, spu_id=0, l1_addr=0x100, l0_reg=0)
    assert (mem.l1_byte(0, 0)[0x100:0x120] == pattern).all()

    # Round-trip: load back to L0[64:96] (l0_reg=2)
    dma_engine.exec_load_svr(mem, nest_id=0, spu_id=0, l1_addr=0x100, l0_reg=2)
    assert (mem.l0_byte(0, 0)[64:96] == pattern).all()


def test_exec_transpose_4x8_to_8x4():
    """4x8 FP16 matrix at addr_a -> 8x4 transposed at addr_r (byte-pair)."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    rows, cols = 4, 8
    addr_a = 0
    addr_r = 0x200

    # Build pattern: M[i,j] = i*cols + j (16-bit), written byte-wise LE
    for i in range(rows):
        for j in range(cols):
            val = i * cols + j
            off = addr_a + (j + cols * i) * 2
            mem.l1_byte(0, 0)[off] = val & 0xFF
            mem.l1_byte(0, 0)[off + 1] = (val >> 8) & 0xFF

    dma_engine.exec_transpose(
        mem, nest_id=0, spu_id=0, rows=rows, cols=cols,
        addr_a=addr_a, addr_r=addr_r)

    # Transposed: T[j,i] = M[i,j]
    for i in range(rows):
        for j in range(cols):
            val = i * cols + j
            off = addr_r + (i + rows * j) * 2
            actual_lo = mem.l1_byte(0, 0)[off]
            actual_hi = mem.l1_byte(0, 0)[off + 1]
            assert actual_lo == (val & 0xFF), f"i={i} j={j} lo mismatch"
            assert actual_hi == ((val >> 8) & 0xFF), f"i={i} j={j} hi mismatch"


def test_exec_fill_writes_le_byte_pair():
    """fill_val=0x3C00 (FP16 1.0) -> L1 contains [0x00, 0x3C, 0x00, 0x3C, ...]"""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    dma_engine.exec_fill(
        mem, nest_id=0, spu_id=0, length=10, fill_val=0x3C00, addr_r=0x80)
    expected = np.array([0x00, 0x3C] * 10, dtype=np.uint8)
    assert (mem.l1_byte(0, 0)[0x80:0x80 + 20] == expected).all()


def test_exec_transpose_ddr_identity_perm():
    """Identity perm: dim2,dim1,dim0 with p2=2,p1=1,p0=0 should match input."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx.ddr import ensure_ddr
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    # 2x3x4 tensor of FP16 (each elem 2 bytes), 24 elements -> 48 bytes
    dim2, dim1, dim0 = 2, 3, 4
    nelem = dim2 * dim1 * dim0
    ensure_ddr(mem, 256)

    # Pattern: byte index i mod 256 at offset i, for src
    src_addr = 0
    dst_addr = 128
    for i in range(nelem):
        mem._ddr_bytes[src_addr + i * 2] = i & 0xFF
        mem._ddr_bytes[src_addr + i * 2 + 1] = (i >> 8) & 0xFF

    dma_engine.exec_transpose_ddr(
        mem, src_addr=src_addr, dst_addr=dst_addr,
        dim2=dim2, dim1=dim1, dim0=dim0,
        p2=2, p1=1, p0=0)

    # Identity perm: output should equal input byte-wise
    assert (mem._ddr_bytes[dst_addr:dst_addr + nelem * 2] ==
            mem._ddr_bytes[src_addr:src_addr + nelem * 2]).all()


def test_firmware_dma_sloop_store_pushes_one_request():
    """firmware_dma_sloop_store appends DeferredDdrStore with correct fields."""
    from riscv.gtx.dma_engine import (
        DeferredDdrStore, firmware_dma_sloop_store)
    from riscv.gtx.params import GTX_DDR_BASE

    class _DummyNpu:
        def __init__(self):
            self.deferred_ddr_stores = []

    npu = _DummyNpu()
    # Use addr_hi >= GTX_DDR_BASE to test the offset translation
    addr_hi = GTX_DDR_BASE + 0x1000
    rc = firmware_dma_sloop_store(
        npu, nest=2, addr_hi=addr_hi, addr_lo=0x500,
        length=256, height=4, rd_stride=512, wr_stride=1024)
    assert rc == 0
    assert len(npu.deferred_ddr_stores) == 1
    req = npu.deferred_ddr_stores[0]
    assert isinstance(req, DeferredDdrStore)
    assert req.nest == 2
    assert req.l2_off == 0x500
    assert req.ddr_off == 0x1000  # addr_hi - GTX_DDR_BASE
    assert req.length == 256
    assert req.height == 4
    assert req.l2_stride == 512
    assert req.ddr_stride == 1024


def test_firmware_dma_sloop_load_immediate_copy():
    """firmware_dma_sloop_load: DDR -> L2 immediate row-by-row."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx.ddr import ensure_ddr
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    ensure_ddr(mem, 1024)
    pattern = (np.arange(64, dtype=np.uint8) + 50)
    mem._ddr_bytes[0:64] = pattern

    rc = dma_engine.firmware_dma_sloop_load(
        mem, nest=0, addr_hi=0, addr_lo=0,
        length=64, height=1, rd_stride=0, wr_stride=0)
    assert rc == 0
    assert (mem.l2_byte(0)[0:64] == pattern).all()


def test_firmware_dma_tloop_load_store_l2_l1():
    """firmware_dma_tloop_load_store: L2 <-> L1 strided per-row."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(128, dtype=np.uint8) + 30
    mem.l2_byte(0)[0:128] = pattern

    # LOAD: L2 -> L1
    rc = dma_engine.firmware_dma_tloop_load_store(
        mem, nest=0, spu=0, is_store=False,
        addr_hi=0, addr_lo=0,
        length=64, height=2, rd_stride=64, wr_stride=0)
    assert rc == 0
    # row 0: L2[0:64] -> L1[0:64]
    # row 1: L2[64:128] -> L1[64:128]
    assert (mem.l1_byte(0, 0)[0:128] == pattern).all()

    # STORE: L1 -> L2 (different L2 region)
    mem.l1_byte(0, 0)[0:64] = np.arange(64, dtype=np.uint8) + 99
    rc = dma_engine.firmware_dma_tloop_load_store(
        mem, nest=0, spu=0, is_store=True,
        addr_hi=0x1000, addr_lo=0,
        length=64, height=1, rd_stride=0, wr_stride=64)
    assert rc == 0
    expected = np.arange(64, dtype=np.uint8) + 99
    assert (mem.l2_byte(0)[0x1000:0x1040] == expected).all()


def test_firmware_dma_tloop_copy_l1_to_l1():
    """firmware_dma_tloop_copy: L1 -> L1 same-SPU; .copy() guards overlapping ranges."""
    from riscv.gtx.memory import GtxMemory
    from riscv.gtx import dma_engine

    mem = GtxMemory()
    pattern = np.arange(64, dtype=np.uint8) + 7
    mem.l1_byte(0, 0)[0:64] = pattern

    # Non-overlapping copy
    rc = dma_engine.firmware_dma_tloop_copy(
        mem, nest=0, spu=0,
        src_addr=0, dst_addr=0x100,
        length=64, height=1)
    assert rc == 0
    assert (mem.l1_byte(0, 0)[0x100:0x140] == pattern).all()

    # Overlapping copy: src=0, dst=32, len=64.
    # Without .copy(), numpy slice assignment may corrupt overlapping ranges.
    mem.l1_byte(0, 0)[0:128] = 0  # clear
    base = np.arange(64, dtype=np.uint8) + 1  # 1..64
    mem.l1_byte(0, 0)[0:64] = base
    rc = dma_engine.firmware_dma_tloop_copy(
        mem, nest=0, spu=0,
        src_addr=0, dst_addr=32,
        length=64, height=1)
    assert rc == 0
    # After: L1[32:96] should equal original L1[0:64] = base
    assert (mem.l1_byte(0, 0)[32:96] == base).all()
