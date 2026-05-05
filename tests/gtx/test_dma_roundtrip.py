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
"""DMA-05 integration tests: full L1 <-> L2 <-> DDR round-trip bit-exactness.

Plan 05 Task 2. Validates Plan 01 (dma_engine helpers + DeferredDdrStore
queue), Plan 02 (npu.deferred_ddr_stores + flush_deferred_ddr_stores), Plan
03 (ddr_init_from_file + ddr_dump_to_file), and Plan 04 (dispatch_4mode +
dispatch_iss_opcode credit_st_chk wiring) all working together end-to-end.

Round-trip recipe (per 03-RESEARCH "Test Patterns" §test_dma_roundtrip.py):

  1. Pre-populate L1 with FP16 pattern (np.arange).
  2. Forward L1 -> L2 via dma_engine.exec_dma_2d (is_load=False).
  3. Forward L2 -> DDR via firmware_dma_sloop_store + flush_deferred_ddr_stores;
     verify pre-flush DDR is zero, post-flush matches L2.
  4. Dump DDR to file (LTR mode).
  5. New GtxNpu; ddr_init_from_file; verify DDR bytes identical.
  6. Reverse DDR -> L2 via firmware_dma_sloop_load.
  7. Reverse L2 -> L1 via exec_dma_2d (is_load=True).
  8. Final byte-exact assertion: L1.view(uint16) == pattern.view(uint16).

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
    reason="dma_roundtrip tests use GtxNpu; require _riscv.so; see Plan 05",
)


def _make_npu():
    from riscv.gtx import GtxNpu
    return GtxNpu()


# ============================================================================
# Full L1 -> L2 -> DDR -> file -> re-init -> L2 -> L1 round-trip
# ============================================================================
def test_dma_l1_to_ddr_roundtrip_ltr(tmp_path, monkeypatch):
    """DMA-05: full round-trip in default LTR mode is byte-exact."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    from riscv.gtx import dma_engine
    from riscv.gtx.ddr import ddr_dump_to_file, ddr_init_from_file, ensure_ddr

    npu = _make_npu()
    pattern = np.arange(4096, dtype=np.float16)
    npu.mem.l1_f16(0, 0)[0:4096] = pattern

    # Forward L1 -> L2 (8192 bytes = 4096 fp16 elements x 2 bytes).
    dma_engine.exec_dma_2d(npu.mem, nest_id=0,
                            l2_addr=0, l1_addr=0,
                            width=8192, height=1, is_load=False)
    l1_bytes = bytes(npu.mem.l1_byte(0, 0)[0:8192])
    l2_bytes = bytes(npu.mem.l2_byte(0)[0:8192])
    assert l1_bytes == l2_bytes

    # Forward L2 -> DDR via deferred-store queue.
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 0
    dma_engine.firmware_dma_sloop_store(
        npu, nest=0, addr_hi=0, addr_lo=0,
        length=8192, height=1, rd_stride=8192, wr_stride=8192,
    )
    assert len(npu.deferred_ddr_stores) == 1

    # Pre-flush snapshot -- DDR is untouched (deferred queue is a no-op).
    ensure_ddr(npu.mem, 8192)
    pre_flush = bytes(npu.mem._ddr_bytes[0:8192])
    assert pre_flush == bytes(8192)   # all zeros

    npu.flush_deferred_ddr_stores()
    post_flush = bytes(npu.mem._ddr_bytes[0:8192])
    assert post_flush == l2_bytes
    assert pre_flush != post_flush
    assert npu.deferred_ddr_stores == []

    # Dump DDR to file.
    hexf = tmp_path / "rt.hex"
    ddr_dump_to_file(npu.mem, str(hexf), 0, 8192)
    assert hexf.exists()
    assert hexf.stat().st_size > 0

    # Re-init in fresh NPU.
    npu2 = _make_npu()
    ddr_init_from_file(npu2.mem, str(hexf))
    assert bytes(npu2.mem._ddr_bytes[0:8192]) == post_flush

    # Reverse DDR -> L2.
    dma_engine.firmware_dma_sloop_load(
        npu2.mem, nest=0, addr_hi=0, addr_lo=0,
        length=8192, height=1, rd_stride=8192, wr_stride=8192,
    )
    assert bytes(npu2.mem.l2_byte(0)[0:8192]) == l2_bytes

    # Reverse L2 -> L1.
    dma_engine.exec_dma_2d(npu2.mem, nest_id=0,
                            l2_addr=0, l1_addr=0,
                            width=8192, height=1, is_load=True)

    # Final byte-exact assertion via uint16 view.
    final_l1_u16 = npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16)
    assert np.array_equal(final_l1_u16, pattern.view(np.uint16))


def test_dma_l1_to_ddr_roundtrip_reversed(tmp_path, monkeypatch):
    """DMA-05 + DMA-04: full round-trip in REVERSED mode also bit-exact.

    The dump+init cancel out across the boundary -- both reverse the same
    32-byte windows, so the in-memory DDR bytes match across npu / npu2.
    """
    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    from riscv.gtx import dma_engine
    from riscv.gtx.ddr import ddr_dump_to_file, ddr_init_from_file

    npu = _make_npu()
    pattern = np.arange(4096, dtype=np.float16)
    npu.mem.l1_f16(0, 0)[0:4096] = pattern

    dma_engine.exec_dma_2d(npu.mem, nest_id=0, l2_addr=0, l1_addr=0,
                            width=8192, height=1, is_load=False)
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    dma_engine.firmware_dma_sloop_store(
        npu, nest=0, addr_hi=0, addr_lo=0,
        length=8192, height=1, rd_stride=8192, wr_stride=8192,
    )
    npu.flush_deferred_ddr_stores()
    ddr_pre = bytes(npu.mem._ddr_bytes[0:8192])

    hexf = tmp_path / "rt_rev.hex"
    ddr_dump_to_file(npu.mem, str(hexf), 0, 8192)

    npu2 = _make_npu()
    ddr_init_from_file(npu2.mem, str(hexf))
    # Reversed dump + reversed init -> same in-memory bytes.
    assert bytes(npu2.mem._ddr_bytes[0:8192]) == ddr_pre

    dma_engine.firmware_dma_sloop_load(
        npu2.mem, nest=0, addr_hi=0, addr_lo=0,
        length=8192, height=1, rd_stride=8192, wr_stride=8192,
    )
    dma_engine.exec_dma_2d(npu2.mem, nest_id=0, l2_addr=0, l1_addr=0,
                            width=8192, height=1, is_load=True)
    assert np.array_equal(
        npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16),
        pattern.view(np.uint16),
    )


def test_dma_l1_to_l1_copy_via_firmware_dma_tloop_copy():
    """DMA-05 ancillary: L1 -> L1 same-SPU copy is bit-exact.

    Validates Plan 01 firmware_dma_tloop_copy std::memmove semantics
    (the .copy() guard against overlapping numpy slice corruption).
    """
    from riscv.gtx import dma_engine
    npu = _make_npu()
    src_pattern = np.arange(2048, dtype=np.float16)
    # Write to L1 starting at offset 0 (fp16 elements 0..2048).
    # Copy to L1 starting at offset 4096 bytes (fp16 elements 2048..4096).
    npu.mem.l1_f16(0, 0)[0:2048] = src_pattern
    dma_engine.firmware_dma_tloop_copy(
        npu.mem, nest=0, spu=0,
        src_addr=0, dst_addr=4096, length=4096, height=1,
    )
    assert np.array_equal(
        npu.mem.l1_f16(0, 0)[2048:4096].view(np.uint16),
        src_pattern.view(np.uint16),
    )
