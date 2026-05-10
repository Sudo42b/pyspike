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
"""P8 MTDMA-03 + MTDMA-04: tile-1 <-> tile-2 boundary regression guard.

Vendor-`.elf`-free unit test (D-09). Drives a 2-tile DMA+ABS sequence
programmatically through MockProcessor and asserts:
  - state-machine reset across tile boundaries (MTDMA-04, 8 transition rows
    per RESEARCH.md "State-Machine Reset Audit")
  - DDR byte-exact match against np.abs(input).astype(np.float16) for
    tile 0 AND tile 1 (MTDMA-03; RED before Plan 04 fix lands, GREEN after)

Self-contained _RISCV_AVAILABLE detection (Phase 5 plan-05 D-01 pattern;
matches test_dma_roundtrip.py / test_deferred_store.py precedent) -- per-test
skipif decorators so the module loads cleanly even when _riscv.so is absent.

Plan 08-01: tasks 1 + 2 land atomically as a single Wave 0 commit (no
inter-task commit) so the _drive_full_tile helper is never stub-only on
disk. test_tile_boundary_byte_exact is decorated @pytest.mark.xfail
(strict=False) for the Wave 0 RED state; Plan 08-04 surgical fix flips it
to strict=True once GREEN.

Deviation notes (Rule 3 -- fix blocking issues to match real production API):
  - Plan's <interfaces> block listed firmware_dma_sloop_load(npu, args, ...).
    Actual signature is firmware_dma_sloop_load(mem, *, nest, addr_hi, addr_lo,
    length, height, rd_stride, wr_stride). We use the real API and skip the
    packed-args decoder (cleaner, no rs1/rs2 bit-shuffling).
  - Plan listed firmware_vec_op(npu, proc, insn, xs1, xs2). Actual is
    firmware_vec_op(npu, proc, insn) -- it reads xs1/xs2 from insn.rs1/rs2
    via proc.state.XPR. We seed XPR with proc.state.XPR.write(idx, val).
  - Plan listed npu.reset() with no arg. Actual signature is reset(proc).
    A freshly-constructed GtxNpu() already has zero-init state, so we skip
    the explicit reset() call and assert the post-construction invariants
    directly.
"""
from __future__ import annotations

import numpy as np
import pytest

# Module-level _RISCV_AVAILABLE detection (self-contained, --noconftest safe).
try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

from tests.gtx._mocks import MockProcessor, MockInsn  # noqa: E402


# Fixture geometry (D-09 + RESEARCH "Plan-Stage Hand-Off"):
#   HEIGHT          = SHARED_TILE_MAX_ROWS + 2 -> exactly 2 tiles
#   ROW_BYTES       = 16 (8 FP16 per row)
#   TILE_MAX_ROWS   = MAX_SHARED_DMA_BYTES / ROW_BYTES = 65535 / 16 = 4095
HEIGHT = 4097
ROW_BYTES = 16
TILE_MAX_ROWS = 4095
NEST_ID = 0
SPU_ID = 0

# DDR layout: input region at 0x0, result region at 0x10_0000 (well above
# input); chosen above any plausible input length so the two regions never
# overlap for HEIGHT up to ~1M rows.
INPUT_DDR_BASE = 0x0
RESULT_DDR_BASE = 0x10_0000

# L1 staging addresses (firmware writes via WRSPR LSPR_SPM_ADDRA / ADDRR
# before each tile). Distinct so LOAD->compute->STORE doesn't alias.
L1_ADDRA_BASE = 0x0
L1_ADDRR_BASE = 0x10000


# =========================================================================
# Helper: programmatic firmware tile sequence
# =========================================================================
def _drive_full_tile(npu, proc, tile_idx: int, total_height: int) -> None:
    """Programmatic firmware tile sequence (mirrors n1s16_abs.c kernel body).

    Per-tile sequence (matches RESEARCH.md "Plan-Stage Hand-Off" spec):
        WRSPR LSPR_SPM_ADDRA + ADDRR
        start_p(NEST_ID)
        start_s(0)
        firmware_dma LOAD (DDR -> L2)
        exec_dma_2d (L2 -> L1)
        end_s
        start_t(SPU_ID)
        firmware_vec_op SIGN-ABS  (L1 -> L1)
        end_t
        start_s(0)
        exec_dma_2d (L1 -> L2)
        firmware_dma STORE (L2 -> DDR via deferred queue)
        end_s
        end_p   (flushes deferred queue via !wsplit_seen path)

    Tile geometry:
        tile_rows = min(TILE_MAX_ROWS, total_height - tile_idx*TILE_MAX_ROWS)
        tile_byte_len_per_row = ROW_BYTES   (length param of firmware_dma)
        ddr_off_in  = tile_idx * TILE_MAX_ROWS * ROW_BYTES
        ddr_off_out = RESULT_DDR_BASE + tile_idx * TILE_MAX_ROWS * ROW_BYTES

    The firmware_dma_sloop_load/store API operates per-row with
    length=ROW_BYTES + height=tile_rows + stride=ROW_BYTES (row-major
    contiguous), matching the vendor n1s16_abs.c packed encoding.
    """
    from riscv.gtx import dma_engine
    from riscv.gtx.vec_engine import firmware_vec_op
    from riscv.gtx.ops import control as ctl
    from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRR

    tile_rows = min(TILE_MAX_ROWS, total_height - tile_idx * TILE_MAX_ROWS)
    if tile_rows <= 0:
        return
    tile_byte_len = tile_rows * ROW_BYTES
    ddr_off_in = INPUT_DDR_BASE + tile_idx * TILE_MAX_ROWS * ROW_BYTES
    ddr_off_out = RESULT_DDR_BASE + tile_idx * TILE_MAX_ROWS * ROW_BYTES

    # --- WRSPR per-tile LSPR addresses (firmware does this once per tile) ---
    npu.lspr[NEST_ID][SPU_ID][LSPR_SPM_ADDRA] = L1_ADDRA_BASE
    npu.lspr[NEST_ID][SPU_ID][LSPR_SPM_ADDRR] = L1_ADDRR_BASE

    # --- start_p; start_s ---
    ctl._do_startp(npu, NEST_ID, 0)
    ctl._do_starts(npu, 0, 0)

    # --- firmware_dma LOAD: DDR -> L2 (S-loop branch) ---
    # Per-row contiguous: length=ROW_BYTES, height=tile_rows, strides=ROW_BYTES
    dma_engine.firmware_dma_sloop_load(
        npu.mem,
        nest=NEST_ID,
        addr_hi=ddr_off_in,
        addr_lo=0,                      # L2 base
        length=ROW_BYTES,
        height=tile_rows,
        rd_stride=ROW_BYTES,
        wr_stride=ROW_BYTES,
    )

    # T-loop transfer L2 -> L1 (firmware_dma_tloop is the canonical T-loop;
    # exec_dma_2d is the lower-level helper that supports L2<->L1 strided
    # 2D copy; both produce identical bytes for contiguous rows).
    dma_engine.exec_dma_2d(
        npu.mem,
        nest_id=NEST_ID,
        spu_id=SPU_ID,
        l2_addr=0,
        l1_addr=L1_ADDRA_BASE,
        width=ROW_BYTES,
        height=tile_rows,
        is_load=True,
    )
    ctl._do_ends(npu, 0, 0)

    # --- start_t; firmware_vec_op SIGN-ABS; end_t ---
    ctl._do_startt(npu, SPU_ID, 0)

    # firmware_vec_op reads vec_size from rs1[15:0] via proc.state.XPR[insn.rs1].
    # Seed XPR[1] = vec_size (= tile_rows * 8 fp16 elements).
    vec_size = tile_rows * 8
    proc.state.XPR.write(1, vec_size)
    proc.state.XPR.write(2, 0)        # rs2 unused for unary SIGN-ABS
    insn = MockInsn(
        opcode=0x0b,
        funct=0x1D,                   # GTX_F7_VEC_SIGN -> SIGN family
        xd=0, xs1=0, xs2=0,           # funct3 = (xd<<2)|(xs1<<1)|xs2 = 0 -> ABS sub-op
        rs1=1, rs2=2,
        rd=3,
    )
    firmware_vec_op(npu, proc, insn)

    ctl._do_endt(npu, 0, 0)

    # --- start_s; transfer L1 -> L2; firmware_dma STORE; end_s ---
    ctl._do_starts(npu, 0, 0)
    dma_engine.exec_dma_2d(
        npu.mem,
        nest_id=NEST_ID,
        spu_id=SPU_ID,
        l2_addr=0,
        l1_addr=L1_ADDRR_BASE,
        width=ROW_BYTES,
        height=tile_rows,
        is_load=False,
    )
    dma_engine.firmware_dma_sloop_store(
        npu,
        nest=NEST_ID,
        addr_hi=ddr_off_out,
        addr_lo=0,                      # L2 base
        length=ROW_BYTES,
        height=tile_rows,
        rd_stride=ROW_BYTES,
        wr_stride=ROW_BYTES,
    )
    ctl._do_ends(npu, 0, 0)

    # --- end_p (flushes deferred queue via !wsplit_seen path) ---
    ctl._do_endp(npu, 0, 0)


# =========================================================================
# MTDMA-04: state-machine reset audit (verify-only -- no compute assertions)
# =========================================================================
@pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="GtxNpu construction requires _riscv.so (rocc_t base class)",
)
def test_tile_boundary_state_reset() -> None:
    """MTDMA-04: assert state-machine reset at tile 1 entry per RESEARCH.md
    'State-Machine Reset Audit' 8 transition rows.

    Pre-test invariants (clean construction): tmu_id == 0, curr_id == 0, all
    *loop flags False, deferred queue empty, mxe_accum all-zero.

    Post-tile-0 end_p: deferred queue empty (flushed via !wsplit_seen),
    is_ploop / is_tloop / is_sloop all False.

    Post-tile-1 end_p: same invariants -- proves no cross-tile state leakage.
    Plus: tmu_id == NEST_ID (start_p overwrites fresh), curr_id == SPU_ID
    (start_t overwrites fresh), mxe_accum still zero (ABS does not touch mxe).
    """
    from riscv.gtx import GtxNpu

    npu = GtxNpu()
    proc = MockProcessor()

    # --- Pre-test invariants (clean GtxNpu() construction) ---
    assert npu.warp.tmu_id == 0
    assert npu.warp.curr_id == 0
    assert npu.warp.is_ploop is False
    assert npu.warp.is_tloop is False
    assert npu.warp.is_sloop is False
    assert len(npu.deferred_ddr_stores) == 0
    assert (npu._mxe_accum == 0).all()

    # --- Drive tile 0 (rows 0..4094, 4095 rows) ---
    _drive_full_tile(npu, proc, tile_idx=0, total_height=HEIGHT)

    # MTDMA-04 row "end_p when !wsplit_seen": deferred queue flushed.
    assert len(npu.deferred_ddr_stores) == 0, (
        "tile 0 end_p must flush deferred queue (!wsplit_seen path)"
    )
    # MTDMA-04 row "end_p clears is_ploop"
    assert npu.warp.is_ploop is False, "end_p must clear is_ploop"
    # MTDMA-04 row "end_t clears is_tloop"
    assert npu.warp.is_tloop is False, "end_t must clear is_tloop"
    # MTDMA-04 row "end_s clears is_sloop"
    assert npu.warp.is_sloop is False, "end_s must clear is_sloop"

    # --- Drive tile 1 (rows 4095..4096, 2 rows) ---
    _drive_full_tile(npu, proc, tile_idx=1, total_height=HEIGHT)

    # MTDMA-04 row "start_p overwrites tmu_id fresh per RESEARCH MATCH-for-none delta"
    assert npu.warp.tmu_id == NEST_ID, (
        "start_p must overwrite tmu_id fresh (no stale value across tiles)"
    )
    # MTDMA-04 row "start_t overwrites curr_id fresh"
    assert npu.warp.curr_id == SPU_ID, (
        "start_t must overwrite curr_id fresh (no stale value across tiles)"
    )
    # MTDMA-04 row "end_p flushes deferred queue across tile boundary"
    assert len(npu.deferred_ddr_stores) == 0, (
        "tile 1 end_p must flush deferred queue (no leak across tiles)"
    )
    # MTDMA-04 ABS-path invariant: vec_engine SIGN-ABS path does not touch mxe.
    assert (npu._mxe_accum == 0).all(), (
        "ABS (vec_engine SIGN-ABS) must not write mxe_accum"
    )
    # MTDMA-04 row "all *loop flags cleared at tile end"
    assert npu.warp.is_ploop is False
    assert npu.warp.is_tloop is False
    assert npu.warp.is_sloop is False


# =========================================================================
# MTDMA-03: byte-exact tile-boundary RED-state proof
# =========================================================================
@pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="GtxNpu construction requires _riscv.so (rocc_t base class)",
)
def test_tile_boundary_byte_exact() -> None:
    """MTDMA-03 GREEN: tile 0 + tile 1 ABS output byte-exact (post Plan 08-04 fix).

    Plan 01 created this test as RED (xfail strict=False) anticipating that
    multi-tile divergence would surface here. Plan 03 INVESTIGATION confirmed
    the divergence reproduces only via the vendor `.elf` RoCC dispatch path
    (not via this programmatic API path); Plan 04 fix landed the production
    correction (credit.ld.chk -> deferred queue flush) which the vendor sweep
    (test_regression_fw_full_sweep.py) exercises end-to-end. This unit test
    guards the programmatic API path against any future regression that might
    reintroduce a tile-boundary bug.

    Both tiles must be byte-exact for the test to PASS; xfail decorator
    removed by Plan 04 so any regression hard-fails.
    """
    from riscv.gtx import GtxNpu
    from riscv.gtx.ddr import ensure_ddr

    npu = GtxNpu()
    proc = MockProcessor()

    # Pre-stage DDR input: deterministic FP16 pattern with mixed signs so
    # ABS is distinguishable from identity.
    fp16_count = HEIGHT * 8                                  # 8 fp16 per row
    rng = np.arange(fp16_count, dtype=np.int16) - (fp16_count // 2)
    input_fp16 = rng.astype(np.float16)                       # mix of +/-
    input_bytes = input_fp16.view(np.uint8)
    input_byte_len = len(input_bytes)                         # = HEIGHT*16

    # Allocate DDR large enough to cover both input + result regions.
    ensure_ddr(npu.mem, RESULT_DDR_BASE + HEIGHT * ROW_BYTES)
    npu.mem._ddr_bytes[INPUT_DDR_BASE:INPUT_DDR_BASE + input_byte_len] = input_bytes

    # Golden: ABS via FP32-internal-then-cast (matches vec_engine SIGN-ABS path).
    expected_fp16 = np.abs(input_fp16.astype(np.float32)).astype(np.float16)
    expected_u16 = expected_fp16.view(np.uint16)

    # --- Drive both tiles ---
    _drive_full_tile(npu, proc, tile_idx=0, total_height=HEIGHT)
    _drive_full_tile(npu, proc, tile_idx=1, total_height=HEIGHT)

    # Read back the result region (HEIGHT * 16 bytes = HEIGHT * 8 fp16).
    result_bytes = bytes(
        npu.mem._ddr_bytes[RESULT_DDR_BASE:RESULT_DDR_BASE + HEIGHT * ROW_BYTES]
    )
    actual_u16 = np.frombuffer(result_bytes, dtype=np.uint16)

    # Tile 0 (rows 0..4094, fp16 elements 0..4095*8) -- expected PASS pre-fix.
    np.testing.assert_array_equal(
        actual_u16[0:TILE_MAX_ROWS * 8],
        expected_u16[0:TILE_MAX_ROWS * 8],
        err_msg="tile 0 byte-exact (4095 rows) -- known PASS pre-fix",
    )

    # Tile 1 (rows 4095..4096, fp16 elements 4095*8..4097*8) -- RED state on
    # current codebase. This is the actual MTDMA-01 bug surface; Plan 08-04
    # surgical fix flips this to GREEN.
    np.testing.assert_array_equal(
        actual_u16[TILE_MAX_ROWS * 8:HEIGHT * 8],
        expected_u16[TILE_MAX_ROWS * 8:HEIGHT * 8],
        err_msg="tile 1 byte-exact (2 rows) -- RED before MTDMA-01 fix",
    )
