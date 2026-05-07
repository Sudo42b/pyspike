"""P5 pooling op unit tests -- Wave 1b plan 04 GREEN.

Covers ACT-03: max-pool + avg-pool with stride = kernel_size, output length
n_out = n_in / kernel_size, signed-zero canonicalization, always-forward
direction (ADDRA -> ADDRR per gtx_npu_act.cc:177-178).

Plan 04 lands act_core.pool_max + pool_avg + act_engine.firmware_pool.
"""
import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

# pylint: disable=import-error,wrong-import-position
from riscv.gtx import act_engine, act_core
from riscv.gtx.encoding import (
    GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from riscv.gtx.npu import GtxNpu

from tests.gtx._mocks import MockProcessor, MockInsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_npu():
    """GtxNpu with default L1 ADDRA/ADDRR offsets (avoid zero-collision)."""
    npu = GtxNpu()
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0x0000
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0x2000
    return npu


def _make_insn(*, rs1_idx: int = 1, rs2_idx: int = 2, rd_idx: int = 0,
                funct: int = 0, funct3: int = 0) -> MockInsn:
    xd = (funct3 >> 2) & 1
    xs1 = (funct3 >> 1) & 1
    xs2 = funct3 & 1
    return MockInsn(funct=funct, rs1=rs1_idx, rs2=rs2_idx, rd=rd_idx,
                    xd=xd, xs1=xs1, xs2=xs2)


# =========================================================================
# act_core kernel-level tests
# =========================================================================
def test_max_pool_output_length():
    """ACT-03: Max-pool output length = n_in / kernel_size (integer div).
    Source: gtx_npu_act.cc:195. Non-overlapping kernel-stride windows; no padding."""
    arr = np.array([1, 3, 2, 4, 5, 7, 6, 8], dtype=np.float16)
    out = act_core.pool_max(arr, kernel_size=4)
    assert out.dtype == np.float16
    assert out.shape == (2,)
    assert (out == np.array([4, 8], dtype=np.float16)).all()

    # Tail-discard: length=10 with kernel=4 => 2 windows (last 2 elements dropped).
    arr2 = np.array([1, 2, 3, 9, 5, 6, 7, 8, 99, 100], dtype=np.float16)
    out2 = act_core.pool_max(arr2, kernel_size=4)
    assert out2.shape == (2,)
    assert (out2 == np.array([9, 8], dtype=np.float16)).all()


def test_avg_pool_signed_zero_canon():
    """ACT-03: Avg-pool canonicalizes -0.0 -> +0.0 via `avg += 0.0` (line 211).
    Hex output is deterministic; -0.0 (0x8000) and +0.0 (0x0000) have
    different bit patterns; canon ensures golden-hex matching."""
    arr = np.array([0.0, -0.0, -0.0, 0.0], dtype=np.float16)
    out = act_core.pool_avg(arr, kernel_size=2)
    assert out.shape == (2,)
    # Both windows have +0.0 + (-0.0) = -0.0 in some FP16 paths; canon must yield +0.0.
    out_u16 = out.view(np.uint16)
    assert int(out_u16[0]) == 0x0000, f"window 0 should be +0.0; got 0x{int(out_u16[0]):04x}"
    assert int(out_u16[1]) == 0x0000, f"window 1 should be +0.0; got 0x{int(out_u16[1]):04x}"

    # Sanity: non-zero windows still average correctly.
    arr2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    out2 = act_core.pool_avg(arr2, kernel_size=2)
    assert (out2 == np.array([1.5, 3.5], dtype=np.float16)).all()


def test_pool_always_forward():
    """ACT-03: firmware_pool always reads ADDRA, writes ADDRR (CONTEXT D-08).
    Distinct sentinel patterns at ADDRA + ADDRR; after pool, only ADDRR
    must be overwritten with the pooled result; ADDRA must be unchanged."""
    npu = _new_npu()
    proc = MockProcessor()

    # Seed length=8 at ADDRA (0x0000) with [1,3,2,4,5,7,6,8] FP16; ADDRR with sentinel 7.5
    addra = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addrr = npu.lspr[0][0][LSPR_SPM_ADDRR]
    l1_f16 = npu.mem.l1_f16(0, 0)
    addra_off = addra // 2
    addrr_off = addrr // 2
    l1_f16[addra_off:addra_off + 8] = np.array(
        [1, 3, 2, 4, 5, 7, 6, 8], dtype=np.float16)
    l1_f16[addrr_off:addrr_off + 8] = np.array(
        [7.5] * 8, dtype=np.float16)

    # Stage operands: length in OPERAND1, kernel_size in OPERAND2.
    npu.gspr[GSPR_GTX_OPERAND1] = 8
    npu.gspr[GSPR_GTX_OPERAND2] = 4

    insn = _make_insn(rs1_idx=1, rs2_idx=2)
    proc.state.XPR.write(1, 8)  # length staged via XPR for engine length read

    rc = act_engine.firmware_pool(npu, proc, insn, is_max=True)
    assert rc == 0

    # ADDRA preserved verbatim
    assert (l1_f16[addra_off:addra_off + 8] ==
            np.array([1, 3, 2, 4, 5, 7, 6, 8], dtype=np.float16)).all(), \
        "ADDRA must NOT be mutated by forward pool"

    # ADDRR has 2 max-pool outputs followed by sentinels
    assert l1_f16[addrr_off + 0] == np.float16(4)
    assert l1_f16[addrr_off + 1] == np.float16(8)
    # Tail beyond out_len=2 must remain sentinel
    assert l1_f16[addrr_off + 2] == np.float16(7.5)
    assert l1_f16[addrr_off + 7] == np.float16(7.5)
