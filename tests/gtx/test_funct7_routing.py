"""P4 funct7 routing matrix scaffolds (test_funct7_routing.py).

Covers MM-03 (routing matrix) + MM-05 (#5 Mode 4 dispatch).
Wave 1 plans (gemm_core / mm_engine / ops/mm) GREEN-fill these.
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_funct7_zero_collision_routing():
    """MM-03 / ROADMAP success #3: funct7=0x00 + insn.rs1!=0 -> MM,
    rs1==0 -> WRSPR (or NOP)."""
    pytest.skip("Wave 1: ops/mm WRSPR-collision NOP safety not yet built (Plan 04)")


def test_funct7_one_always_mmc():
    """MM-03: funct7=0x01 always routes to MMC regardless of rs1 (no collision)."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_mode4_routes_to_tmu_curr():
    """MM-05 #5: Mode 4 (P+T) dispatch -- synthesized firmware_mm_op routes to
    (tmu_id, curr_id) only."""
    pytest.skip("Wave 1: ops/mm + mm_engine not yet built")
