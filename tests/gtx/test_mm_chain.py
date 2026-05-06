"""P4 MM chain integration scaffolds (test_mm_chain.py).

Covers MM-04 (ADDRC + mxe_accum + isolation + dtype).
Wave 1 plans (gemm_core / mm_engine / ops/mm) GREEN-fill these.
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_mm_addrc_chain_continuity():
    """MM-04 / ROADMAP success #2: mm.s -> mmc.s -> mmc chain via ADDRC FP32 bias.

    NOTE per RESEARCH Pitfall B: this is the ADDRC-bias chain, NOT an mxe_accum
    chain. mm.s/mmc.s/mmc variants do NOT touch mxe_accum -- they use
    LSPR_SPM_ADDRC=0x902."""
    pytest.skip("Wave 1: ops/mm + mm_engine not yet built")


def test_mxe_accum_chain_continuity():
    """MM-04 / Pitfall 3: mm.o -> mmc.o chain on mxe_accum[(nest=1,spu=5)] --
    only MM_O/MMC_O/MM_V/MMC_V touch mxe_accum."""
    pytest.skip("Wave 1: ops/mm + mm_engine not yet built")


def test_mxe_accum_per_cell_isolation():
    """MM-04: only mxe_accum[1,5] mutates; other 4*16-1=63 cells unchanged
    (snapshot diff)."""
    pytest.skip("Wave 1: ops/mm not yet built")


def test_mxe_accum_dtype_locked():
    """MM-04: npu._mxe_accum.dtype == np.float32 stays float32 across chain
    (Pitfall 3 dtype-slip guard)."""
    pytest.skip("Wave 1: ops/mm not yet built")
