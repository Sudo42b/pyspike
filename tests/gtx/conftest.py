"""Pytest fixtures for tests/gtx/* -- ORDER.md FSM smoke set.

Invariants enforced here:
  - GTX_USE_CUDA env var gates cupy presence. Collection fails fast only if
    GTX_USE_CUDA=1 is set without cupy installed; the numpy default needs no
    GPU, so test collection succeeds on no-GPU boxes (D-01/D-04).
  - GTX_DDR_LOAD / GTX_DDR_DUMP env vars are cleared so vendor binary I/O
    never fires during tests.
"""
from __future__ import annotations
import os

import pytest

# GTX_USE_CUDA gate. Default path = numpy. Only assert cupy presence when
# user explicitly opts in.
if os.environ.get("GTX_USE_CUDA", "").strip() in ("1", "true", "TRUE"):
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.exit(
            "GTX_USE_CUDA=1 set but cupy is not installed. "
            "Install with: pip install 'spike[cuda]'",
            returncode=1,
        )

# Strip vendor DDR I/O env vars before any fixture builds a GtxNpu.
for _var in ("GTX_DDR_LOAD", "GTX_DDR_DUMP"):
    os.environ.pop(_var, None)

from ._mocks import DummyInsn, MockProcessor  # noqa: E402


@pytest.fixture(scope="function")
def mock_proc() -> MockProcessor:
    """Fresh MockProcessor per test (function scope avoids XPR/CSR leak)."""
    return MockProcessor()


@pytest.fixture(scope="function")
def dummy_insn() -> DummyInsn:
    """Default-zeroed rocc_insn_t stand-in. Override fields per-test."""
    return DummyInsn()


@pytest.fixture(scope="function")
def gtx_npu(mock_proc):
    """Real GtxNpu instance (function scope -- RegisterFile state leak guard).

    Triggers @isa.register('gtx') side effect on first import, so
    construction here also validates the C++ -> Python registration path.
    """
    # Import inside fixture so backend gate above runs FIRST (before any
    # xp.zeros allocation in riscv.gtx package init).
    from riscv.gtx.npu import GtxNpu

    npu = GtxNpu()
    npu.reset(mock_proc)
    return npu
