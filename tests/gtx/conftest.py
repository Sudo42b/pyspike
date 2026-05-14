"""Pytest fixtures for tests/gtx/* -- ORDER.md FSM smoke set.

Invariants enforced here:
  - CUDA is REQUIRED (ORDER.md: "DDR은 CPU, 나머지 메모리 계층은 반드시 cuda").
    Collection fails fast with pytest.exit(returncode=1) if torch.cuda is
    unavailable -- never a skip, because skipping would mask design-rule
    violations.
  - GTX_DDR_LOAD / GTX_DDR_DUMP env vars are cleared so vendor binary I/O
    never fires during tests.
"""
from __future__ import annotations
import os

import pytest
import torch

# CUDA gate -- collection-time, per CONTEXT.md decision D-CUDA-REQUIRED.
if not torch.cuda.is_available():
    pytest.exit(
        "GTX tests require CUDA -- ORDER.md constraint", returncode=1
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
    # Import inside fixture so CUDA gate above runs FIRST (before any
    # torch tensor allocation in riscv.gtx package init).
    from riscv.gtx.npu import GtxNpu

    npu = GtxNpu()
    npu.reset(mock_proc)
    return npu
