"""P5 pooling op unit tests -- Wave 0 RED scaffolds (test_pooling.py).

Covers ACT-03: max-pool + avg-pool with stride = kernel_size, output length
n_out = n_in / kernel_size, signed-zero canonicalization, always-forward
direction (ADDRA -> ADDRR per gtx_npu_act.cc:177-178).

Wave 1b plan 04 GREEN-fills these.
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_max_pool_output_length():
    """ACT-03: Max-pool output length = n_in / kernel_size (integer div).
    Source: gtx_npu_act.cc:195."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: pool_max with kernel-stride invariant")


def test_avg_pool_signed_zero_canon():
    """ACT-03: Avg-pool canonicalizes -0.0 -> +0.0 via `avg += 0.0` (line 211).
    Hex output is deterministic; -0.0 (0x8000) and +0.0 (0x0000) have
    different bit patterns; canon ensures golden-hex matching."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: pool_avg signed-zero canonicalization")


def test_pool_always_forward():
    """ACT-03: Pool ignores `is_reversed` -- always reads ADDRA, writes ADDRR.
    Source: gtx_npu_act.cc:177-178 (CONTEXT D-08)."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: pool always-forward direction")
