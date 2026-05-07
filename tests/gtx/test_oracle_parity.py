"""P5 VRF-02 oracle parity tests -- Wave 0 RED scaffolds (test_oracle_parity.py).

Covers VRF-02: 20 directly-mapped oracles from verify_ref.py:185-226 must
match the corresponding GTX op output bit-exact (np.array_equal on uint16
view, P4 D-15 pattern).

10 of 30 verify_ref ops are documented as DEFERRED in _oracles.py:
  - GELU_ERF (scipy ban), SIN/COS (not in C++ exec_vector_op),
    SILU/GELU_QUICK/ELU/SOFTPLUS/LEAKY_RELU/HARDSIGMOID/HARDSWISH (composed),
    FILL (P3 territory).

Wave 2 plan 05 GREEN-fills _oracles.py bodies + DIRECT_MAPPED_ORACLES dict
+ this parametrize body. Plan 01 ships single-entry placeholder per
RESEARCH §Validation Architecture (line 948 single parametrize covering 20 ops).
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


@pytest.mark.parametrize("name", ["abs"])
def test_oracle_parity(name):
    """VRF-02: Plan 05 wave 2 expands the parametrize list to 20 names and
    fills the body to compare GTX op output vs in-Python oracle bit-exact.

    Plan 01 ships single-entry placeholder so the file imports clean and the
    pytest collection lands a parametrized test ID.
    """
    pytest.skip(f"Wave 2 plan 05 GREEN-fills: oracle parity for {name!r}")
