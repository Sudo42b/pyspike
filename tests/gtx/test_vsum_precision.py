"""P5 VSUM/DOT precision unit tests -- Wave 0 RED scaffolds (test_vsum_precision.py).

Covers VEC-02 dual-mode precision (CONTEXT D-09 / D-10):
  - Mode A (kernel): vec_core.vsum_kernel always FP32 internal accumulate.
  - Mode B (firmware): row-by-row VSUM N times + N FP16 partial sums re-summed.

Wave 1b plan 02 GREEN-fills these. Plan 01 ships pytest.skip(...) bodies.
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_vsum_fp32_internal_anti_pattern():
    """VEC-02 (D-09 mode A): np.float16([1.0, 1e-4]*1000) summed via
    vec_core.vsum_kernel must equal the FP32-internal-accumulate result
    (~100.1), NOT the naive FP16 saturation result (~1000.0). RESEARCH
    locked the corrected expected value during ROADMAP synthesis."""
    pytest.skip("Wave 1b plan 02 GREEN-fills: VSUM FP32 internal anti-pattern")


@pytest.mark.parametrize("rows", [2, 4, 8, 16])
def test_vsum_row_split_matches_cpp(rows):
    """VEC-02 (D-10 mode B): firmware row-split VSUM matches C++ ordering --
    N row-by-row VSUM calls + N FP16 partial sums re-accumulated in FP32 +
    final FP16 cast. Plan 05 wave 2 synthesizes the golden in-Python."""
    pytest.skip(f"Wave 1b plan 02 GREEN-fills: VSUM mode-B row-split (rows={rows})")
