"""P5 VSUM/DOT precision unit tests -- Wave 1b plan 02 GREEN-fill (test_vsum_precision.py).

Covers VEC-02 dual-mode precision (CONTEXT D-09 / D-10):
  - Mode A (kernel): vec_core.vsum_kernel always FP32 internal accumulate.
  - Mode B (firmware): row-by-row VSUM N times + N FP16 partial sums re-summed.
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
from riscv.gtx import vec_core


def test_vsum_fp32_internal_anti_pattern():
    """VEC-02 (D-09 mode A): np.float16([1.0, 1e-4]*1000) summed via
    vec_core.vsum_kernel must equal the FP32-internal-accumulate result
    (~100.1), NOT the naive FP16 saturation result (~1000.0). RESEARCH
    locked the corrected expected value during ROADMAP synthesis."""
    arr = np.array([1.0, 1e-4] * 1000, dtype=np.float16)
    expected = np.float16(arr.astype(np.float32).sum(dtype=np.float32))
    actual = vec_core.vsum_kernel(arr)
    assert actual == expected
    # Confirm the FP16-only path WOULD fail this test:
    fp16_naive = arr.sum()  # NumPy default keeps FP16 dtype
    assert abs(float(fp16_naive) - 1000.0) < 1.0  # naive saturates near 1000
    assert abs(float(actual) - 100.1) < 0.2       # FP32-internal preserves precision


@pytest.mark.parametrize("rows", [2, 4, 8, 16])
def test_vsum_row_split_matches_cpp(rows):
    """VEC-02 (D-10 mode B): firmware-orchestrated row-split VSUM matches
    C++ ordering -- N row-by-row VSUM calls + N FP16 partial sums
    re-accumulated in FP32 + final FP16 cast.
    """
    np.random.seed(rows)
    row_len = 64
    flat = np.random.randn(rows * row_len).astype(np.float16)
    rows_2d = flat.reshape(rows, row_len)

    # Mode B: firmware splits, calls kernel per row, re-accumulates partial sums.
    partial_fp16 = np.array(
        [vec_core.vsum_kernel(rows_2d[r]) for r in range(rows)],
        dtype=np.float16,
    )
    s = np.float32(0.0)
    for x in partial_fp16:
        s += np.float32(x)
    actual = np.float16(s)

    # Oracle: same composition done in pure Python (no kernel) -- should match.
    expected_partials = np.array(
        [np.float16(rows_2d[r].astype(np.float32).sum(dtype=np.float32))
         for r in range(rows)],
        dtype=np.float16,
    )
    s_oracle = np.float32(0.0)
    for x in expected_partials:
        s_oracle += np.float32(x)
    expected = np.float16(s_oracle)
    assert actual == expected
