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
    """VEC-02 (D-09 mode A): vec_core.vsum_kernel must equal the
    FP32-internal-accumulate result, NOT a naive FP16-keeping accumulate.

    Anti-pattern proof uses a deliberately divergent input:
    `[1024.0] + 5000*[0.4]`. The FP16-naive path drops sub-ULP additions at
    the running 1024+ accumulator; the FP32-internal path preserves them.

    The 1e-4-input variant cited in the original ROADMAP success criterion
    happens to round to identical FP16 results both ways once you cast at
    the end (RESEARCH §VSUM Precision corrected during research).
    """
    # Divergent anti-pattern: FP16 naive vs FP32-internal differ before cast.
    arr = np.array([1024.0] + [0.4] * 5000, dtype=np.float16)
    expected = np.float16(arr.astype(np.float32).sum(dtype=np.float32))
    actual = vec_core.vsum_kernel(arr)
    # Kernel must match an FP32-internal oracle, not the naive FP16 result.
    assert actual == expected
    # Confirm the kernel ISN'T silently doing the naive-FP16 thing -- the
    # FP16 explicit cumulative add diverges from the FP32-internal expected.
    naive_fp16 = np.float16(0.0)
    for x in arr:
        naive_fp16 = np.float16(naive_fp16 + x)
    assert naive_fp16 != expected, (
        f"Anti-pattern broke: FP16-cumulative {naive_fp16} == FP32-internal "
        f"{expected}; pick a more divergent input."
    )

    # Overflow corner (D-12): vsum on >65504 worth of contributions casts to
    # FP16 inf via the FP32->FP16 single-cast at writeback.
    big = np.full(70000, 1.0, dtype=np.float16)
    assert np.isinf(vec_core.vsum_kernel(big))


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
