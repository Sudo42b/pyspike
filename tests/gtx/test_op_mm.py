"""P4 MM op unit tests -- Wave 0 scaffolds (test_op_mm.py).

Covers MM-01, MM-02, MM-03 (decode), MM-05 (verify_minimal unit).
Wave 1 plans (gemm_core / mm_engine / ops/mm) GREEN-fill these.
"""
import os
import pathlib

import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


def test_gemm_core_explicit_3loop_matches_oracle():
    """MM-01: gemm_core uses explicit Python 3-loop FP32 accumulate (NOT np.matmul)
    per RESEARCH np.matmul Bit-Exactness section."""
    from riscv.gtx.gemm_core import gemm_core
    np.random.seed(42)
    A = np.random.randn(16, 16).astype(np.float16)
    B = np.random.randn(16, 16).astype(np.float16)
    actual = gemm_core(A, B)

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    expected_f32 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        for j in range(16):
            s = np.float32(0.0)
            for k in range(16):
                s += A_f32[i, k] * B_f32[k, j]
            expected_f32[i, j] = s
    expected = expected_f32.astype(np.float16)

    assert actual.dtype == np.float16
    assert actual.shape == (16, 16)
    # Bit-exact compare via uint16 view (D-15)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_gemm_core_fp32_internal_not_fp16():
    """MM-01 / Pitfall 2: regression -- np.float16([1.0, 1e-4]*1000).sum() must NOT inf."""
    from riscv.gtx.gemm_core import gemm_reduce_sum_a
    # Pitfall 2 textbook case: long vector, mixed magnitudes.
    # FP16-internal accumulate would inf or saturate; FP32 stays finite.
    arr = np.array([1.0, 1e-4] * 1000, dtype=np.float16)
    result = gemm_reduce_sum_a(arr)
    assert np.isfinite(result), \
        f"gemm_reduce_sum_a should accumulate in FP32, got {result}"
    # Expected ~1000 + 1000*1e-4 ~= 1000.1; FP16 input loses precision on 1e-4.
    assert 999.0 < result < 1001.0, f"sum out of expected range: {result}"
    assert isinstance(result, float), \
        f"gemm_reduce_sum_a must return Python float, got {type(result)}"


def test_handler_registry_has_all_10_mm_variants():
    """MM-02: 10 @handler entries (mm/mm_s/mm_o/mm_v/mm_t + mmc.* family)."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_exec_mm_basic_bit_exact():
    """MM-02: 16x16x16 mm bit-exact vs explicit 3-loop oracle."""
    pytest.skip("Wave 1: ops/mm + gemm_core not yet built (Plans 02+04)")


def test_exec_mm_s_writes_fp32_to_addrc():
    """MM-02: mm_s writes FP32 result bytes to ADDRC (LSPR_SPM_ADDRC=0x902) staging."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_exec_mm_o_writes_scalar_to_l0_be():
    """MM-02: mm_o writes scalar sum(A) to L0 in BIG-endian (gtx_npu_mm.cc:217-218)."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_exec_mm_v_writes_dot_to_l0_le():
    """MM-02: mm_v writes scalar dot(A,B) to L0 in little-endian
    (gtx_npu_mm.cc:274-275 -- asymmetry vs mm_o!)."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_exec_mm_t_writes_transposed():
    """MM-02 / Pitfall D: mm_t writes C^T to ADDRR in NxM layout (NOT MxN)."""
    pytest.skip("Wave 1: ops/mm not yet built (Plan 04)")


def test_decode_firmware_mm_args():
    """MM-03: rs1 packed = colB[63:48]|colA[31:16]|rowA[15:0],
    0=65536 per field (dim16 lambda)."""
    pytest.skip("Wave 1: mm_engine not yet built (Plan 03)")


def test_verify_minimal_be_fp16_pairs():
    """MM-05 / Pitfall 1: _verify_minimal.compare_hex uses BE FP16 bit-pair (verify.py:235)."""
    # _verify_minimal landed in Task 1; smoke-test that the function exists.
    from tests.gtx._verify_minimal import compare_hex
    assert callable(compare_hex)
    pytest.skip("Wave 1: full BE-pair regression assertion (Plan 02 or 05)")


def test_gemm_core_signature_stateless():
    """MM-01 / D-03: gemm_core is array-in/array-out, no npu/proc/insn dependency."""
    import inspect
    from riscv.gtx import gemm_core as gemm_core_mod
    sig = inspect.signature(gemm_core_mod.gemm_core)
    params = list(sig.parameters.keys())
    forbidden = {'npu', 'proc', 'insn', 'self', 'mem', 'memory'}
    for p in params:
        assert p not in forbidden, \
            f"gemm_core must be stateless (D-03), but found '{p}' parameter"
    src = inspect.getsource(gemm_core_mod)
    assert 'from .npu' not in src, "gemm_core must not depend on npu (leaf module)"
    assert 'from .memory' not in src, "gemm_core must not depend on memory (leaf module)"
    assert 'from .dispatch' not in src, "gemm_core must not depend on dispatch"
