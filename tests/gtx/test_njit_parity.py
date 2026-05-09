#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""P7 NJIT-05 Tier 1: per-kernel ULP-0 parity (NumPy oracle vs JIT).

Plan 02 GREEN-fills gemm_core / gemm_reduce_sum_a / gemm_dot.
Plans 03/04 GREEN-fill the rest. Wave 0 default body = pytest.skip.
"""
from __future__ import annotations
import numpy as np
import pytest

from ._njit_helpers import (
    ALL_NJIT_KERNEL_NAMES,
    get_public_fn,
)

# Plan 02 GREEN: 3 gemm kernels
GEMM_KERNELS = {"gemm_core", "gemm_reduce_sum_a", "gemm_dot"}


def _generate_gemm_inputs(kernel_name: str, seed: int = 42):
    """Fixed-seed FP16 inputs for gemm parity. 16x16 matrices / length-16 vectors."""
    rng = np.random.default_rng((seed + hash(kernel_name)) & 0xFFFFFFFF)
    if kernel_name == "gemm_core":
        A = rng.random((16, 16), dtype=np.float32).astype(np.float16)
        B = rng.random((16, 16), dtype=np.float32).astype(np.float16)
        return ("matrix", A, B)
    if kernel_name == "gemm_reduce_sum_a":
        A = rng.random(16, dtype=np.float32).astype(np.float16)
        return ("reduce", A)
    if kernel_name == "gemm_dot":
        A = rng.random(16, dtype=np.float32).astype(np.float16)
        B = rng.random(16, dtype=np.float32).astype(np.float16)
        return ("dot", A, B)
    raise KeyError(kernel_name)


def _run_gemm_parity(kernel_name: str) -> None:
    """ULP-0 parity for one gemm kernel (numpy oracle vs njit-compiled _impl)."""
    from riscv.gtx.gemm_core import (
        _gemm_core_njit, _gemm_reduce_sum_a_njit, _gemm_dot_njit,
    )

    inputs = _generate_gemm_inputs(kernel_name)
    kind = inputs[0]

    if kind == "matrix":
        _, A, B = inputs
        # Numpy oracle = current public API
        numpy_out = get_public_fn(kernel_name)(A, B)  # FP16
        # JIT path = direct _njit call with FP32 inputs + FP16 cast
        A_f32 = np.ascontiguousarray(A, dtype=np.float32)
        B_f32 = np.ascontiguousarray(B, dtype=np.float32)
        zero_bias = np.zeros(numpy_out.shape, dtype=np.float32)
        njit_out_f32 = _gemm_core_njit(A_f32, B_f32, False, zero_bias)
        njit_out = njit_out_f32.astype(np.float16)
        assert np.array_equal(
            numpy_out.view(np.uint16), njit_out.view(np.uint16)
        ), f"{kernel_name}: ULP-0 parity failed (matrix kind)"

    elif kind == "reduce":
        _, A = inputs
        # Numpy oracle = current public API (returns Python float)
        numpy_out = float(get_public_fn(kernel_name)(A))
        A_f32 = np.ascontiguousarray(A, dtype=np.float32)
        njit_out = float(_gemm_reduce_sum_a_njit(A_f32, np.float32(0.0)))
        # FP16 byte compare on the (cast) result so floating-point identity
        # is verified at storage precision (P4 prior_accum returns FP32 scalar
        # but caller always casts to FP16 for L0 writeback).
        numpy_f16 = np.float16(numpy_out)
        njit_f16 = np.float16(njit_out)
        assert np.array_equal(
            np.array([numpy_f16]).view(np.uint16),
            np.array([njit_f16]).view(np.uint16),
        ), (
            f"{kernel_name}: ULP-0 parity failed (reduce); "
            f"numpy={numpy_out}, njit={njit_out}"
        )

    elif kind == "dot":
        _, A, B = inputs
        numpy_out = float(get_public_fn(kernel_name)(A, B))
        A_f32 = np.ascontiguousarray(A, dtype=np.float32)
        B_f32 = np.ascontiguousarray(B, dtype=np.float32)
        njit_out = float(_gemm_dot_njit(A_f32, B_f32, np.float32(0.0)))
        numpy_f16 = np.float16(numpy_out)
        njit_f16 = np.float16(njit_out)
        assert np.array_equal(
            np.array([numpy_f16]).view(np.uint16),
            np.array([njit_f16]).view(np.uint16),
        ), (
            f"{kernel_name}: ULP-0 parity failed (dot); "
            f"numpy={numpy_out}, njit={njit_out}"
        )


@pytest.mark.parametrize("kernel_name", ALL_NJIT_KERNEL_NAMES)
def test_kernel_parity(kernel_name: str) -> None:
    """ULP-0 parity: NumPy public API vs JIT-compiled _impl."""
    if kernel_name in GEMM_KERNELS:
        _run_gemm_parity(kernel_name)
    else:
        pytest.skip(f"Plan 03/04 GREEN-fills parity body for {kernel_name}")


def test_has_numba_detection(_numba_available: bool) -> None:
    """NJIT-01 sentinel: HAS_NUMBA reflects environment numba presence."""
    from riscv.gtx._jit import HAS_NUMBA
    assert HAS_NUMBA == _numba_available, (
        f"HAS_NUMBA={HAS_NUMBA} but _numba_available={_numba_available}"
    )
