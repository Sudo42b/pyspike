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

# Plan 03 GREEN: 7 vec kernels
VEC_KERNELS = {
    "sasmd_kernel", "dot_kernel", "vsum_kernel",
    "clamp_min_kernel", "clamp_max_kernel",
    "accum_kernel", "arange_kernel",
}


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


def _generate_vec_inputs(kernel_name: str, seed: int = 42):
    """Fixed-seed FP16 inputs for vec parity. Length-16 vectors / scalars.

    W3 coverage note: SASMD Tier-1 covers the array-b path only. The
    scalar-b broadcast path (used by IS variants in vec_engine) is
    exercised by Tier 2 vendor sweep in Plan 07-05; documented in
    07-03-SUMMARY.md.
    """
    rng = np.random.default_rng((seed + hash(kernel_name)) & 0xFFFFFFFF)
    if kernel_name == "sasmd_kernel":
        from riscv.gtx.encoding import GTX_VEC_ADD
        a = rng.random(16, dtype=np.float32).astype(np.float16)
        b = rng.random(16, dtype=np.float32).astype(np.float16)
        return ("sasmd", a, b, GTX_VEC_ADD)
    if kernel_name == "dot_kernel":
        a = rng.random(16, dtype=np.float32).astype(np.float16)
        b = rng.random(16, dtype=np.float32).astype(np.float16)
        return ("dot", a, b)
    if kernel_name == "vsum_kernel":
        view = rng.random(16, dtype=np.float32).astype(np.float16)
        return ("vsum", view)
    if kernel_name in {"clamp_min_kernel", "clamp_max_kernel"}:
        # range [-2, 2] so the scalar=0.5 actually clips on both sides
        a = (rng.random(16, dtype=np.float32) * 4 - 2).astype(np.float16)
        scalar = np.float16(0.5)
        return ("clamp", a, scalar)
    if kernel_name == "accum_kernel":
        # small-magnitude inputs keep the FP32 cumulative accumulator inside
        # FP16 representable range so the per-step FP16 cast is bit-exact.
        a = (rng.random(16, dtype=np.float32) * 0.1).astype(np.float16)
        return ("accum", a)
    if kernel_name == "arange_kernel":
        return ("arange", 16, np.float16(1.0), np.float16(0.5))
    raise KeyError(kernel_name)


def _run_vec_parity(kernel_name: str) -> None:
    """ULP-0 parity for one vec kernel (numpy oracle vs njit-compiled _impl)."""
    from riscv.gtx.vec_core import (
        _sasmd_njit, _dot_njit, _vsum_njit,
        _clamp_min_njit, _clamp_max_njit,
        _accum_njit, _arange_njit,
    )

    inputs = _generate_vec_inputs(kernel_name)
    kind = inputs[0]

    if kind == "sasmd":
        _, a, b, op = inputs
        numpy_out = get_public_fn(kernel_name)(a, b, op)  # FP16 array
        a_f32 = np.ascontiguousarray(a, dtype=np.float32)
        b_f32 = np.ascontiguousarray(b, dtype=np.float32)  # array, no broadcast
        njit_out = _sasmd_njit(a_f32, b_f32, int(op)).astype(np.float16)
        assert np.array_equal(
            numpy_out.view(np.uint16), njit_out.view(np.uint16)
        ), f"{kernel_name}: ULP-0 parity failed (sasmd)"

    elif kind == "dot":
        _, a, b = inputs
        numpy_out = get_public_fn(kernel_name)(a, b)  # np.float16 scalar
        a_f32 = np.ascontiguousarray(a.ravel(), dtype=np.float32)
        b_f32 = np.ascontiguousarray(b.ravel(), dtype=np.float32)
        njit_scalar = np.float16(_dot_njit(a_f32, b_f32))
        assert np.array_equal(
            np.array([numpy_out]).view(np.uint16),
            np.array([njit_scalar]).view(np.uint16),
        ), f"{kernel_name}: ULP-0 parity failed (dot); numpy={numpy_out}, njit={njit_scalar}"

    elif kind == "vsum":
        _, view = inputs
        numpy_out = get_public_fn(kernel_name)(view)
        f32 = np.ascontiguousarray(view.ravel(), dtype=np.float32)
        njit_scalar = np.float16(_vsum_njit(f32))
        assert np.array_equal(
            np.array([numpy_out]).view(np.uint16),
            np.array([njit_scalar]).view(np.uint16),
        ), f"{kernel_name}: ULP-0 parity failed (vsum); numpy={numpy_out}, njit={njit_scalar}"

    elif kind == "clamp":
        _, a, scalar = inputs
        numpy_out = get_public_fn(kernel_name)(a, scalar)
        a_f32 = np.ascontiguousarray(a, dtype=np.float32)
        scalar_f32 = np.float32(scalar)
        njit_call = _clamp_min_njit if kernel_name == "clamp_min_kernel" else _clamp_max_njit
        njit_out = njit_call(a_f32, scalar_f32).astype(np.float16)
        assert np.array_equal(
            numpy_out.view(np.uint16), njit_out.view(np.uint16)
        ), f"{kernel_name}: ULP-0 parity failed (clamp)"

    elif kind == "accum":
        _, a = inputs
        numpy_out = get_public_fn(kernel_name)(a)
        a_f32 = np.ascontiguousarray(a, dtype=np.float32)
        njit_out = _accum_njit(a_f32).astype(np.float16)
        assert np.array_equal(
            numpy_out.view(np.uint16), njit_out.view(np.uint16)
        ), f"{kernel_name}: ULP-0 parity failed (accum)"

    elif kind == "arange":
        _, n, start, step = inputs
        numpy_out = get_public_fn(kernel_name)(n, start, step)
        njit_out = _arange_njit(int(n), np.float32(start), np.float32(step)).astype(np.float16)
        assert np.array_equal(
            numpy_out.view(np.uint16), njit_out.view(np.uint16)
        ), f"{kernel_name}: ULP-0 parity failed (arange)"

    else:
        raise AssertionError(f"unhandled kind {kind} for {kernel_name}")


@pytest.mark.parametrize("kernel_name", ALL_NJIT_KERNEL_NAMES)
def test_kernel_parity(kernel_name: str) -> None:
    """ULP-0 parity: NumPy public API vs JIT-compiled _impl."""
    if kernel_name in GEMM_KERNELS:
        _run_gemm_parity(kernel_name)
    elif kernel_name in VEC_KERNELS:
        _run_vec_parity(kernel_name)
    else:
        pytest.skip(f"Plan 04 GREEN-fills parity body for {kernel_name}")


def test_has_numba_detection(_numba_available: bool) -> None:
    """NJIT-01 sentinel: HAS_NUMBA reflects environment numba presence."""
    from riscv.gtx._jit import HAS_NUMBA
    assert HAS_NUMBA == _numba_available, (
        f"HAS_NUMBA={HAS_NUMBA} but _numba_available={_numba_available}"
    )
