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
"""Pure stateless GEMM kernel -- direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:27-94, 200-211, 262-265.

Per CONTEXT D-01 / D-03: This module has NO `npu`/`proc`/`insn` dependencies.
Array-in / scalar-in / array-out / scalar-out only. P7 numba `@njit` boundary.

Per RESEARCH np.matmul Bit-Exactness Analysis: explicit 3-loop FP32 accumulate
is REQUIRED for P4 strict-mode. `np.matmul` (BLAS) drifts up to 4 ULP / 0.0078 abs
on 41/500 random 16x16x16 FP16-cast-to-FP32 trials -- exceeds verify.py --ulp 1
--atol 0.001. P7 reactivates BLAS-equivalent perf via @njit.

Per PITFALLS Pitfall 2: every reduction MUST upcast FP16 -> FP32 for accumulation,
then single FP16 cast at the end. Never accumulate in FP16.

Phase 4 plan 02 Task 1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def gemm_core(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    *,
    has_bias: bool = False,
    bias_fp32: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float16]:
    """C = A @ B [+ bias_fp32]  -->  FP16, FP32-internal accumulate.

    Direct port of gtx_npu_mm.cc:27-94. Uses explicit Python 3-loop
    (NOT np.matmul) to guarantee bit-exact match against C++ scalar
    accumulate ordering.

    Args:
        A: FP16 (M, K)
        B: FP16 (K, N)
        has_bias: if True, add `bias_fp32` to FP32 accumulator before FP16 cast
        bias_fp32: FP32 (M, N) bias staged from L1 ADDRC region; required iff has_bias

    Returns:
        C: FP16 (M, N) result.

    NOTE: Scalar reductions (MM_O, MM_V) live in gemm_reduce_sum_a and gemm_dot --
    caller (mm_engine) selects the right kernel per variant. mxe_accum read/write
    is the caller's responsibility (D-06).
    """
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    M, K = A_f32.shape
    K2, N = B_f32.shape
    if K != K2:
        raise ValueError(
            f"shape mismatch: A is (M={M}, K={K}), B is (K={K2}, N={N})"
        )

    C_f32 = np.zeros((M, N), dtype=np.float32)
    # Explicit 3-loop. Bit-exact match with gtx_npu_mm.cc:73-79.
    # P7 numba @njit accelerates this back to BLAS-equivalent throughput.
    for i in range(M):
        for j in range(N):
            s = np.float32(0.0)
            for k in range(K):
                s += A_f32[i, k] * B_f32[k, j]
            C_f32[i, j] = s

    if has_bias:
        if bias_fp32 is None:
            raise ValueError("has_bias=True requires bias_fp32 ndarray")
        if bias_fp32.shape != (M, N):
            raise ValueError(
                f"bias_fp32 shape {bias_fp32.shape} != C shape ({M}, {N})"
            )
        if bias_fp32.dtype != np.float32:
            raise TypeError(
                f"bias_fp32 dtype must be float32, got {bias_fp32.dtype}"
            )
        C_f32 += bias_fp32

    return C_f32.astype(np.float16)


def gemm_reduce_sum_a(
    A: NDArray[np.float16],
    *,
    prior_accum: float = 0.0,
) -> float:
    """MM_O / MMC_O scalar: sum(A) + prior_accum, FP32 internal.

    Direct port of gtx_npu_mm.cc:200-211. Returns Python float (FP32 precision)
    for caller to cast to FP16 for L0 write AND store back into mxe_accum.

    Args:
        A: FP16 array of any shape (typically (col_A,))
        prior_accum: FP32 prior mxe_accum[nest, spu] if MMC_O; 0.0 if MM_O

    Returns:
        FP32 scalar (Python float) = sum(A_f32) + prior_accum
    """
    A_f32 = A.astype(np.float32)
    s = float(np.sum(A_f32, dtype=np.float32))
    return s + float(prior_accum)


def gemm_dot(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    *,
    prior_accum: float = 0.0,
) -> float:
    """MM_V / MMC_V scalar: dot(A, B) + prior_accum, FP32 internal.

    Direct port of gtx_npu_mm.cc:262-265. Returns Python float (FP32 precision).

    Uses explicit loop (NOT np.dot) -- np.dot may dispatch to BLAS for large
    vectors and drift like np.matmul (RESEARCH np.matmul Bit-Exactness).
    Vectors here are short so loop overhead is small.

    Args:
        A, B: FP16 1-D arrays of equal length
        prior_accum: FP32 prior mxe_accum[nest, spu] if MMC_V; 0.0 if MM_V

    Returns:
        FP32 scalar (Python float) = dot(A_f32, B_f32) + prior_accum
    """
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A {A.shape} vs B {B.shape}")
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    s = np.float32(0.0)
    for k in range(A_f32.shape[0]):
        s += A_f32[k] * B_f32[k]
    return float(s) + float(prior_accum)
