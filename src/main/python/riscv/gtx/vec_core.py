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
"""Pure stateless VEC kernels -- direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc.

Per CONTEXT D-01 / D-03: this module has NO `npu`/`proc`/`insn` dependencies.
Array-in / scalar-in / array-out / scalar-out only. P7 numba @njit boundary.

Per RESEARCH Pitfall 1 + 2: every reduction (VSUM/DOT) MUST upcast FP16 -> FP32
explicitly with Python `for` loop. NEVER `np.sum(x, dtype=np.float32)` or `np.dot`
(both use pairwise summation; ULP-different from C++ scalar accumulate).

Phase 5 plan 02 GREEN-fills the 7 kernels.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from .encoding import GTX_VEC_ADD, GTX_VEC_SUB, GTX_VEC_MUL, GTX_VEC_DIV


def sasmd_kernel(a: NDArray[np.float16], b,
                 op: int) -> NDArray[np.float16]:
    """SASMD element-wise: out[i] = a[i] OP b[i] for OP in {add,sub,mul,div}.

    `b` may be a scalar (broadcast) or an array. FP32 internal compute,
    single FP16 cast at writeback. Direct port of gtx_npu_vec.cc:50+
    exec_vector_op switch (and exec_vec_scalar:325-337 scalar-broadcast form).

    op ∈ {GTX_VEC_ADD=0, GTX_VEC_SUB=1, GTX_VEC_MUL=2, GTX_VEC_DIV=3}.
    """
    a_f32 = a.astype(np.float32)
    if hasattr(b, 'astype'):
        b_f32 = b.astype(np.float32)
    else:
        b_f32 = np.float32(b)
    if op == GTX_VEC_ADD:
        c_f32 = a_f32 + b_f32
    elif op == GTX_VEC_SUB:
        c_f32 = a_f32 - b_f32
    elif op == GTX_VEC_MUL:
        c_f32 = a_f32 * b_f32
    elif op == GTX_VEC_DIV:
        # gtx_npu_vec.cc:333 -- divide-by-zero produces 0.0 (HW convention).
        if hasattr(b_f32, 'shape') and b_f32.shape:
            c_f32 = np.where(b_f32 != 0.0, a_f32 / np.where(b_f32 != 0.0, b_f32, 1.0), 0.0)
        else:
            c_f32 = a_f32 / b_f32 if float(b_f32) != 0.0 else np.zeros_like(a_f32)
    else:
        raise ValueError(f"unknown SASMD op: {op}")
    return c_f32.astype(np.float16)


def dot_kernel(a: NDArray[np.float16], b: NDArray[np.float16]) -> np.float16:
    """FP16 dot product: explicit Python for-loop FP32 accumulate.

    Direct port of gtx_npu_vec.cc:251-262. NEVER np.dot/np.matmul/np.einsum
    (BLAS pairwise summation drifts vs C++ scalar order; RESEARCH Pitfall 2).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    s = np.float32(0.0)
    flat_a = a.ravel()
    flat_b = b.ravel()
    for i in range(flat_a.shape[0]):
        s += np.float32(flat_a[i]) * np.float32(flat_b[i])
    return np.float16(s)


def vsum_kernel(view: NDArray[np.float16]) -> np.float16:
    """FP16 vector sum: explicit Python for-loop FP32 accumulate + single FP16 cast.

    Direct port of gtx_npu_vec.cc:102-112. NEVER np.sum on FP16 -- pairwise
    summation drifts vs C++ scalar accumulate (RESEARCH Pitfall 2). NEVER
    `np.sum(view, dtype=np.float32)` either -- the dtype kwarg uses pairwise
    summation just in FP32, still ULP-different from C++.
    """
    s = np.float32(0.0)
    for x in view.ravel():
        s += np.float32(x)
    return np.float16(s)


def clamp_min_kernel(a: NDArray[np.float16],
                      scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = max(a[i], scalar) -- floor at lower bound.

    Direct port of gtx_npu_vec.cc:233-242 (GTX_VEC_CLAMP_MIN).
    """
    a_f32 = a.astype(np.float32)
    s_f32 = np.float32(scalar)
    return np.maximum(a_f32, s_f32).astype(np.float16)


def clamp_max_kernel(a: NDArray[np.float16],
                      scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = min(a[i], scalar) -- cap at upper bound.

    Direct port of gtx_npu_vec.cc:223-231 (GTX_VEC_CLAMP_MAX).
    """
    a_f32 = a.astype(np.float32)
    s_f32 = np.float32(scalar)
    return np.minimum(a_f32, s_f32).astype(np.float16)


def accum_kernel(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """Prefix sum (cumulative). FP32 accumulator across the whole vector;
    each writeback casts to FP16 but the accumulator is NOT reset between
    steps -- gtx_npu_vec.cc:215-221 line-for-line.
    """
    out = np.empty(a.shape[0], dtype=np.float16)
    s = np.float32(0.0)
    for i in range(a.shape[0]):
        s += np.float32(a[i])
        out[i] = np.float16(s)
    return out


def arange_kernel(n: int, start: np.float16,
                   step: np.float16) -> NDArray[np.float16]:
    """out[i] = start + i * step (FP32 internal).

    Direct port of gtx_npu_vec.cc:243-249. C++ uses `v += step` cumulative
    update; equivalent to `start + i*step` in FP32 (no rounding between
    accumulator updates because step is FP16 normalized to FP32).
    """
    out = np.empty(n, dtype=np.float16)
    s_f32 = np.float32(start)
    st_f32 = np.float32(step)
    v = s_f32
    for i in range(n):
        out[i] = np.float16(v)
        v += st_f32
    return out
