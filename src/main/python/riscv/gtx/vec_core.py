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

P7 NJIT-02 / NJIT-FP32-BOUNDARY (Plan 07-03):
    The 7 kernels are factored into FP32-only `_impl` functions wrapped by
    `@njit(cache=True)` (real numba when installed; passthrough no-op when
    absent per `_jit.py`). The public functions retain the P5 FP16 in/out
    signatures EXACTLY -- they pre-cast FP16 -> FP32 (numba CPU does NOT
    support np.float16; raises NotImplementedError at compile time, see
    RESEARCH §"FP16 NotImplementedError"), call the JIT'd `_impl`, and
    post-cast FP32 -> FP16 (or FP16 scalar for VSUM/DOT reductions).
    Callers in `vec_engine.py` see ZERO behavior change.

    Why `_impl` takes positional args (no kwonly, no Optional):
        - numba's lazy-typed dispatch does NOT support kwonly arguments.
        - For SASMD: `b` may be scalar or array in the public API. The
          `_impl` accepts ONLY ndarray `b_f32` -- the public wrapper
          broadcasts a scalar to a length-N array before the JIT call.
          (Mixing scalar+ndarray polymorphism would force two numba
          specializations per op-int; the broadcast keeps the JIT
          boundary mono-typed.)

    Bit-exactness vs P5 (verified by Tier 1 ULP-0 parity in
    tests/gtx/test_njit_parity.py):
        - SASMD: explicit per-element loop matches P5 broadcast (`+`/`-`/`*`)
          and div-by-zero=0.0 vendor convention (gtx_npu_vec.cc:333).
        - DOT / VSUM: P5 D-09 invariant preserved -- explicit Python `for`
          FP32 accumulate (NEVER np.sum, np.dot, np.einsum).
        - CLAMP_MIN/MAX: per-element max/min is bit-equivalent to
          np.maximum/np.minimum on FP32 inputs.
        - ACCUM: FP32 cumulative-sum array returned by `_impl`; per-element
          FP16 cast in wrapper. Equivalent to P5's per-iteration
          `out[i] = np.float16(s)` because the FP32 accumulator is unaffected
          by FP16 writebacks.
        - ARANGE: FP32 cumulative `v += step` returned; FP16 cast in wrapper.
          Equivalent to P5's per-iteration FP16 cast for the same reason.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from ._jit import njit, HAS_NUMBA  # noqa: F401  (HAS_NUMBA re-exposed for callers)
from .encoding import GTX_VEC_ADD, GTX_VEC_SUB, GTX_VEC_MUL, GTX_VEC_DIV


# ============================================================================
# SECTION B -- FP32-only `_impl` functions (numba JIT boundary)
# ============================================================================


def _sasmd_impl(
    a_f32: NDArray[np.float32],
    b_f32: NDArray[np.float32],
    op_int: int,
) -> NDArray[np.float32]:
    """SASMD element-wise FP32 op switch. Direct port of gtx_npu_vec.cc:50+.

    Numba @njit boundary: FP32 in / FP32 out, positional args only.
    `b_f32` MUST be an ndarray of the same shape as `a_f32` (caller broadcasts
    scalar to length-N before the JIT call).

    op ∈ {GTX_VEC_ADD=0, GTX_VEC_SUB=1, GTX_VEC_MUL=2, GTX_VEC_DIV=3}.
    DIV: gtx_npu_vec.cc:333 -- divide-by-zero produces 0.0 (HW convention).
    """
    n = a_f32.shape[0]
    c_f32 = np.empty(n, dtype=np.float32)
    if op_int == 0:  # GTX_VEC_ADD
        for i in range(n):
            c_f32[i] = a_f32[i] + b_f32[i]
    elif op_int == 1:  # GTX_VEC_SUB
        for i in range(n):
            c_f32[i] = a_f32[i] - b_f32[i]
    elif op_int == 2:  # GTX_VEC_MUL
        for i in range(n):
            c_f32[i] = a_f32[i] * b_f32[i]
    elif op_int == 3:  # GTX_VEC_DIV
        for i in range(n):
            if b_f32[i] == np.float32(0.0):
                c_f32[i] = np.float32(0.0)
            else:
                c_f32[i] = a_f32[i] / b_f32[i]
    else:
        raise ValueError("unknown SASMD op")
    return c_f32


def _dot_impl(
    a_f32: NDArray[np.float32],
    b_f32: NDArray[np.float32],
) -> np.float32:
    """FP32-only flat dot. Direct port of gtx_npu_vec.cc:251-262.

    Numba @njit boundary: FP32 in / FP32 out (np.float32 scalar).
    Explicit Python loop over flat a_f32, b_f32 (assumes equal shape).
    NEVER np.dot -- BLAS pairwise drift (RESEARCH Pitfall 2 / P5 D-09).
    """
    s = np.float32(0.0)
    for i in range(a_f32.shape[0]):
        s += a_f32[i] * b_f32[i]
    return s


def _vsum_impl(
    view_f32: NDArray[np.float32],
) -> np.float32:
    """FP32-only flat sum. Direct port of gtx_npu_vec.cc:102-112.

    Numba @njit boundary: FP32 in / FP32 out (np.float32 scalar).
    Explicit Python loop over flat view_f32. NEVER np.sum -- pairwise drift
    even with `dtype=np.float32` kwarg (RESEARCH Pitfall 2 / P5 D-09).
    """
    s = np.float32(0.0)
    for i in range(view_f32.shape[0]):
        s += view_f32[i]
    return s


def _clamp_min_impl(
    a_f32: NDArray[np.float32],
    scalar_f32: np.float32,
) -> NDArray[np.float32]:
    """FP32-only element-wise max(a, scalar). Direct port of gtx_npu_vec.cc:233-242.

    Numba @njit boundary: FP32 in / FP32 out.
    """
    n = a_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        if a_f32[i] > scalar_f32:
            out[i] = a_f32[i]
        else:
            out[i] = scalar_f32
    return out


def _clamp_max_impl(
    a_f32: NDArray[np.float32],
    scalar_f32: np.float32,
) -> NDArray[np.float32]:
    """FP32-only element-wise min(a, scalar). Direct port of gtx_npu_vec.cc:223-231.

    Numba @njit boundary: FP32 in / FP32 out.
    """
    n = a_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        if a_f32[i] < scalar_f32:
            out[i] = a_f32[i]
        else:
            out[i] = scalar_f32
    return out


def _accum_impl(
    a_f32: NDArray[np.float32],
) -> NDArray[np.float32]:
    """FP32-only cumulative prefix sum. Direct port of gtx_npu_vec.cc:215-221.

    Numba @njit boundary: FP32 in / FP32 out. Public wrapper handles per-element
    FP16 cast at writeback (bit-equivalent to P5's per-iteration `np.float16(s)`
    write because the FP32 accumulator is unaffected by FP16 cast).
    """
    n = a_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    s = np.float32(0.0)
    for i in range(n):
        s += a_f32[i]
        out[i] = s
    return out


def _arange_impl(
    n: int,
    start_f32: np.float32,
    step_f32: np.float32,
) -> NDArray[np.float32]:
    """FP32-only cumulative arange. Direct port of gtx_npu_vec.cc:243-249.

    Numba @njit boundary: int + FP32 in / FP32 out. Public wrapper handles
    per-element FP16 cast (bit-equivalent to P5's per-iteration FP16 cast --
    FP32 step accumulator is unaffected by FP16 writebacks).
    """
    out = np.empty(n, dtype=np.float32)
    v = start_f32
    for i in range(n):
        out[i] = v
        v += step_f32
    return out


# ============================================================================
# SECTION C -- @njit(cache=True) wrappers (re-call pattern, CONTEXT D-11 Option B)
# ============================================================================

_sasmd_njit       = njit(cache=True)(_sasmd_impl)
_dot_njit         = njit(cache=True)(_dot_impl)
_vsum_njit        = njit(cache=True)(_vsum_impl)
_clamp_min_njit   = njit(cache=True)(_clamp_min_impl)
_clamp_max_njit   = njit(cache=True)(_clamp_max_impl)
_accum_njit       = njit(cache=True)(_accum_impl)
_arange_njit      = njit(cache=True)(_arange_impl)


# ============================================================================
# SECTION D -- Public API (P5 signatures preserved verbatim; FP16 in/out)
# ============================================================================


def sasmd_kernel(a: NDArray[np.float16], b,
                 op: int) -> NDArray[np.float16]:
    """SASMD element-wise: out[i] = a[i] OP b[i] for OP in {add,sub,mul,div}.

    `b` may be a scalar (broadcast) or an array. FP32 internal compute,
    single FP16 cast at writeback. Direct port of gtx_npu_vec.cc:50+
    exec_vector_op switch (and exec_vec_scalar:325-337 scalar-broadcast form).

    op ∈ {GTX_VEC_ADD=0, GTX_VEC_SUB=1, GTX_VEC_MUL=2, GTX_VEC_DIV=3}.
    """
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    if hasattr(b, 'shape') and getattr(b, 'shape', ()):
        b_f32 = np.ascontiguousarray(b, dtype=np.float32)
    else:
        # Scalar-b broadcast lifted to wrapper to keep `_impl` mono-typed
        # (numba lazy dispatch would otherwise need a 2nd specialization).
        b_f32 = np.full(a_f32.shape, np.float32(b), dtype=np.float32)
    c_f32 = _sasmd_njit(a_f32, b_f32, int(op))
    return c_f32.astype(np.float16)


def dot_kernel(a: NDArray[np.float16], b: NDArray[np.float16]) -> np.float16:
    """FP16 dot product: explicit Python for-loop FP32 accumulate.

    Direct port of gtx_npu_vec.cc:251-262. NEVER np.dot/np.matmul/np.einsum
    (BLAS pairwise summation drifts vs C++ scalar order; RESEARCH Pitfall 2).
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_f32 = np.ascontiguousarray(a.ravel(), dtype=np.float32)
    b_f32 = np.ascontiguousarray(b.ravel(), dtype=np.float32)
    s = _dot_njit(a_f32, b_f32)
    return np.float16(s)


def vsum_kernel(view: NDArray[np.float16]) -> np.float16:
    """FP16 vector sum: explicit Python for-loop FP32 accumulate + single FP16 cast.

    Direct port of gtx_npu_vec.cc:102-112. NEVER np.sum on FP16 -- pairwise
    summation drifts vs C++ scalar accumulate (RESEARCH Pitfall 2). NEVER
    `np.sum(view, dtype=np.float32)` either -- the dtype kwarg uses pairwise
    summation just in FP32, still ULP-different from C++.
    """
    flat = np.ascontiguousarray(view.ravel(), dtype=np.float32)
    s = _vsum_njit(flat)
    return np.float16(s)


def clamp_min_kernel(a: NDArray[np.float16],
                      scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = max(a[i], scalar) -- floor at lower bound.

    Direct port of gtx_npu_vec.cc:233-242 (GTX_VEC_CLAMP_MIN).
    """
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    s_f32 = np.float32(scalar)
    out_f32 = _clamp_min_njit(a_f32, s_f32)
    return out_f32.astype(np.float16)


def clamp_max_kernel(a: NDArray[np.float16],
                      scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = min(a[i], scalar) -- cap at upper bound.

    Direct port of gtx_npu_vec.cc:223-231 (GTX_VEC_CLAMP_MAX).
    """
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    s_f32 = np.float32(scalar)
    out_f32 = _clamp_max_njit(a_f32, s_f32)
    return out_f32.astype(np.float16)


def accum_kernel(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """Prefix sum (cumulative). FP32 accumulator across the whole vector;
    each writeback casts to FP16 but the accumulator is NOT reset between
    steps -- gtx_npu_vec.cc:215-221 line-for-line.
    """
    a_f32 = np.ascontiguousarray(a, dtype=np.float32)
    out_f32 = _accum_njit(a_f32)
    return out_f32.astype(np.float16)


def arange_kernel(n: int, start: np.float16,
                   step: np.float16) -> NDArray[np.float16]:
    """out[i] = start + i * step (FP32 internal).

    Direct port of gtx_npu_vec.cc:243-249. C++ uses `v += step` cumulative
    update; equivalent to `start + i*step` in FP32 (no rounding between
    accumulator updates because step is FP16 normalized to FP32).
    """
    out_f32 = _arange_njit(int(n), np.float32(start), np.float32(step))
    return out_f32.astype(np.float16)
