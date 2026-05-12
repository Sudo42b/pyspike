from __future__ import annotations

from typing import Optional


# ============================================================================
# SECTION B -- FP32-only `_impl` functions
# ============================================================================

def _gemm_core_impl(
    A_f32: NDArray[np.float32],
    B_f32: NDArray[np.float32],
    has_bias: bool,
    bias_fp32: NDArray[np.float32],
) -> NDArray[np.float32]:

    M, K = A_f32.shape
    K2, N = B_f32.shape
    if K != K2:
        raise ValueError(
            "shape mismatch: A is (M=" + str(M) + ", K=" + str(K) +
            "), B is (K=" + str(K2) + ", N=" + str(N) + ")"
        )

    C_f32 = np.zeros((M, N), dtype=np.float32)
    # Explicit 3-loop. Bit-exact match with gtx_npu_mm.cc:73-79.
    for i in range(M):
        for j in range(N):
            s = np.float32(0.0)
            for k in range(K):
                s += A_f32[i, k] * B_f32[k, j]
            C_f32[i, j] = s

    if has_bias:
        # Element-wise FP32 add (gtx_npu_mm.cc:84-91).
        for i in range(M):
            for j in range(N):
                C_f32[i, j] = C_f32[i, j] + bias_fp32[i, j]

    return C_f32


def _gemm_reduce_sum_a_impl(
    A_f32: NDArray[np.float32],
    prior_accum: np.float32,
) -> np.float32:
    """FP32-only flat sum + prior. Direct port of gtx_npu_mm.cc:200-211.

    Numba @njit boundary: FP32 in / FP32 out (np.float32 scalar).
    Explicit Python loop over flat A_f32 (NO np.sum -- pairwise reordering).
    """
    flat = A_f32.ravel()
    s = np.float32(0.0)
    for k in range(flat.shape[0]):
        s += flat[k]
    return s + prior_accum


def _gemm_dot_impl(
    A_f32: NDArray[np.float32],
    B_f32: NDArray[np.float32],
    prior_accum: np.float32,
) -> np.float32:
    """FP32-only dot + prior. Direct port of gtx_npu_mm.cc:262-265.

    Numba @njit boundary: FP32 in / FP32 out (np.float32 scalar).
    Explicit Python loop over flat A_f32, B_f32 (assumes equal shape).
    """
    A_flat = A_f32.ravel()
    B_flat = B_f32.ravel()
    s = np.float32(0.0)
    for k in range(A_flat.shape[0]):
        s += A_flat[k] * B_flat[k]
    return s + prior_accum


# ============================================================================
# SECTION C -- @njit(cache=True) wrappers (re-call pattern, CONTEXT D-11 Option B)
# ============================================================================

_gemm_core_njit = njit(cache=True)(_gemm_core_impl)
_gemm_reduce_sum_a_njit = njit(cache=True)(_gemm_reduce_sum_a_impl)
_gemm_dot_njit = njit(cache=True)(_gemm_dot_impl)


# ============================================================================
# SECTION D -- Public API (P4 signatures preserved verbatim; FP16 in/out)
# ============================================================================


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
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(
            f"shape mismatch: A is (M={M}, K={K}), B is (K={K2}, N={N})"
        )
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
        bias_arg = np.ascontiguousarray(bias_fp32, dtype=np.float32)
    else:
        bias_arg = np.zeros((M, N), dtype=np.float32)

    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    B_f32 = np.ascontiguousarray(B, dtype=np.float32)
    C_f32 = _gemm_core_njit(A_f32, B_f32, has_bias, bias_arg)
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
    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    s = _gemm_reduce_sum_a_njit(A_f32, np.float32(prior_accum))
    return float(s)


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
    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    B_f32 = np.ascontiguousarray(B, dtype=np.float32)
    s = _gemm_dot_njit(A_f32, B_f32, np.float32(prior_accum))
    return float(s)
