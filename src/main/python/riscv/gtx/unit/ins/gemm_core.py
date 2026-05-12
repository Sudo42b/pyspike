"""GEMM kernels — PyTorch-only, FP32 internal accumulate.

Direct port of ``gtx_npu_mm.cc`` reductions; uses :func:`torch.matmul`
for the main matmul (vectorised, GPU-friendly) and :func:`torch.sum` /
:func:`torch.dot` for the scalar variants. FP32 accumulate keeps ULP
match with the C++ reference within the project's tolerance (see
``_verify.py`` removed; tolerance is checked at the .elf regression
layer).

Public API
    gemm_core(A, B, has_bias=False, bias_fp32=None) -> Tensor (FP16)
    gemm_reduce_sum_a(A, prior_accum=0.0)            -> float
    gemm_dot(A, B, prior_accum=0.0)                   -> float
"""
from __future__ import annotations

from typing import Optional

import torch


def _as_f32(x: torch.Tensor) -> torch.Tensor:
    """Return a contiguous FP32 view (cast if needed) for accumulation."""
    if x.dtype is torch.float32:
        return x.contiguous()
    return x.to(torch.float32).contiguous()


def gemm_core(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    has_bias: bool = False,
    bias_fp32: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``C = A @ B [+ bias_fp32]`` — FP16 result with FP32 accumulate.

    Args
        A: FP16 ``(M, K)`` tensor.
        B: FP16 ``(K, N)`` tensor.
        has_bias: when True, add ``bias_fp32`` to the FP32 accumulator
            before downcasting to FP16.
        bias_fp32: FP32 ``(M, N)`` bias staged from L1 ADDRC; required iff
            ``has_bias``.

    Returns
        FP16 ``(M, N)`` result.

    Scalar reductions (MM_O / MM_V) live in :func:`gemm_reduce_sum_a` and
    :func:`gemm_dot`; the caller picks the right kernel per variant and
    is responsible for any ``mxe_accum`` read/write.
    """
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"shape mismatch: A is (M={M}, K={K}), B is (K={K2}, N={N})")

    A_f32 = _as_f32(A)
    B_f32 = _as_f32(B)
    C_f32 = torch.matmul(A_f32, B_f32)

    if has_bias:
        if bias_fp32 is None:
            raise ValueError("has_bias=True requires bias_fp32 tensor")
        if tuple(bias_fp32.shape) != (M, N):
            raise ValueError(
                f"bias_fp32 shape {tuple(bias_fp32.shape)} != C shape ({M}, {N})"
            )
        if bias_fp32.dtype is not torch.float32:
            raise TypeError(
                f"bias_fp32 dtype must be float32, got {bias_fp32.dtype}"
            )
        C_f32 = C_f32 + bias_fp32

    return C_f32.to(torch.float16)


def gemm_reduce_sum_a(
    A: torch.Tensor,
    *,
    prior_accum: float = 0.0,
) -> float:
    """``MM_O`` / ``MMC_O`` scalar: ``sum(A) + prior_accum`` with FP32 reduce.

    Direct port of ``gtx_npu_mm.cc:200-211``. Returns a Python float so the
    caller can both cast to FP16 for the L0 write and store back into
    ``mxe_accum`` as FP32.

    Args
        A: FP16 tensor of any shape (typically ``(col_A,)``).
        prior_accum: FP32 prior ``mxe_accum[nest, spu]`` (MMC_O); 0.0 for MM_O.

    Returns
        FP32 scalar (Python float) = ``sum(A_f32) + prior_accum``.
    """
    A_f32 = _as_f32(A)
    s = torch.sum(A_f32) + torch.tensor(prior_accum, dtype=torch.float32,
                                        device=A_f32.device)
    return float(s.item())


def gemm_dot(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    prior_accum: float = 0.0,
) -> float:
    """``MM_V`` / ``MMC_V`` scalar: ``dot(A, B) + prior_accum`` with FP32 reduce.

    Args
        A, B: FP16 1-D tensors of equal length.
        prior_accum: FP32 prior ``mxe_accum[nest, spu]`` (MMC_V); 0.0 for MM_V.

    Returns
        FP32 scalar (Python float).
    """
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A {tuple(A.shape)} vs B {tuple(B.shape)}")
    A_f32 = _as_f32(A).flatten()
    B_f32 = _as_f32(B).flatten()
    s = torch.dot(A_f32, B_f32) + torch.tensor(prior_accum, dtype=torch.float32,
                                                device=A_f32.device)
    return float(s.item())
