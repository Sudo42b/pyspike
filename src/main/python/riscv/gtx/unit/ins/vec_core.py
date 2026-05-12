"""Pure stateless VEC kernels (torch-backed).

Migrated 2026-05-11 from numpy/numba → torch. CPU tensors throughout; CUDA
move is opt-in (callers may .to(device) on inputs).

Per RESEARCH Pitfall 1 + 2: every reduction (VSUM/DOT) MUST upcast FP16 -> FP32
explicitly. C++ does scalar-order accumulate; torch.sum on FP16 uses pairwise
which drifts ULP-different. We emulate scalar order via a Python loop where
bit-exactness vs vendor matters (DOT, VSUM).

Public API (signatures preserved from P5 / numpy era):
  sasmd_kernel(a, b, op) -> torch.Tensor[float16]
  dot_kernel(a, b)       -> torch.Tensor[float16]   (0-d scalar)
  vsum_kernel(view)      -> torch.Tensor[float16]   (0-d scalar)
  clamp_min_kernel(a, s) -> torch.Tensor[float16]
  clamp_max_kernel(a, s) -> torch.Tensor[float16]
  accum_kernel(a)        -> torch.Tensor[float16]
  arange_kernel(n, s, st)-> torch.Tensor[float16]
"""
from __future__ import annotations

import torch

from .encoding import GTX_VEC_ADD, GTX_VEC_SUB, GTX_VEC_MUL, GTX_VEC_DIV

# HAS_NUMBA retained for back-compat with tests that reference the symbol;
# pyspike no longer depends on numba.
HAS_NUMBA: bool = False


def _as_fp32(a) -> torch.Tensor:
    """Convert ndarray/tensor/scalar to FP32 torch tensor (zero-copy when possible)."""
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32)
    # Numpy or scalar
    return torch.as_tensor(a, dtype=torch.float32)


def sasmd_kernel(a, b, op: int) -> torch.Tensor:
    """SASMD element-wise FP32 internal, FP16 output. `b` scalar or array."""
    a_f32 = _as_fp32(a)
    if isinstance(b, torch.Tensor) and b.dim() > 0:
        b_f32 = b.to(torch.float32)
    elif hasattr(b, 'shape') and getattr(b, 'shape', ()):
        b_f32 = torch.as_tensor(b, dtype=torch.float32)
    else:
        b_f32 = torch.full_like(a_f32, float(b))
    if op == GTX_VEC_ADD:
        out = a_f32 + b_f32
    elif op == GTX_VEC_SUB:
        out = a_f32 - b_f32
    elif op == GTX_VEC_MUL:
        out = a_f32 * b_f32
    elif op == GTX_VEC_DIV:
        # Vendor convention (gtx_npu_vec.cc:333): div-by-zero -> 0.0.
        safe_b = torch.where(b_f32 == 0.0, torch.ones_like(b_f32), b_f32)
        raw = a_f32 / safe_b
        out = torch.where(b_f32 == 0.0, torch.zeros_like(raw), raw)
    else:
        raise ValueError(f"unknown SASMD op {op}")
    return out.to(torch.float16)


def dot_kernel(a, b) -> torch.Tensor:
    """FP16 dot product (FP32 scalar accumulator, scalar-order)."""
    a_f32 = _as_fp32(a).reshape(-1)
    b_f32 = _as_fp32(b).reshape(-1)
    if a_f32.shape != b_f32.shape:
        raise ValueError(f"shape mismatch: {a_f32.shape} vs {b_f32.shape}")
    # Explicit scalar-order to match C++ accumulate (RESEARCH Pitfall 2).
    s = torch.tensor(0.0, dtype=torch.float32, device=a_f32.device)
    n = a_f32.shape[0]
    for i in range(n):
        s = s + a_f32[i] * b_f32[i]
    return s.to(torch.float16)


def vsum_kernel(view) -> torch.Tensor:
    """FP16 vector sum (FP32 scalar accumulator, scalar-order)."""
    flat = _as_fp32(view).reshape(-1)
    s = torch.tensor(0.0, dtype=torch.float32, device=flat.device)
    for i in range(flat.shape[0]):
        s = s + flat[i]
    return s.to(torch.float16)


def clamp_min_kernel(a, scalar) -> torch.Tensor:
    """out[i] = max(a[i], scalar)."""
    a_f32 = _as_fp32(a)
    return torch.clamp(a_f32, min=float(scalar)).to(torch.float16)


def clamp_max_kernel(a, scalar) -> torch.Tensor:
    """out[i] = min(a[i], scalar)."""
    a_f32 = _as_fp32(a)
    return torch.clamp(a_f32, max=float(scalar)).to(torch.float16)


def accum_kernel(a) -> torch.Tensor:
    """Prefix sum: FP32 accumulator across whole vec, per-element FP16 cast."""
    a_f32 = _as_fp32(a).reshape(-1)
    out = torch.empty_like(a_f32)
    s = torch.tensor(0.0, dtype=torch.float32, device=a_f32.device)
    for i in range(a_f32.shape[0]):
        s = s + a_f32[i]
        out[i] = s
    return out.to(torch.float16)


def arange_kernel(n: int, start, step) -> torch.Tensor:
    """out[i] = start + i*step (FP32 internal)."""
    from ...config_params import DEVICE
    s_f32 = float(start)
    st_f32 = float(step)
    # torch.arange with float arithmetic; matches start + i*step in FP32.
    idx = torch.arange(int(n), dtype=torch.float32, device=DEVICE)
    out = s_f32 + idx * st_f32
    return out.to(torch.float16)
