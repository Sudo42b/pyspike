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
"""Pure stateless ACT/Pool/Format kernels + FP8 LUTs.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc + gtx_npu.h:154-221.

Per CONTEXT D-02: bundled ACT module (act+pool+format_cvt share LSPR ADDRA/ADDRR).
Per CONTEXT D-14/D-15: FP8<->FP16 codecs are LUTs precomputed at module import.
Per RESEARCH Pitfall 5: GTX FP8 uses 2^-6 subnormal base + inf-on-exp=0xF (NOT NVIDIA E4M3).

Plan 03 GREEN-fills the 7 activation kernels (relu/prelu/gelu/tanh_act/sigmoid/
softmax/esum). Plan 04 GREEN-fills pool + cvt kernels + FP8 LUTs.

Per RESEARCH §VSUM/DOT Precision: SOFTMAX exp_sum AND ESUM accumulator MUST use
explicit Python `for` loop FP32 accumulator with single FP16 cast at end.
NEVER `np.sum`, `np.dot`, `np.einsum` on FP16 arrays. (Same precedent as
vec_core.vsum_kernel / vec_core.dot_kernel.)
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Activation kernels -- Plan 03 GREEN
# ============================================================================
def relu(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """RELU: max(0, x). Source: gtx_npu_act.cc:60-67 (forward).

    FP32 internal compute then single FP16 cast at writeback.
    """
    f32 = arr.astype(np.float32)
    return np.maximum(f32, np.float32(0.0)).astype(np.float16)


def prelu(arr: NDArray[np.float16], slope: np.float16) -> NDArray[np.float16]:
    """PRELU: x if x > 0 else slope * x. Source: gtx_npu_act.cc:118-131 (reversed).

    Vendor C++ uses `(a < 0.0f) ? slope * a : a`; we mirror exactly.
    """
    f32 = arr.astype(np.float32)
    s = np.float32(slope)
    return np.where(f32 < np.float32(0.0), f32 * s, f32).astype(np.float16)


def gelu(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """GELU (tanh approximation). Source: gtx_npu_act.cc:95-107 (reversed).

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))
    """
    f32 = arr.astype(np.float32)
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    inner = sqrt_2_over_pi * (f32 + np.float32(0.044715) * f32 * f32 * f32)
    return (np.float32(0.5) * f32 *
            (np.float32(1.0) + np.tanh(inner))).astype(np.float16)


def tanh_act(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """tanh(x). Source: gtx_npu_act.cc:69-76 (reversed)."""
    f32 = arr.astype(np.float32)
    return np.tanh(f32).astype(np.float16)


def sigmoid(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """sigmoid(x) = 1 / (1 + exp(-x)). Source: gtx_npu_act.cc:109-116 (reversed)."""
    f32 = arr.astype(np.float32)
    return (np.float32(1.0) /
            (np.float32(1.0) + np.exp(-f32))).astype(np.float16)


def softmax(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """Softmax: exp(x - max) / sum(exp(x - max)).

    Source: gtx_npu_act.cc:78-93 (forward). Two-pass:
      (1) find max
      (2) compute tmp[i] = exp(x[i] - max); accumulate sum via explicit FP32 loop
      (3) normalize: out[i] = tmp[i] / sum
    Single FP16 cast at writeback. Pitfall 2 lock: NEVER np.sum on FP16.
    """
    f32 = arr.astype(np.float32)
    m = np.float32(float(np.max(f32)))
    tmp = np.exp(f32 - m)
    # Explicit FP32 for-loop sum (Pitfall 2). NEVER np.sum/np.dot on FP16.
    s = np.float32(0.0)
    for v in tmp.ravel():
        s += np.float32(v)
    if s > np.float32(0.0):
        out = tmp / s
    else:
        # Vendor: if sum <= 0, leave wr_addr untouched. We return zeros (writeback
        # is a no-op upstream when this branch fires).
        out = np.zeros_like(tmp)
    return out.astype(np.float16)


def esum(arr: NDArray[np.float16], max_val: np.float16,
         init_accum: np.float16) -> np.float16:
    """ESUM scalar reduction: r = init_accum + sum(exp(x[i] - max_val)).

    Source: gtx_npu_act.cc:133-148 (forward, writes scalar to L0).
    Vendor takes max + accum from GSPR_OPERAND2 [hi:lo] FP16 packed; engine
    extracts and passes; here we just accept FP16 args.

    FP32 internal accumulate + single FP16 cast (Pitfall 2). NEVER np.sum.
    """
    f32 = arr.astype(np.float32)
    m = np.float32(max_val)
    s = np.float32(init_accum)
    for x in f32.ravel():
        s += np.exp(x - m)
    return np.float16(s)


# ============================================================================
# Pool kernels -- Plan 04 GREEN
# ============================================================================
def pool_max(arr: NDArray[np.float16], kernel_size: int) -> NDArray[np.float16]:
    """Max-pool, stride = kernel_size. Source: gtx_npu_act.cc:166-205."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def pool_avg(arr: NDArray[np.float16], kernel_size: int) -> NDArray[np.float16]:
    """Avg-pool, stride = kernel_size. Note: avg += 0.0 canonicalises -0.0 -> +0.0
    (gtx_npu_act.cc:211)."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


# ============================================================================
# Format-cvt kernels -- Plan 04 GREEN
# ============================================================================
def cvt_qh(arr: NDArray[np.float16], scale: np.float16, offset: np.float16) -> NDArray[np.uint8]:
    """FP16 -> FP8. Source: gtx_npu_act.cc:262-271."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_hq(arr: NDArray[np.uint8], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """FP8 -> FP16. Source: gtx_npu_act.cc:251-260."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_ih(arr: NDArray[np.float16], scale: np.float16, offset: np.float16) -> NDArray[np.int8]:
    """FP16 -> INT8. Source: gtx_npu_act.cc:288-297."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_hi(arr: NDArray[np.int8], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """INT8 -> FP16. Source: gtx_npu_act.cc:277-286."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_hn(arr: NDArray[np.int32], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """INT32 -> FP16 normalize (gtx_npu_act.cc:301-313)."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_sh(arr: NDArray[np.float32]) -> NDArray[np.float16]:
    """FP32 -> FP16 (bit-pattern preserving; no scale/offset).
    Source: gtx_npu_act.cc:326-335."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_hs(arr: NDArray[np.float16]) -> NDArray[np.float32]:
    """FP16 -> FP32 (bit-pattern preserving). Source: gtx_npu_act.cc:317-324."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_dh(arr: NDArray[np.float64]) -> NDArray[np.float16]:
    """FP64 -> FP16 (RESEARCH Adjustment 1). Source: gtx_npu_act.cc:351-360."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


def cvt_hd(arr: NDArray[np.float16]) -> NDArray[np.float64]:
    """FP16 -> FP64. Source: gtx_npu_act.cc:342-349."""
    raise NotImplementedError("Plan 04 wave 1b GREEN-fill")


# FP8 LUTs -- Plan 04 builds these at module import time (D-14, D-15).
# Plan 01 ships placeholders so `from .act_core import FP8_TO_FP16_LUT` succeeds.
FP8_TO_FP16_LUT: np.ndarray = np.zeros(256, dtype=np.float16)        # Plan 04 fills
FP16_TO_FP8_LUT: np.ndarray = np.zeros(65536, dtype=np.uint8)        # Plan 04 fills
