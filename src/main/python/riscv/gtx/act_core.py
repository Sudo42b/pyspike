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

Per RESEARCH §VSUM/DOT Precision: SOFTMAX exp_sum AND ESUM accumulator MUST use
explicit Python `for` loop FP32 accumulator with single FP16 cast at end.
NEVER `np.sum`, `np.dot`, `np.einsum` on FP16 arrays. (Same precedent as
vec_core.vsum_kernel / vec_core.dot_kernel.)

P7 NJIT-02 / NJIT-03 / NJIT-FP32-BOUNDARY (Plan 07-04):
    The 18 stateless kernels (7 act + 2 pool + 9 cvt) are factored into FP32-only
    `_impl` functions wrapped by `@njit(cache=True)` (real numba when installed;
    passthrough no-op when absent per `_jit.py`). The public functions retain the
    P5 signatures EXACTLY -- they pre-cast FP16 -> FP32 (numba CPU does NOT support
    np.float16; raises NotImplementedError at compile time, see RESEARCH §"FP16
    NotImplementedError"), call the JIT'd `_impl`, and post-cast FP32 -> FP16
    (or other dtype where applicable). Callers in `act_engine.py` see ZERO behavior
    change.

P7 NJIT-03 / D-09 refined (objmode escape for transcendentals):
    The 5 transcendental kernels (gelu, tanh_act, sigmoid, softmax, esum) wrap
    their `np.tanh` / `np.exp` calls inside `with numba.objmode(...)` to delegate
    to NumPy's libm path. WITHOUT objmode, LLVM's `tanhf` / `expf` intrinsics
    differ from glibc's by ~1 ULP, propagating to FP16 ULP-0 mismatches (RESEARCH
    empirical: 9/1024 GELU mismatches without objmode; 0/1024 with).

    Because `with numba.objmode(...)` is invalid syntax under HAS_NUMBA=False
    (`numba` is None), the 5 transcendental impls are dual-defined: the
    `if HAS_NUMBA` branch contains the @njit-decorated body with objmode; the
    `else` branch contains a pure-NumPy body that does NOT reference `numba`.

P7 FP8 LUTs (NJIT-02 + RESEARCH "Module-level LUT capture"):
    `FP8_TO_FP16_LUT` (256B) and `FP16_TO_FP8_LUT` (64KB) are precomputed at
    module import (P5 D-14/D-15 lock). LUT lookups happen in the public wrappers
    (NOT inside @njit) because the FP16 cast required to view a FP32 result as
    uint16 is unavailable inside @njit (NotImplementedError float16). The wrapper
    pre/post-LUT discipline preserves P5 byte-for-byte semantics.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from ._jit import njit, HAS_NUMBA  # noqa: F401  (HAS_NUMBA re-exposed for callers)

# Optional numba import for objmode (transcendentals); when numba is absent,
# the objmode block is unreachable because the dual-define `else` path is
# taken (HAS_NUMBA=False).
try:
    import numba  # noqa: F401  -- guards `numba.objmode(...)` reference
except ImportError:  # pragma: no cover -- exercised when `spike[fast]` not installed
    numba = None  # type: ignore[assignment]


# ============================================================================
# SECTION A -- FP8 codec LUTs (preserved verbatim from P5)
# gtx_npu.h:154-221 (RESEARCH §FP8 Codec lines 419-468)
# ============================================================================
def _build_fp8_to_fp16_lut() -> np.ndarray:
    """Direct port of gtx_npu.h:154-179 gtx_fp8_to_32 (cast to FP16 at output).

    DIVERGENCES from NVIDIA E4M3 (Pitfall 5):
      - Subnormal (h_exp=0, h_frac>0): uses 2^-6 base (NVIDIA: 2^-9).
      - h_exp=0xF, h_frac=0: maps to inf (NVIDIA: NaN; no inf).
      - h_exp=0xF, h_frac>0: maps to NaN.

    Bit layout: sign[7] exp[6:3] frac[2:0]; bias = 7.
    """
    out = np.zeros(256, dtype=np.float16)
    for h in range(256):
        h_sign = (h & 0x80) >> 7
        h_exp  = (h & 0x78) >> 3
        h_frac = h & 0x07
        if h_exp == 0:
            if h_frac == 0:
                val = 0.0
            else:
                val = (h_frac / 8.0) * (2.0 ** -6)
        elif h_exp == 0xF:
            if h_frac == 0:
                val = float('inf')
            else:
                val = float('nan')
        else:
            val = (1.0 + h_frac / 8.0) * (2.0 ** (h_exp - 7))
        if h_sign and not np.isnan(val):
            val = -val
        out[h] = np.float16(val)
    # Preserve negative-zero bit pattern for h=0x80 (sign=1, exp=0, frac=0).
    # `val = -0.0` followed by `np.float16(-0.0)` gives 0x8000 LE -- correct.
    out[0x80] = np.float16(-0.0)
    return out


def _build_fp16_to_fp8_lut() -> np.ndarray:
    """Direct port of gtx_npu.h:182-221 gtx_fp16_to_8.

    For all 65536 FP16 inputs, compute the closest FP8 byte. Cases:
      h_exp=0x1F (FP16 NaN/inf): NaN -> sign|0xF8|0x01; inf -> sign|0xF8.
      new_e in [1, 14]: normal -- shift mantissa, RNE round, handle overflow.
      new_e <= 0: subnormal -- right-shift, RNE round; promote to normal at 0x8.
      new_e > 14: overflow -> sign|0xF8 (inf).

    Built once at module import (D-15). 64KB cost; one-time ~50-100 ms.
    """
    out = np.zeros(65536, dtype=np.uint8)
    for h in range(65536):
        h_sign = (h >> 15) & 0x1
        h_exp  = (h >> 10) & 0x1F
        h_frac = h & 0x03FF
        sign8 = (h_sign & 0x1) << 7
        if h_exp == 0x1F:
            out[h] = (sign8 | 0xF8 | 0x01) if h_frac else (sign8 | 0xF8)
            continue
        e16 = -14 if h_exp == 0 else (int(h_exp) - 15)
        new_e = e16 + 7
        sig = h_frac if h_exp == 0 else (0x400 | h_frac)
        if 1 <= new_e <= 14:
            out_bits = 4
            shift = 7
            main = sig >> shift
            round_bit = (sig >> (shift - 1)) & 1
            sticky = sig & ((1 << (shift - 1)) - 1)
            if round_bit and (sticky or (main & 1)):
                main += 1
            if main == (1 << out_bits):
                new_e += 1
                main = 1 << (out_bits - 1)
                if new_e >= 0xF:
                    out[h] = sign8 | 0xF8
                    continue
            out[h] = sign8 | ((new_e & 0xF) << 3) | (main & 0x7)
            continue
        if new_e <= 0:
            total_shift = 8 - new_e
            if total_shift >= 32:
                out[h] = sign8
                continue
            frac = sig >> total_shift
            rb_pos = total_shift - 1
            round_bit = (sig >> rb_pos) & 1
            sticky = (sig & ((1 << rb_pos) - 1)) if rb_pos > 0 else 0
            if round_bit and (sticky or (frac & 1)):
                frac += 1
                if frac == 0x8:
                    out[h] = sign8 | (1 << 3)
                    continue
            out[h] = sign8 | (frac & 0x7)
            continue
        # new_e > 14: overflow -> inf
        out[h] = sign8 | 0xF8
    return out


# Build at module import (D-14, D-15). Numba captures these as Read-only Globals
# (RESEARCH "Module-level LUT capture" verified empirical) BUT FP16 globals are
# not directly usable inside @njit. The LUTs are accessed exclusively from the
# public wrappers (NOT inside `_impl` bodies).
FP8_TO_FP16_LUT: np.ndarray = _build_fp8_to_fp16_lut()
FP16_TO_FP8_LUT: np.ndarray = _build_fp16_to_fp8_lut()


# ============================================================================
# SECTION B -- FP32-only `_impl` functions (numba JIT boundary)
# 18 kernels: 7 activation + 2 pool + 9 cvt
# ============================================================================

# ---- Activations: 2 non-transcendental (relu, prelu) -----------------------
def _relu_impl(arr_f32: NDArray[np.float32]) -> NDArray[np.float32]:
    """RELU FP32: max(0, x). Source: gtx_npu_act.cc:60-67 (forward).

    Numba @njit boundary: FP32 in / FP32 out. `np.maximum` is supported.
    """
    return np.maximum(arr_f32, np.float32(0.0))


def _prelu_impl(
    arr_f32: NDArray[np.float32],
    slope_f32: np.float32,
) -> NDArray[np.float32]:
    """PRELU FP32: x if x >= 0 else slope * x. Source: gtx_npu_act.cc:118-131
    (reversed). Vendor C++ uses `(a < 0.0f) ? slope * a : a`; we mirror exactly.
    Numba @njit boundary: FP32 in / FP32 out.
    """
    return np.where(arr_f32 < np.float32(0.0), slope_f32 * arr_f32, arr_f32)


# ---- Activations: 5 transcendentals (objmode escape per NJIT-03 / D-09) ----
#
# Dual-define under HAS_NUMBA fork (W5 lock). The `if HAS_NUMBA` branch contains
# the @njit-decorated body with `with numba.objmode(...)`; the `else` branch
# contains a pure-NumPy body that does NOT reference `numba.objmode` at all.
#
# RESEARCH empirical: WITHOUT objmode, GELU drifts 9/1024 FP16 ULP-0 mismatches;
# WITH objmode, 0/1024.
if HAS_NUMBA:
    @njit(cache=True)
    def _gelu_njit(arr_f32):
        """GELU FP32 (tanh approx) — `np.tanh` via objmode escape.
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x**3)))
        """
        sqrt_2_over_pi = np.float32(0.7978845608028654)
        c = np.float32(0.044715)
        n = arr_f32.shape[0]
        inner = np.empty(n, dtype=np.float32)
        for i in range(n):
            inner[i] = sqrt_2_over_pi * (arr_f32[i] + c * arr_f32[i] * arr_f32[i] * arr_f32[i])
        with numba.objmode(t='float32[:]'):
            t = np.tanh(inner).astype(np.float32)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = np.float32(0.5) * arr_f32[i] * (np.float32(1.0) + t[i])
        return out

    @njit(cache=True)
    def _tanh_act_njit(arr_f32):
        """tanh(x) FP32 — full objmode delegation."""
        n = arr_f32.shape[0]
        # Pre-build a contiguous flat view for objmode (objmode prefers explicit
        # 1-D arrays).
        flat = np.empty(n, dtype=np.float32)
        for i in range(n):
            flat[i] = arr_f32[i]
        with numba.objmode(t='float32[:]'):
            t = np.tanh(flat).astype(np.float32)
        return t

    @njit(cache=True)
    def _sigmoid_njit(arr_f32):
        """sigmoid(x) = 1/(1+exp(-x)) FP32 — `np.exp` via objmode escape."""
        n = arr_f32.shape[0]
        neg = np.empty(n, dtype=np.float32)
        for i in range(n):
            neg[i] = -arr_f32[i]
        with numba.objmode(e='float32[:]'):
            e = np.exp(neg).astype(np.float32)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = np.float32(1.0) / (np.float32(1.0) + e[i])
        return out

    @njit(cache=True)
    def _softmax_njit(arr_f32):
        """Softmax FP32: exp(x - max) / sum(exp(x - max)) with zero-fallback.
        Mirror current act_core.py:84-106 — when sum<=0, return zeros (vendor
        leaves wr_addr untouched; we return zeros and writeback is no-op).
        Source: gtx_npu_act.cc:78-93. NEVER np.sum on FP16 (P5 D-09 / Pitfall 2).
        """
        n = arr_f32.shape[0]
        m = arr_f32[0]
        for i in range(1, n):
            if arr_f32[i] > m:
                m = arr_f32[i]
        shifted = np.empty(n, dtype=np.float32)
        for i in range(n):
            shifted[i] = arr_f32[i] - m
        with numba.objmode(e='float32[:]'):
            e = np.exp(shifted).astype(np.float32)
        s = np.float32(0.0)
        for i in range(n):
            s += e[i]
        out = np.empty(n, dtype=np.float32)
        if s > np.float32(0.0):
            for i in range(n):
                out[i] = e[i] / s
        else:
            for i in range(n):
                out[i] = np.float32(0.0)
        return out

    @njit(cache=True)
    def _esum_njit(arr_f32, max_val_f32, init_accum_f32):
        """ESUM scalar reduction FP32: r = init_accum + sum(exp(x[i] - max_val)).
        Source: gtx_npu_act.cc:133-148. Public wrapper does FP16 cast.
        Explicit Python for-loop FP32 accumulate (P5 D-09 lineage). objmode
        escape for `np.exp`.
        """
        n = arr_f32.shape[0]
        shifted = np.empty(n, dtype=np.float32)
        for i in range(n):
            shifted[i] = arr_f32[i] - max_val_f32
        with numba.objmode(e='float32[:]'):
            e = np.exp(shifted).astype(np.float32)
        s = init_accum_f32
        for i in range(n):
            s += e[i]
        return s

    # Aliases for parity-test introspection (test imports `_<name>_njit`).
    _gelu_impl = _gelu_njit
    _tanh_act_impl = _tanh_act_njit
    _sigmoid_impl = _sigmoid_njit
    _softmax_impl = _softmax_njit
    _esum_impl = _esum_njit
else:  # pragma: no cover -- HAS_NUMBA=False path (no numba installed)
    def _gelu_impl(arr_f32):
        """GELU FP32 pure-NumPy. NOT inside @njit. No numba/objmode reference."""
        sqrt_2_over_pi = np.float32(0.7978845608028654)
        inner = sqrt_2_over_pi * (
            arr_f32 + np.float32(0.044715) * arr_f32 * arr_f32 * arr_f32
        )
        t = np.tanh(inner).astype(np.float32)
        return np.float32(0.5) * arr_f32 * (np.float32(1.0) + t)

    def _tanh_act_impl(arr_f32):
        """tanh(x) FP32 pure-NumPy."""
        return np.tanh(arr_f32).astype(np.float32)

    def _sigmoid_impl(arr_f32):
        """sigmoid FP32 pure-NumPy."""
        return (np.float32(1.0) /
                (np.float32(1.0) + np.exp(-arr_f32))).astype(np.float32)

    def _softmax_impl(arr_f32):
        """Softmax FP32 pure-NumPy with zero-fallback."""
        m = np.float32(float(np.max(arr_f32)))
        tmp = np.exp(arr_f32 - m).astype(np.float32)
        s = np.float32(0.0)
        for v in tmp.ravel():
            s += np.float32(v)
        if s > np.float32(0.0):
            return (tmp / s).astype(np.float32)
        return np.zeros_like(tmp, dtype=np.float32)

    def _esum_impl(arr_f32, max_val_f32, init_accum_f32):
        """ESUM scalar reduction FP32 pure-NumPy."""
        s = np.float32(init_accum_f32)
        for x in arr_f32.ravel():
            s += np.exp(x - max_val_f32)
        return s

    _gelu_njit = _gelu_impl
    _tanh_act_njit = _tanh_act_impl
    _sigmoid_njit = _sigmoid_impl
    _softmax_njit = _softmax_impl
    _esum_njit = _esum_impl


# ---- Pool: 2 kernels (max, avg) --------------------------------------------
def _pool_max_impl(
    arr_f32: NDArray[np.float32],
    kernel_size: int,
) -> NDArray[np.float32]:
    """Max-pool FP32, stride = kernel_size, non-overlapping windows.
    Source: gtx_npu_act.cc:166-205 (exec_pooling, is_max=True branch).
    """
    if kernel_size == 0:
        return np.empty(0, dtype=np.float32)
    n = arr_f32.shape[0]
    out_len = n // kernel_size
    out = np.empty(out_len, dtype=np.float32)
    for o in range(out_len):
        base = o * kernel_size
        m = arr_f32[base]
        for k in range(1, kernel_size):
            if arr_f32[base + k] > m:
                m = arr_f32[base + k]
        out[o] = m
    return out


def _pool_avg_impl(
    arr_f32: NDArray[np.float32],
    kernel_size: int,
) -> NDArray[np.float32]:
    """Avg-pool FP32. Source: gtx_npu_act.cc:166-220.
    +0.0 canonicalization (line 211): `avg += 0.0` forces (-0.0)+(+0.0)=+0.0.
    P5 Plan 04 D-3 invariant.
    """
    if kernel_size == 0:
        return np.empty(0, dtype=np.float32)
    n = arr_f32.shape[0]
    out_len = n // kernel_size
    ks_f32 = np.float32(kernel_size)
    out = np.empty(out_len, dtype=np.float32)
    for o in range(out_len):
        base = o * kernel_size
        s = arr_f32[base]
        for k in range(1, kernel_size):
            s += arr_f32[base + k]
        avg = s / ks_f32
        avg = avg + np.float32(0.0)  # signed-zero canon (cc:211)
        out[o] = avg
    return out


# ---- Cvt: 9 kernels (FP8/INT8/INT32/FP32/FP64 conversions) -----------------
#
# DESIGN NOTE — FP16 boundary in @njit:
#   FP16-output cvts (cvt_hq, cvt_hi, cvt_hn, cvt_sh, cvt_dh) cannot return
#   FP16 from `_impl` because numba CPU rejects float16 at typing time. The
#   `_impl` returns FP32 (or the input's natural dtype where no scale/offset
#   applies); the public wrapper casts FP32 -> FP16.
#
#   FP16-input cvts (cvt_qh, cvt_ih, cvt_hq) accept FP32 in `_impl` (wrapper
#   pre-casts FP16 -> FP32). For LUT lookups (cvt_qh outputs uint8, cvt_hq
#   inputs uint8), the LUT lookup that needs FP16 -> uint16 view stays in the
#   wrapper (cvt_qh post-cast); the LUT lookup that needs uint8 indexing is
#   pre-applied in the wrapper for cvt_hq.

def _cvt_qh_impl(
    arr_f32: NDArray[np.float32],
    scale_f32: np.float32,
    offset_f32: np.float32,
) -> NDArray[np.float32]:
    """FP16->FP8 inner: just FP32 scale+offset. Wrapper does FP16 cast + LUT.
    Source: gtx_npu_act.cc:262-271.
    """
    n = arr_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = arr_f32[i] * scale_f32 + offset_f32
    return out


def _cvt_hq_impl(
    decoded_f32: NDArray[np.float32],
    scale_f32: np.float32,
    offset_f32: np.float32,
) -> NDArray[np.float32]:
    """FP8->FP16 inner: just FP32 scale+offset on already-LUT-decoded values.
    Wrapper does LUT[uint8] -> FP16 -> FP32 cast pre-step + FP16 cast post.
    Source: gtx_npu_act.cc:251-260.
    """
    n = decoded_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = decoded_f32[i] * scale_f32 + offset_f32
    return out


def _cvt_ih_impl(
    arr_f32: NDArray[np.float32],
    scale_f32: np.float32,
    offset_f32: np.float32,
) -> NDArray[np.int8]:
    """FP16->INT8 inner: scale+offset, round, saturating clip to [-128,127].
    Source: gtx_npu_act.cc:288-297.
    """
    n = arr_f32.shape[0]
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        scaled = arr_f32[i] * scale_f32 + offset_f32
        # numpy default rounding is round-half-to-even; emulate via np.round
        # but inside @njit use explicit IEEE 754 round-to-nearest-even.
        r = np.rint(scaled)
        if r > np.float32(127.0):
            out[i] = np.int8(127)
        elif r < np.float32(-128.0):
            out[i] = np.int8(-128)
        else:
            out[i] = np.int8(r)
    return out


def _cvt_hi_impl(
    arr_i8: NDArray[np.int8],
    scale_f32: np.float32,
    offset_f32: np.float32,
) -> NDArray[np.float32]:
    """INT8->FP16 inner: int8*scale+offset in FP32. Wrapper casts to FP16.
    Source: gtx_npu_act.cc:277-286.
    """
    n = arr_i8.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(arr_i8[i]) * scale_f32 + offset_f32
    return out


def _cvt_hn_impl(
    arr_i32: NDArray[np.int32],
    scale_f32: np.float32,
    offset_f32: np.float32,
) -> NDArray[np.float32]:
    """INT32->FP16 normalize inner: int32*scale+offset in FP32.
    Source: gtx_npu_act.cc:301-313.
    """
    n = arr_i32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(arr_i32[i]) * scale_f32 + offset_f32
    return out


def _cvt_sh_impl(arr_f32: NDArray[np.float32]) -> NDArray[np.float32]:
    """FP32->FP16 inner: passthrough FP32. Wrapper casts to FP16.
    Source: gtx_npu_act.cc:326-335 (no scale/offset).
    """
    n = arr_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = arr_f32[i]
    return out


def _cvt_hs_impl(arr_f32: NDArray[np.float32]) -> NDArray[np.float32]:
    """FP16->FP32 inner: passthrough FP32 (wrapper pre-casts FP16->FP32).
    Source: gtx_npu_act.cc:317-324.
    """
    n = arr_f32.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = arr_f32[i]
    return out


def _cvt_dh_impl(arr_f64: NDArray[np.float64]) -> NDArray[np.float32]:
    """FP64->FP16 inner: cast FP64->FP32 (wrapper does FP32->FP16 cast).
    Source: gtx_npu_act.cc:351-360.

    Note: FP64->FP16 directly differs from FP64->FP32->FP16 by at most 1 ULP
    on the boundary. The current pure-NumPy P5 path uses `arr.astype(np.float16)`
    which is FP64->FP16 directly. To preserve byte-for-byte parity, the wrapper
    bypasses _impl when needed; see cvt_dh public function.
    """
    n = arr_f64.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(arr_f64[i])
    return out


def _cvt_hd_impl(arr_f32: NDArray[np.float32]) -> NDArray[np.float64]:
    """FP16->FP64 inner: cast FP32->FP64 (wrapper pre-casts FP16->FP32).
    Source: gtx_npu_act.cc:342-349.
    """
    n = arr_f32.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.float64(arr_f32[i])
    return out


# ============================================================================
# SECTION C -- @njit(cache=True) wrappers (re-call pattern)
# Transcendentals (5) are already decorated inside the HAS_NUMBA fork above;
# below covers the 13 non-transcendentals.
# ============================================================================
_relu_njit       = njit(cache=True)(_relu_impl)
_prelu_njit      = njit(cache=True)(_prelu_impl)
_pool_max_njit   = njit(cache=True)(_pool_max_impl)
_pool_avg_njit   = njit(cache=True)(_pool_avg_impl)
_cvt_qh_njit     = njit(cache=True)(_cvt_qh_impl)
_cvt_hq_njit     = njit(cache=True)(_cvt_hq_impl)
_cvt_ih_njit     = njit(cache=True)(_cvt_ih_impl)
_cvt_hi_njit     = njit(cache=True)(_cvt_hi_impl)
_cvt_hn_njit     = njit(cache=True)(_cvt_hn_impl)
_cvt_sh_njit     = njit(cache=True)(_cvt_sh_impl)
_cvt_hs_njit     = njit(cache=True)(_cvt_hs_impl)
_cvt_dh_njit     = njit(cache=True)(_cvt_dh_impl)
_cvt_hd_njit     = njit(cache=True)(_cvt_hd_impl)


# ============================================================================
# SECTION D -- Public API (P5 signatures preserved verbatim; FP16 in/out)
# ============================================================================

# ---- Activations (7) -------------------------------------------------------
def relu(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """RELU: max(0, x). Source: gtx_npu_act.cc:60-67 (forward).
    FP32 internal compute then single FP16 cast at writeback.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _relu_njit(a_f32)
    return out_f32.astype(np.float16)


def prelu(arr: NDArray[np.float16], slope: np.float16) -> NDArray[np.float16]:
    """PRELU: x if x >= 0 else slope * x. Source: gtx_npu_act.cc:118-131
    (reversed). Vendor C++ uses `(a < 0.0f) ? slope * a : a`.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    s_f32 = np.float32(slope)
    out_f32 = _prelu_njit(a_f32, s_f32)
    return out_f32.astype(np.float16)


def gelu(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """GELU (tanh approximation). Source: gtx_npu_act.cc:95-107 (reversed).
    NJIT-03: `np.tanh` via `numba.objmode` escape (RESEARCH D-09 refined).
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _gelu_njit(a_f32)
    return out_f32.astype(np.float16)


def tanh_act(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """tanh(x). Source: gtx_npu_act.cc:69-76 (reversed).
    NJIT-03: `np.tanh` via `numba.objmode` escape.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _tanh_act_njit(a_f32)
    return out_f32.astype(np.float16)


def sigmoid(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """sigmoid(x) = 1/(1+exp(-x)). Source: gtx_npu_act.cc:109-116 (reversed).
    NJIT-03: `np.exp` via `numba.objmode` escape.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _sigmoid_njit(a_f32)
    return out_f32.astype(np.float16)


def softmax(arr: NDArray[np.float16]) -> NDArray[np.float16]:
    """Softmax: exp(x - max) / sum(exp(x - max)).
    Source: gtx_npu_act.cc:78-93 (forward). NJIT-03: `np.exp` via objmode.
    Pitfall 2 lock: NEVER np.sum on FP16. Zero-fallback when sum<=0.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _softmax_njit(a_f32)
    return out_f32.astype(np.float16)


def esum(arr: NDArray[np.float16], max_val: np.float16,
         init_accum: np.float16) -> np.float16:
    """ESUM scalar reduction: r = init_accum + sum(exp(x[i] - max_val)).
    Source: gtx_npu_act.cc:133-148 (forward, writes scalar to L0).
    FP32 internal accumulate + single FP16 cast (Pitfall 2). NJIT-03: `np.exp`
    via objmode escape.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    m_f32 = np.float32(max_val)
    init_f32 = np.float32(init_accum)
    s = _esum_njit(a_f32, m_f32, init_f32)
    return np.float16(s)


# ---- Pool (2) --------------------------------------------------------------
def pool_max(arr: NDArray[np.float16], kernel_size: int) -> NDArray[np.float16]:
    """Max-pool, stride = kernel_size, non-overlapping windows.
    Source: gtx_npu_act.cc:166-205 (exec_pooling, is_max=True branch).
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _pool_max_njit(a_f32, int(kernel_size))
    return out_f32.astype(np.float16)


def pool_avg(arr: NDArray[np.float16], kernel_size: int) -> NDArray[np.float16]:
    """Avg-pool, stride = kernel_size. Source: gtx_npu_act.cc:166-220.
    +0.0 canonicalization preserved (P5 Plan 04 D-3 invariant).
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _pool_avg_njit(a_f32, int(kernel_size))
    return out_f32.astype(np.float16)


# ---- Cvt (9) ---------------------------------------------------------------
def cvt_qh(arr: NDArray[np.float16], scale: np.float16,
           offset: np.float16) -> NDArray[np.uint8]:
    """FP16 -> FP8. `a = a * scale + offset` then encode via LUT.
    Source: gtx_npu_act.cc:262-271. LUT lookup stays in wrapper (FP16 cast
    forbidden inside @njit per RESEARCH Pitfall 1).
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    s_f32 = np.float32(scale)
    o_f32 = np.float32(offset)
    f32_scaled = _cvt_qh_njit(a_f32, s_f32, o_f32)
    fp16 = f32_scaled.astype(np.float16)
    return FP16_TO_FP8_LUT[fp16.view(np.uint16).astype(np.intp)]


def cvt_hq(arr: NDArray[np.uint8], scale: np.float16,
           offset: np.float16) -> NDArray[np.float16]:
    """FP8 -> FP16. Decode via LUT then `out = decoded * scale + offset`.
    Source: gtx_npu_act.cc:251-260. LUT decode pre-step in wrapper.
    """
    decoded_f16 = FP8_TO_FP16_LUT[arr.view(np.uint8).astype(np.intp)]
    decoded_f32 = decoded_f16.astype(np.float32)
    s_f32 = np.float32(scale)
    o_f32 = np.float32(offset)
    out_f32 = _cvt_hq_njit(decoded_f32, s_f32, o_f32)
    return out_f32.astype(np.float16)


def cvt_ih(arr: NDArray[np.float16], scale: np.float16,
           offset: np.float16) -> NDArray[np.int8]:
    """FP16 -> INT8. `int8(round(a * scale + offset))` saturating to [-128, 127].
    Source: gtx_npu_act.cc:288-297.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    s_f32 = np.float32(scale)
    o_f32 = np.float32(offset)
    return _cvt_ih_njit(a_f32, s_f32, o_f32)


def cvt_hi(arr: NDArray[np.int8], scale: np.float16,
           offset: np.float16) -> NDArray[np.float16]:
    """INT8 -> FP16. `out = int8 * scale + offset`.
    Source: gtx_npu_act.cc:277-286.
    """
    a = np.ascontiguousarray(arr, dtype=np.int8)
    s_f32 = np.float32(scale)
    o_f32 = np.float32(offset)
    out_f32 = _cvt_hi_njit(a, s_f32, o_f32)
    return out_f32.astype(np.float16)


def cvt_hn(arr: NDArray[np.int32], scale: np.float16,
           offset: np.float16) -> NDArray[np.float16]:
    """INT32 -> FP16 normalize. `out = int32 * scale + offset`.
    Source: gtx_npu_act.cc:301-313.
    """
    a = np.ascontiguousarray(arr, dtype=np.int32)
    s_f32 = np.float32(scale)
    o_f32 = np.float32(offset)
    out_f32 = _cvt_hn_njit(a, s_f32, o_f32)
    return out_f32.astype(np.float16)


def cvt_sh(arr: NDArray[np.float32]) -> NDArray[np.float16]:
    """FP32 -> FP16 (bit-pattern preserving; NO scale/offset).
    Source: gtx_npu_act.cc:326-335.
    """
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    out_f32 = _cvt_sh_njit(a_f32)
    return out_f32.astype(np.float16)


def cvt_hs(arr: NDArray[np.float16]) -> NDArray[np.float32]:
    """FP16 -> FP32 (bit-pattern preserving). Source: gtx_npu_act.cc:317-324."""
    a_f32 = np.ascontiguousarray(arr, dtype=np.float32)
    return _cvt_hs_njit(a_f32)


def cvt_dh(arr: NDArray[np.float64]) -> NDArray[np.float16]:
    """FP64 -> FP16 (bit-pattern preserving). Source: gtx_npu_act.cc:351-360.

    NumPy's `arr.astype(np.float16)` does FP64->FP16 directly (single rounding).
    To preserve P5 byte-for-byte semantics, the wrapper performs the direct cast
    here; the `_cvt_dh_njit` is unused for the public path but registered for
    the parity test (Tier 1 compares public output to a separately-cast variant).
    """
    return arr.astype(np.float16)


def cvt_hd(arr: NDArray[np.float16]) -> NDArray[np.float64]:
    """FP16 -> FP64 (bit-pattern preserving). Source: gtx_npu_act.cc:342-349.

    NumPy's `arr.astype(np.float64)` does FP16->FP64 directly (single widening,
    bit-exact). To preserve P5 byte-for-byte semantics, the wrapper performs the
    direct cast here.
    """
    return arr.astype(np.float64)
