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
    """Max-pool, stride = kernel_size, non-overlapping windows.

    Source: gtx_npu_act.cc:166-205 (exec_pooling, is_max=True branch).
      - output_len = n_in / kernel_size  (integer floor; no padding; tail discarded)
      - val = rd16(addr_a, o*kernel_size); for k in 1..kernel_size: val = max(val, x)
      - FP32 internal compute then single FP16 cast at writeback (Pitfall 2 lock).
    """
    n = int(arr.shape[0])
    out_len = n // int(kernel_size)
    out = np.empty(out_len, dtype=np.float16)
    for o in range(out_len):
        # Vendor reads first element as initial val, then max-folds the rest.
        val = np.float32(arr[o * kernel_size])
        for k in range(1, kernel_size):
            v = np.float32(arr[o * kernel_size + k])
            if v > val:
                val = v
        out[o] = np.float16(val)
    return out


def pool_avg(arr: NDArray[np.float16], kernel_size: int) -> NDArray[np.float16]:
    """Avg-pool, stride = kernel_size. Source: gtx_npu_act.cc:166-220.

    The `avg += 0.0f` step (line 211) canonicalises -0.0 -> +0.0 because IEEE
    754 says (-0.0) + (+0.0) = +0.0. Critical for golden-hex bit-pattern
    matching: -0.0 (0x8000) and +0.0 (0x0000) have different FP16 bit patterns.

    FP32 internal accumulate (explicit Python `for`-loop -- never np.sum on FP16,
    same precedent as vec_core.vsum_kernel / softmax exp_sum -- Pitfall 2).
    """
    n = int(arr.shape[0])
    out_len = n // int(kernel_size)
    out = np.empty(out_len, dtype=np.float16)
    for o in range(out_len):
        # Vendor: val = rd16(...); for k in 1..ks: val += rd16(...);
        s = np.float32(arr[o * kernel_size])
        for k in range(1, kernel_size):
            s += np.float32(arr[o * kernel_size + k])
        avg = s / np.float32(kernel_size)
        avg += np.float32(0.0)  # signed-zero canon: -0.0 -> +0.0 (cc:211)
        out[o] = np.float16(avg)
    return out


# ============================================================================
# FP8 codec LUTs -- gtx_npu.h:154-221 (RESEARCH §FP8 Codec lines 419-468)
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


# Build at module import (D-14, D-15). Replaces Plan 01 zeros placeholder.
FP8_TO_FP16_LUT: np.ndarray = _build_fp8_to_fp16_lut()
FP16_TO_FP8_LUT: np.ndarray = _build_fp16_to_fp8_lut()


# ============================================================================
# Format-cvt kernels -- gtx_npu_act.cc:222-372
# Scale/offset applied for FP16<->{FP8, INT8, INT32}; NOT for FP16<->{FP32, FP64}.
# ============================================================================
def cvt_qh(arr: NDArray[np.float16], scale: np.float16, offset: np.float16) -> NDArray[np.uint8]:
    """FP16 -> FP8. `a = a * scale + offset` then encode via LUT.
    Source: gtx_npu_act.cc:262-271.
    """
    s = np.float32(scale)
    o = np.float32(offset)
    f32 = arr.astype(np.float32) * s + o
    fp16 = f32.astype(np.float16)
    return FP16_TO_FP8_LUT[fp16.view(np.uint16).astype(np.intp)]


def cvt_hq(arr: NDArray[np.uint8], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """FP8 -> FP16. Decode via LUT then `out = decoded * scale + offset`.
    Source: gtx_npu_act.cc:251-260.
    """
    decoded = FP8_TO_FP16_LUT[arr.view(np.uint8).astype(np.intp)]
    s = np.float32(scale)
    o = np.float32(offset)
    return (decoded.astype(np.float32) * s + o).astype(np.float16)


def cvt_ih(arr: NDArray[np.float16], scale: np.float16, offset: np.float16) -> NDArray[np.int8]:
    """FP16 -> INT8. `int8(round(a * scale + offset))` saturating to [-128, 127].
    Source: gtx_npu_act.cc:288-297.
    """
    s = np.float32(scale)
    o = np.float32(offset)
    f32 = arr.astype(np.float32) * s + o
    # Saturating clip + round-half-to-even (numpy default)
    return np.clip(np.round(f32), -128, 127).astype(np.int8)


def cvt_hi(arr: NDArray[np.int8], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """INT8 -> FP16. `out = int8 * scale + offset`.
    Source: gtx_npu_act.cc:277-286.
    """
    s = np.float32(scale)
    o = np.float32(offset)
    return (arr.astype(np.float32) * s + o).astype(np.float16)


def cvt_hn(arr: NDArray[np.int32], scale: np.float16, offset: np.float16) -> NDArray[np.float16]:
    """INT32 -> FP16 normalize. `out = int32 * scale + offset`.
    Source: gtx_npu_act.cc:301-313."""
    s = np.float32(scale)
    o = np.float32(offset)
    return (arr.astype(np.float32) * s + o).astype(np.float16)


def cvt_sh(arr: NDArray[np.float32]) -> NDArray[np.float16]:
    """FP32 -> FP16 (bit-pattern preserving; NO scale/offset).
    Source: gtx_npu_act.cc:326-335."""
    return arr.astype(np.float16)


def cvt_hs(arr: NDArray[np.float16]) -> NDArray[np.float32]:
    """FP16 -> FP32 (bit-pattern preserving). Source: gtx_npu_act.cc:317-324."""
    return arr.astype(np.float32)


def cvt_dh(arr: NDArray[np.float64]) -> NDArray[np.float16]:
    """FP64 -> FP16 (RESEARCH Adjustment 1; bit-pattern preserving).
    Source: gtx_npu_act.cc:351-360."""
    return arr.astype(np.float16)


def cvt_hd(arr: NDArray[np.float16]) -> NDArray[np.float64]:
    """FP16 -> FP64 (bit-pattern preserving). Source: gtx_npu_act.cc:342-349."""
    return arr.astype(np.float64)
