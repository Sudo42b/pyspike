from __future__ import annotations
import torch
from torch import Tensor

def _build_fp8_to_fp16_lut() -> torch.Tensor:
    vals: list[float] = []
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
        if h_sign and not (val != val):  # x!=x is NaN-safe (no numpy needed)
            val = -val
        vals.append(val)
    out = torch.tensor(vals, dtype=torch.float16)
    # Preserve negative-zero bit pattern for h=0x80 (sign=1, exp=0, frac=0).
    # torch.tensor(-0.0, dtype=torch.float16) -> 0x8000 LE (verified empirically).
    out[0x80] = torch.tensor(-0.0, dtype=torch.float16)
    return out

def _build_fp16_to_fp8_lut() -> torch.Tensor:
    out = torch.zeros(65536, dtype=torch.uint8)
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

FP8_TO_FP16_LUT: torch.Tensor = _build_fp8_to_fp16_lut()
FP16_TO_FP8_LUT: torch.Tensor = _build_fp16_to_fp8_lut()

def fp8_e4m3_to_fp16(t_e4m3: torch.Tensor) -> torch.Tensor:
    return t_e4m3.to(torch.float16)

def fp16_to_fp8_e4m3(t_fp16: torch.Tensor) -> torch.Tensor:
    return t_fp16.to(torch.float8_e4m3fn)

def _cvt_qh_impl(
    arr_f32:Tensor,
    scale_f32:float,
    offset_f32:float,
) -> torch.Tensor:
    """ FP16 -> FP8. `a = a * scale + offset` then quantize to FP8 with round-to-nearest-ties-to-even.

    Args:
        arr_f32 (_type_): _description_
        scale_f32 (_type_): _description_
        offset_f32 (_type_): _description_

    Returns:
        torch.Tensor: _description_
    """
    arr = arr_f32.to(torch.float32)
    scale = torch.as_tensor(scale_f32, dtype=torch.float32)
    offset = torch.as_tensor(offset_f32, dtype=torch.float32)
    return (arr * scale + offset).to(torch.float8_e4m3fn)

def _cvt_hq_impl(
    decoded_f32: Tensor,
    scale_f32: float,
    offset_f32: float,
) -> torch.Tensor:
    decoded = decoded_f32.to(torch.float32)
    scale = torch.as_tensor(scale_f32, dtype=torch.float32)
    offset = torch.as_tensor(offset_f32, dtype=torch.float32)
    return (decoded * scale + offset).to(torch.float16)

def _cvt_ih_impl(
    arr_f32: Tensor,
    scale_f32: float,
    offset_f32: float,
) -> torch.Tensor:
    arr = arr_f32.to(torch.float32)
    scale = torch.as_tensor(scale_f32, dtype=torch.float32)
    offset = torch.as_tensor(offset_f32, dtype=torch.float32)
    scaled = arr * scale + offset
    return torch.clamp(torch.round(scaled), -128, 127).to(torch.int8)


def _cvt_hi_impl(
    arr_i8: Tensor,
    scale_f32: float,
    offset_f32: float,
) -> torch.Tensor:
    arr = arr_i8.to(torch.float32)
    scale = torch.as_tensor(scale_f32, dtype=torch.float32)
    offset = torch.as_tensor(offset_f32, dtype=torch.float32)
    return (arr * scale + offset).to(torch.float16)

def _cvt_hn_impl(
    arr_i32: Tensor,
    scale_f32: float,
    offset_f32: float,
) -> torch.Tensor:
    arr = arr_i32.to(torch.float32)
    scale = torch.as_tensor(scale_f32, dtype=torch.float32)
    offset = torch.as_tensor(offset_f32, dtype=torch.float32)
    return (arr * scale + offset).to(torch.float16)

def _cvt_sh_impl(arr_f32: Tensor) -> torch.Tensor:
    return arr_f32.to(torch.float16)

def _cvt_hs_impl(arr_f32: Tensor) -> torch.Tensor:
    return arr_f32.to(torch.float32)

def _cvt_dh_impl(arr_f64: Tensor) -> torch.Tensor:
    return arr_f64.to(torch.float16)


def _cvt_hd_impl(arr_f32: Tensor) -> torch.Tensor:
    """FP16->FP64 inner: cast FP32->FP64"""
    return arr_f32.to(torch.float64)

# ---- Activations -------------------------------------------------------
def relu(arr_f16:Tensor) -> torch.Tensor:
    arr = arr_f16.to(torch.float32)
    return torch.relu(arr).to(torch.float16)


def prelu(arr_f16: Tensor, slope: Tensor) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    return torch.nn.functional.prelu(arr_f32, slope).to(torch.float16)


def gelu(arr_f16: Tensor) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    return torch.nn.functional.gelu(arr_f32).to(torch.float16)


def tanh(arr_f16: Tensor) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    return torch.tanh(arr_f32).to(torch.float16)


def sigmoid(arr_f16: Tensor) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    return torch.sigmoid(arr_f32).to(torch.float16)


def softmax(arr_f16: Tensor) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    return torch.nn.functional.softmax(arr_f32, dim=0).to(torch.float16)


def esum(arr_f16: Tensor, 
         max_val: float,
         init_accum: float) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    max_val_f32 = torch.as_tensor(max_val, dtype=torch.float32)
    init_accum_f32 = torch.as_tensor(init_accum, dtype=torch.float32)
    exp_arr = torch.exp(arr_f32 - max_val_f32)
    sum_exp = torch.sum(exp_arr)
    return (init_accum_f32 + sum_exp).to(torch.float16)


# ---- Pool --------------------------------------------------------------
def pool_max(arr_f16: Tensor, kernel_size: int) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    out_f32 = torch.max_pool1d(arr_f32, kernel_size=kernel_size, stride=kernel_size)
    return out_f32.to(torch.float16)


def pool_avg(arr_f16: Tensor, kernel_size: int) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    out = torch.avg_pool1d(arr_f32, kernel_size=kernel_size, stride=kernel_size, count_include_pad=False)
    return out.to(torch.float16) 


# ---- Cvt (9) ---------------------------------------------------------------
def cvt_qh(arr_f16: torch.Tensor, 
           scale: float,
           offset: float) -> torch.Tensor:
    """FP16 -> FP8. `a = a * scale + offset` """
    a_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (a_f32 * s_f32 + o_f32).to(torch.float8_e4m3fn)
    


def cvt_hq(arr_f8: torch.Tensor, 
           scale: float,
           offset: float) -> torch.Tensor:
    """FP8 -> FP16. Decode via LUT then `out = decoded * scale + offset`."""
    arr_f32 = arr_f8.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (arr_f32.to(torch.float16) * s_f32 + o_f32).to(torch.float16)
    


def cvt_ih(arr_f16: torch.Tensor, 
           scale: float,
           offset: float) -> torch.Tensor:
    """FP16 -> INT8. `int8(round(a * scale + offset))` saturating to [-128, 127].
    Source: gtx_npu_act.cc:288-297.
    """
    a_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return torch.clamp(torch.round(a_f32 * s_f32 + o_f32), -128, 127).to(torch.int8)


def cvt_hi(arr_f16: torch.Tensor, 
           scale: float,
           offset: float) -> torch.Tensor:
    """INT8 -> FP16. `out = int8 * scale + offset`. """
    arr_f32 = arr_f16.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    return (arr_f32 * s_f32 + o_f32).to(torch.float16)


def cvt_hn(arr_i32: torch.Tensor, 
           scale: float,
           offset: float) -> torch.Tensor:
    """INT32 -> FP16 normalize. `out = int32 * scale + offset`.
    Source: gtx_npu_act.cc:301-313.
    """
    arr_f32 = arr_i32.to(torch.float32)
    s_f32 = torch.as_tensor(scale, dtype=torch.float32)
    o_f32 = torch.as_tensor(offset, dtype=torch.float32)
    out_f32 = arr_f32 * s_f32 + o_f32
    return out_f32.to(torch.float16)


def cvt_sh(arr_f32: torch.Tensor) -> torch.Tensor:
    """FP32 -> FP16 (bit-pattern preserving; NO scale/offset). """
    return arr_f32.to(torch.float16)


def cvt_hs(arr_f16: torch.Tensor) -> torch.Tensor:
    """FP16 -> FP32 (bit-pattern preserving). Source: gtx_npu_act.cc:317-324."""
    a_f32 = arr_f16.to(torch.float32)
    return a_f32


def cvt_dh(arr_f64: torch.Tensor) -> torch.Tensor:
    """FP64 -> FP16 (bit-pattern preserving). Source: gtx_npu_act.cc:351-360.

    NumPy's `arr.astype(torch.float16)` does FP64->FP16 directly (single rounding).
    To preserve P5 byte-for-byte semantics, the wrapper performs the direct cast
    here; the `_cvt_dh_njit` is unused for the public path but registered for
    the parity test (Tier 1 compares public output to a separately-cast variant).
    """
    return arr_f64.to(torch.float16)


def cvt_hd(arr_f16: torch.Tensor) -> torch.Tensor:
    """FP16 -> FP64 (bit-pattern preserving). Source: gtx_npu_act.cc:342-349.

    NumPy's `arr.astype(torch.float64)` does FP16->FP64 directly (single widening,
    bit-exact). To preserve P5 byte-for-byte semantics, the wrapper performs the
    direct cast here.
    """
    return arr_f16.to(torch.float64)
