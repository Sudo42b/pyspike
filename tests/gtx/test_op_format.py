"""P5 format_cvt op unit tests -- Wave 0 RED scaffolds (test_op_format.py).

Covers ACT-04: 7 format_cvt directions (FP16<->FP8/INT8/INT32, FP32<->FP16,
FP64<->FP16) + scale/offset packing in GSPR_GTX_OPERAND2 + FP8 codec
verification (subnormal, exp=0xF, round-trip).

Wave 1b plan 04 GREEN-fills these. Plan 01 ships pytest.skip(...) bodies.
"""
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


# =========================================================================
# ACT-04: scale + offset packing in GSPR_GTX_OPERAND2
# =========================================================================
def test_scale_offset_packing():
    """ACT-04: scale (FP16) in low 16 bits, offset (FP16) in high 16 bits.
    Source: gtx_npu_act.cc:240-243. Pitfall 6."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: GSPR_OPERAND2 unpack scale+offset")


# =========================================================================
# ACT-04: FP8 codec verification
# =========================================================================
def test_fp8_roundtrip_identity():
    """ACT-04: For all 256 FP8 inputs, FP8->FP16->FP8 is identity (modulo
    NaN/subnormal collision classes). Bit-pair LUT consistency invariant."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: FP8_TO_FP16_LUT + FP16_TO_FP8_LUT")


def test_fp8_subnormal_decode():
    """ACT-04: FP8 subnormal (h_exp=0, h_frac in {1..7}) decodes as
    `(h_frac/8) * 2^-6 * (-1)^sign`. GTX uses 2^-6 base (NOT NVIDIA E4M3
    2^-9). Source: gtx_npu.h:154-179."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: FP8 subnormal LUT-builder")


def test_fp8_exp_max():
    """ACT-04: FP8 exp=0xF; h_frac=0 -> FP32 inf (with sign), h_frac>0 -> NaN.
    GTX has both inf and NaN; NVIDIA E4M3 has only NaN. Pitfall 5."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: FP8 exp=0xF inf/NaN semantics")


# =========================================================================
# ACT-04: 7 format_cvt directions
# =========================================================================
def test_int8_fp16_scale_offset():
    """ACT-04: INT8<->FP16 with scale/offset applied (a = a*sc + os).
    Source: gtx_npu_act.cc:277-297."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: cvt_ih + cvt_hi with scale/offset")


def test_int32_fp16_normalize():
    """ACT-04: INT32->FP16 normalize via SCVT_HN funct7=0x22 (1-direction).
    Source: gtx_npu_act.cc:301-313."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: cvt_hn INT32 normalize")


def test_fp32_fp16_no_scale():
    """ACT-04: FP32<->FP16 is bit-pattern preserving (no scale/offset).
    Source: gtx_npu_act.cc:317-335."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: cvt_sh + cvt_hs identity round-trip")


def test_fp64_fp16_no_scale():
    """ACT-04: FP64<->FP16 is bit-pattern preserving (RESEARCH Adjustment 1).
    Source: gtx_npu_act.cc:342-360."""
    pytest.skip("Wave 1b plan 04 GREEN-fills: cvt_dh + cvt_hd identity round-trip")
