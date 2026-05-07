"""P5 format_cvt op unit tests -- Wave 1b plan 04 GREEN.

Covers ACT-04: 7 format_cvt directions (FP16<->FP8/INT8/INT32, FP32<->FP16,
FP64<->FP16) + scale/offset packing in GSPR_GTX_OPERAND2 + FP8 codec
verification (subnormal, exp=0xF, round-trip).

Plan 04 lands act_core.cvt_* + FP8 LUTs + act_engine.firmware_format.
"""
import numpy as np
import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

# pylint: disable=import-error,wrong-import-position
from riscv.gtx import act_engine, act_core
from riscv.gtx.encoding import (
    GSPR_GTX_OPERAND2, GSPR_GTX_OPCODE,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRR,
)
from riscv.gtx.npu import GtxNpu

from tests.gtx._mocks import MockProcessor, MockInsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_npu():
    """GtxNpu with default L1 ADDRA/ADDRR offsets (avoid zero-collision)."""
    npu = GtxNpu()
    npu.lspr[0][0][LSPR_SPM_ADDRA] = 0x0000
    npu.lspr[0][0][LSPR_SPM_ADDRR] = 0x2000
    return npu


def _make_insn(*, rs1_idx: int = 1, rs2_idx: int = 2, rd_idx: int = 0,
                funct: int = 0, funct3: int = 0) -> MockInsn:
    xd = (funct3 >> 2) & 1
    xs1 = (funct3 >> 1) & 1
    xs2 = funct3 & 1
    return MockInsn(funct=funct, rs1=rs1_idx, rs2=rs2_idx, rd=rd_idx,
                    xd=xd, xs1=xs1, xs2=xs2)


def _pack_scale_offset(scale, offset) -> int:
    """Pack scale (FP16) into low 16 bits, offset (FP16) into high 16 bits."""
    s = int(np.float16(scale).view(np.uint16))
    o = int(np.float16(offset).view(np.uint16))
    return ((o & 0xFFFF) << 16) | (s & 0xFFFF)


# =========================================================================
# ACT-04: FP8 codec verification (LUT level)
# =========================================================================
def test_fp8_subnormal_decode():
    """ACT-04: FP8 subnormal (h_exp=0, h_frac in {1..7}) decodes as
    `(h_frac/8) * 2^-6 * (-1)^sign`. GTX uses 2^-6 base (NOT NVIDIA E4M3
    2^-9). Source: gtx_npu.h:154-179."""
    lut = act_core.FP8_TO_FP16_LUT
    assert lut.shape == (256,)
    assert lut.dtype == np.float16
    for sign in (0, 1):
        for frac in range(1, 8):
            byte = (sign << 7) | (0 << 3) | frac
            expected = ((-1) ** sign) * (frac / 8.0) * (2.0 ** -6)
            actual = float(lut[byte])
            assert abs(actual - expected) < 1e-9, (
                f"byte=0x{byte:02x} sign={sign} frac={frac}: "
                f"got {actual}, expected {expected}"
            )

    # Smallest subnormal (frac=1): exactly (1/8) * 2^-6 = 0.001953125
    assert float(lut[0x01]) == 0.001953125
    # Sign bit: 0x81 (sign=1, exp=0, frac=1) = -0.001953125
    assert float(lut[0x81]) == -0.001953125
    # Zero (sign=0, exp=0, frac=0)
    assert float(lut[0x00]) == 0.0
    # Negative zero (sign=1, exp=0, frac=0): bit pattern -0
    assert int(lut[0x80].view(np.uint16)) == 0x8000


def test_fp8_exp_max():
    """ACT-04: FP8 exp=0xF; h_frac=0 -> FP32 inf (with sign), h_frac>0 -> NaN.
    GTX has both inf and NaN; NVIDIA E4M3 has only NaN. Pitfall 5."""
    lut = act_core.FP8_TO_FP16_LUT
    # 0x78 = sign=0, exp=0xF, frac=0  -> +inf
    assert np.isinf(lut[0x78]) and lut[0x78] > 0
    # 0xF8 = sign=1, exp=0xF, frac=0  -> -inf
    assert np.isinf(lut[0xF8]) and lut[0xF8] < 0
    # 0x7F = sign=0, exp=0xF, frac=7  -> NaN
    assert np.isnan(lut[0x7F])
    # 0xFF = sign=1, exp=0xF, frac=7  -> NaN
    assert np.isnan(lut[0xFF])
    # All 0xF*-frac>0 patterns are NaN (parametrize over 0..7)
    for frac in range(1, 8):
        b_pos = (0 << 7) | (0xF << 3) | frac
        b_neg = (1 << 7) | (0xF << 3) | frac
        assert np.isnan(lut[b_pos]), f"0x{b_pos:02x} should be NaN"
        assert np.isnan(lut[b_neg]), f"0x{b_neg:02x} should be NaN"


def test_fp8_roundtrip_identity():
    """ACT-04: For all 256 FP8 inputs, FP8->FP16->FP8 is identity (modulo
    NaN equivalence classes). Bit-pair LUT consistency invariant."""
    fp8_to_fp16 = act_core.FP8_TO_FP16_LUT
    fp16_to_fp8 = act_core.FP16_TO_FP8_LUT
    assert fp16_to_fp8.shape == (65536,)
    assert fp16_to_fp8.dtype == np.uint8

    failures = []
    for byte in range(256):
        # Skip NaN bytes -- multiple FP16 NaNs may collapse to a single FP8 NaN sentinel.
        decoded = fp8_to_fp16[byte]
        if np.isnan(decoded):
            continue
        # Re-encode: take the FP16 bit pattern, look up FP8 byte
        decoded_u16 = int(decoded.view(np.uint16))
        re_encoded = int(fp16_to_fp8[decoded_u16])
        if re_encoded != byte:
            failures.append((byte, decoded_u16, re_encoded))

    assert not failures, f"Round-trip failures: {failures[:5]}"


# =========================================================================
# ACT-04: scale + offset packing in GSPR_GTX_OPERAND2 (Pitfall 6)
# =========================================================================
def test_scale_offset_packing():
    """ACT-04: scale (FP16) in low 16 bits, offset (FP16) in high 16 bits.
    Source: gtx_npu_act.cc:240-243. Pitfall 6.

    Use INT8->FP16 with scale=2.0, offset=0.5 to verify packing direction.
    If swapped (scale<->offset), result would be drastically different.
    """
    npu = _new_npu()
    proc = MockProcessor()

    # Input: int8 value 3 at L1[ADDRA]; should produce 3*2 + 0.5 = 6.5
    l1 = npu.mem.l1_byte(0, 0)
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]
    l1[addr_a] = np.uint8(np.int8(3).view(np.uint8))

    # Pack scale=2.0 (low 16), offset=0.5 (high 16)
    op2 = _pack_scale_offset(scale=np.float16(2.0), offset=np.float16(0.5))
    npu.gspr[GSPR_GTX_OPERAND2] = op2

    insn = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 1)  # length=1

    rc = act_engine.firmware_format(
        npu, proc, insn, src_kind='int8', dst_kind='fp16')
    assert rc == 0

    # Read FP16 result at ADDRR
    raw = int(l1[addr_r]) | (int(l1[addr_r + 1]) << 8)
    result_fp16 = np.array([raw], dtype=np.uint16).view(np.float16)[0]
    expected = np.float16(3 * 2.0 + 0.5)  # = 6.5
    assert result_fp16 == expected, (
        f"Got {result_fp16}, expected {expected}. "
        f"If swapped (offset=2.0 scale=0.5): would be 3*0.5 + 2.0 = 3.5 -- different!"
    )


# =========================================================================
# ACT-04: 7 format_cvt directions
# =========================================================================
def test_int8_fp16_scale_offset():
    """ACT-04: INT8<->FP16 with scale/offset applied (a = a*sc + os).
    Source: gtx_npu_act.cc:277-297."""
    npu = _new_npu()
    proc = MockProcessor()

    # INT8->FP16 (scvt_hi): input [3, -2, 5, 0]; scale=1.0, offset=0.0
    inp = np.array([3, -2, 5, 0], dtype=np.int8)
    l1 = npu.mem.l1_byte(0, 0)
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]
    l1[addr_a:addr_a + 4] = inp.view(np.uint8)

    npu.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(1.0), np.float16(0.0))
    insn = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 4)  # length=4

    rc = act_engine.firmware_format(
        npu, proc, insn, src_kind='int8', dst_kind='fp16')
    assert rc == 0

    # 4 FP16 outputs
    raw_bytes = bytes(l1[addr_r:addr_r + 8])
    result = np.frombuffer(raw_bytes, dtype=np.float16)
    assert (result == np.array([3.0, -2.0, 5.0, 0.0], dtype=np.float16)).all()

    # FP16 -> INT8 (scvt_ih): round + clip [-128, 127]
    npu2 = _new_npu()
    l1b = npu2.mem.l1_byte(0, 0)
    addr_a2 = npu2.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r2 = npu2.lspr[0][0][LSPR_SPM_ADDRR]
    arr = np.array([1.7, -2.3, 200.0, -200.0], dtype=np.float16)
    l1b[addr_a2:addr_a2 + 8] = np.frombuffer(arr.tobytes(), dtype=np.uint8)
    npu2.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(1.0), np.float16(0.0))
    insn2 = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 4)
    rc2 = act_engine.firmware_format(
        npu2, proc, insn2, src_kind='fp16', dst_kind='int8')
    assert rc2 == 0
    out_i8 = np.frombuffer(bytes(l1b[addr_r2:addr_r2 + 4]), dtype=np.int8)
    # round-half-even default in numpy: round(1.7)=2, round(-2.3)=-2
    assert out_i8[0] == 2
    assert out_i8[1] == -2
    assert out_i8[2] == 127  # clamp
    assert out_i8[3] == -128


def test_int32_fp16_normalize():
    """ACT-04: INT32->FP16 normalize via SCVT_HN funct7=0x22 (1-direction).
    Source: gtx_npu_act.cc:301-313.

    `result = (int32 * scale + offset)` in FP32 then cast to FP16.
    """
    npu = _new_npu()
    proc = MockProcessor()
    l1 = npu.mem.l1_byte(0, 0)
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]

    # Input: int32 [1000, -500]; scale=0.001, offset=0.0 -> FP16 [1.0, -0.5]
    inp = np.array([1000, -500], dtype=np.int32)
    l1[addr_a:addr_a + 8] = np.frombuffer(inp.tobytes(), dtype=np.uint8)

    npu.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(0.001), np.float16(0.0))
    insn = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 2)  # length=2

    rc = act_engine.firmware_format(
        npu, proc, insn, src_kind='int32', dst_kind='fp16')
    assert rc == 0

    out = np.frombuffer(bytes(l1[addr_r:addr_r + 4]), dtype=np.float16)
    # FP16(0.001) is not exactly 0.001; result should be FP16-representable
    # of (int32 * np.float32(np.float16(0.001))) + 0.0
    sc_f32 = np.float32(np.float16(0.001))
    expected = np.array([1000 * sc_f32, -500 * sc_f32], dtype=np.float32).astype(np.float16)
    assert (out == expected).all(), f"Got {out}, expected {expected}"


def test_fp32_fp16_no_scale():
    """ACT-04: FP32<->FP16 is bit-pattern preserving (no scale/offset).
    Source: gtx_npu_act.cc:317-335.

    Verify scale/offset are IGNORED for these directions.
    """
    npu = _new_npu()
    proc = MockProcessor()
    l1 = npu.mem.l1_byte(0, 0)
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]

    # FP32 -> FP16
    inp_f32 = np.array([1.5, -2.5, 0.25, 100.0], dtype=np.float32)
    l1[addr_a:addr_a + 16] = np.frombuffer(inp_f32.tobytes(), dtype=np.uint8)

    # Set scale/offset to non-trivial values that WOULD distort if applied.
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(99.0), np.float16(50.0))

    insn = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 4)

    rc = act_engine.firmware_format(
        npu, proc, insn, src_kind='fp32', dst_kind='fp16')
    assert rc == 0

    out_fp16 = np.frombuffer(bytes(l1[addr_r:addr_r + 8]), dtype=np.float16)
    # bit-pattern preserving: same as direct cast, scale/offset ignored.
    expected = inp_f32.astype(np.float16)
    assert (out_fp16 == expected).all(), (
        f"FP32->FP16 should be bit-pattern preserving. Got {out_fp16}, "
        f"expected {expected}. Scale=99/offset=50 must NOT be applied."
    )

    # FP16 -> FP32 (sub_op&1 not used at this engine API; we pass src='fp16' dst='fp32')
    npu2 = _new_npu()
    l1b = npu2.mem.l1_byte(0, 0)
    addr_a2 = npu2.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r2 = npu2.lspr[0][0][LSPR_SPM_ADDRR]
    inp_fp16 = np.array([1.5, -2.5, 0.25, 100.0], dtype=np.float16)
    l1b[addr_a2:addr_a2 + 8] = np.frombuffer(inp_fp16.tobytes(), dtype=np.uint8)
    npu2.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(99.0), np.float16(50.0))
    insn2 = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 4)
    rc2 = act_engine.firmware_format(
        npu2, proc, insn2, src_kind='fp16', dst_kind='fp32')
    assert rc2 == 0
    out_fp32 = np.frombuffer(bytes(l1b[addr_r2:addr_r2 + 16]), dtype=np.float32)
    expected_fp32 = inp_fp16.astype(np.float32)
    assert (out_fp32 == expected_fp32).all()


def test_fp64_fp16_no_scale():
    """ACT-04: FP64<->FP16 is bit-pattern preserving (RESEARCH Adjustment 1).
    Source: gtx_npu_act.cc:342-360."""
    npu = _new_npu()
    proc = MockProcessor()
    l1 = npu.mem.l1_byte(0, 0)
    addr_a = npu.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r = npu.lspr[0][0][LSPR_SPM_ADDRR]

    # FP64 -> FP16: input [1.5, -2.5]
    inp_f64 = np.array([1.5, -2.5], dtype=np.float64)
    l1[addr_a:addr_a + 16] = np.frombuffer(inp_f64.tobytes(), dtype=np.uint8)
    # Non-trivial scale/offset that MUST NOT be applied (bit-pattern preserving)
    npu.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(99.0), np.float16(50.0))

    insn = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 2)

    rc = act_engine.firmware_format(
        npu, proc, insn, src_kind='fp64', dst_kind='fp16')
    assert rc == 0

    out_fp16 = np.frombuffer(bytes(l1[addr_r:addr_r + 4]), dtype=np.float16)
    expected = inp_f64.astype(np.float16)
    assert (out_fp16 == expected).all(), (
        f"FP64->FP16 should be bit-pattern preserving. Got {out_fp16}, "
        f"expected {expected}. Scale=99/offset=50 must NOT be applied."
    )

    # FP16 -> FP64
    npu2 = _new_npu()
    l1b = npu2.mem.l1_byte(0, 0)
    addr_a2 = npu2.lspr[0][0][LSPR_SPM_ADDRA]
    addr_r2 = npu2.lspr[0][0][LSPR_SPM_ADDRR]
    inp_fp16 = np.array([1.5, -2.5], dtype=np.float16)
    l1b[addr_a2:addr_a2 + 4] = np.frombuffer(inp_fp16.tobytes(), dtype=np.uint8)
    npu2.gspr[GSPR_GTX_OPERAND2] = _pack_scale_offset(np.float16(99.0), np.float16(50.0))
    insn2 = _make_insn(rs1_idx=1)
    proc.state.XPR.write(1, 2)
    rc2 = act_engine.firmware_format(
        npu2, proc, insn2, src_kind='fp16', dst_kind='fp64')
    assert rc2 == 0
    out_fp64 = np.frombuffer(bytes(l1b[addr_r2:addr_r2 + 16]), dtype=np.float64)
    expected_fp64 = inp_fp16.astype(np.float64)
    assert (out_fp64 == expected_fp64).all()
