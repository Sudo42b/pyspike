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
"""Phase 1 acceptance: 65536 FP16 values round-trip exactly through fp.fp16_to_fp32 / fp32_to_fp16.

D-09 risk acknowledgment: NumPy 2.x np.float16 RNE may differ from C++ gtx_fp32_to_16
on subnormal/NaN payload/halfway-rounding edge cases. Phase 1 verifies the *helper-level*
round-trip; full strict-mode comparison vs C++ is deferred to P4/P5.
"""
import numpy as np

# pylint: disable=import-error,no-name-in-module
from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16


def test_all_65536_fp16_values_idempotent():
    """For every FP16 bit pattern x: fp32_to_fp16(fp16_to_fp32(x)) == x (bitwise)."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f16 = all_u16.view(np.float16)

    fp32 = fp16_to_fp32(all_f16)
    back_f16 = fp32_to_fp16(fp32)
    back_u16 = back_f16.view(np.uint16)

    # Empirically verified on NumPy 2.2.6 (cp310 on x86_64 LE):
    # ALL 65536 values round-trip exactly, including all 2046 NaN bit patterns.
    np.testing.assert_array_equal(back_u16, all_u16)


def test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern():
    """NaN inputs produce NaN outputs; bit pattern is preserved (NumPy 2.x behavior)."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f16 = all_u16.view(np.float16)
    nan_mask = np.isnan(all_f16)
    nan_count = int(nan_mask.sum())
    assert nan_count == 2046, f"Expected 2046 NaN bit patterns, got {nan_count}"

    back_u16 = fp32_to_fp16(fp16_to_fp32(all_f16)).view(np.uint16)
    # All NaN inputs produce NaN outputs:
    assert np.all(np.isnan(back_u16.view(np.float16)[nan_mask]))
    # Bit pattern is preserved (HIGH-confidence on NumPy 2.x):
    np.testing.assert_array_equal(back_u16[nan_mask], all_u16[nan_mask])


def test_subnormals_roundtrip():
    """All FP16 subnormals (exp == 0, mantissa != 0) round-trip exactly."""
    # FP16 subnormals: 0x0001..0x03FF and 0x8001..0x83FF
    subnormal_pos = np.arange(0x0001, 0x0400, dtype=np.uint16)
    subnormal_neg = np.arange(0x8001, 0x8400, dtype=np.uint16)
    subs = np.concatenate([subnormal_pos, subnormal_neg]).view(np.float16)

    back = fp32_to_fp16(fp16_to_fp32(subs)).view(np.uint16)
    expected = np.concatenate([subnormal_pos, subnormal_neg])
    np.testing.assert_array_equal(back, expected)


def test_negative_zero_preserved():
    """fp32_to_fp16(fp16_to_fp32(np.float16(-0.0))) preserves -0.0 (sign bit)."""
    neg_zero_u16 = np.array([0x8000], dtype=np.uint16)
    neg_zero_f16 = neg_zero_u16.view(np.float16)
    back = fp32_to_fp16(fp16_to_fp32(neg_zero_f16)).view(np.uint16)
    np.testing.assert_array_equal(back, neg_zero_u16)


def test_known_values():
    """Sanity-check known FP16 <-> FP32 conversions."""
    cases = [
        (np.float16(1.0), np.float32(1.0), 0x3C00),
        (np.float16(2.0), np.float32(2.0), 0x4000),
        (np.float16(0.5), np.float32(0.5), 0x3800),
        (np.float16(-1.0), np.float32(-1.0), 0xBC00),
    ]
    for f16, f32, raw in cases:
        assert fp16_to_fp32(f16) == f32
        assert fp32_to_fp16(f32) == f16
        assert int(fp32_to_fp16(f32).view(np.uint16)) == raw
