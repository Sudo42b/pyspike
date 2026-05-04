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
"""FP16 / FP32 conversion helpers - D-09 (np.float16 view via astype, NOT bit manipulation).

NumPy 2.x guarantees IEEE 754 binary16 RNE for astype(np.float16). All 65536 FP16
values round-trip exactly; NaN bit patterns are preserved (empirically verified on
NumPy 2.2.6, see tests/gtx/test_fp_roundtrip.py).

Risk acknowledgment (D-09): subnormal/NaN payload/halfway-rounding edge cases vs C++
gtx_fp32_to_16 are deferred to P4/P5 strict-mode measurement. If discrepancies arise,
fp_strict.py (bit-manipulation port of gtx_npu.h:89-151) will be added as fallback.
"""
from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, np.float16, np.float32, float]


def fp16_to_fp32(x: ArrayLike) -> np.ndarray:
    """Widen FP16 -> FP32. Lossless (widening cast).

    Note: returns a NEW array (astype always copies). Caller MUST NOT expect
    base preservation - D-12 (view-base invariant) applies to memory accessors,
    not to FP conversion helpers.

    For zero-copy FP32 reduction over FP16 storage, use mem.l1_f16(...) directly
    and pass to NumPy reductions with dtype=np.float32 keyword (Phases 4/5 pattern).
    """
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def fp32_to_fp16(x: ArrayLike) -> np.ndarray:
    """Narrow FP32 -> FP16 with IEEE 754 binary16 RNE (NumPy 2.x default).

    Empirically verified on NumPy 2.2.6: idempotent for all 65536 FP16 values
    (including NaN bit-pattern preservation, subnormals, negative zero).
    """
    return np.asarray(x, dtype=np.float32).astype(np.float16)
