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
"""Pure stateless VEC kernels -- direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc.

Per CONTEXT D-01 / D-03: this module has NO `npu`/`proc`/`insn` dependencies.
Array-in / scalar-in / array-out / scalar-out only. P7 numba @njit boundary.

Per RESEARCH Pitfall 2: every reduction (VSUM/DOT) MUST upcast FP16 -> FP32
explicitly with Python `for` loop. NEVER `np.sum(x, dtype=np.float32)` or `np.dot`
(both use pairwise summation; ULP-different from C++ scalar accumulate).

Phase 5 plan 02 fills the GREEN bodies. Plan 01 ships only NotImplementedError stubs.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def sasmd_kernel(a: NDArray[np.float16], b: NDArray[np.float16],
                 op: int) -> NDArray[np.float16]:
    """SASMD element-wise: out[i] = a[i] OP b[i] for OP in {add,sub,mul,div}.

    Plan 02 Wave 1b will fill with FP32-internal compute + single FP16 cast.
    Source: gtx_npu_vec.cc:50+ exec_vector_op switch.
    """
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def dot_kernel(a: NDArray[np.float16], b: NDArray[np.float16]) -> np.float16:
    """FP16 dot product: explicit Python for-loop FP32 accumulate.

    Plan 02 Wave 1b will fill (gtx_npu_vec.cc:251-262 line-for-line).
    """
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def vsum_kernel(view: NDArray[np.float16]) -> np.float16:
    """FP16 vector sum: explicit Python for-loop FP32 accumulate + single FP16 cast.

    Plan 02 Wave 1b will fill (gtx_npu_vec.cc:102-112 line-for-line).
    """
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def clamp_min_kernel(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = max(a[i], scalar). Plan 02 wave 1b GREEN.
    Source: gtx_npu_vec.cc:233-242.
    """
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def clamp_max_kernel(a: NDArray[np.float16], scalar: np.float16) -> NDArray[np.float16]:
    """out[i] = min(a[i], scalar). Source: gtx_npu_vec.cc:223-231."""
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def accum_kernel(a: NDArray[np.float16]) -> NDArray[np.float16]:
    """Prefix sum (cumulative). gtx_npu_vec.cc:215-221."""
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")


def arange_kernel(n: int, start: np.float16, step: np.float16) -> NDArray[np.float16]:
    """out[i] = start + i * step. gtx_npu_vec.cc:243-249."""
    raise NotImplementedError("Plan 02 wave 1b GREEN-fill")
