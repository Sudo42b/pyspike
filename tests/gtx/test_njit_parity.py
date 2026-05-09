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
"""P7 NJIT-05 Tier 1: per-kernel ULP-0 parity (NumPy oracle vs JIT).

Plan 02/03/04 will GREEN-fill -- each plan completes one module's `_impl`
aliases. Wave 0 leaves all 28 parametrize invocations as `pytest.skip(...)`.
"""
from __future__ import annotations
import pytest

from ._njit_helpers import (
    ALL_NJIT_KERNEL_NAMES,
)


@pytest.mark.parametrize("kernel_name", ALL_NJIT_KERNEL_NAMES)
def test_kernel_parity(kernel_name: str) -> None:
    """ULP-0 parity: NumPy(public_fn) vs JIT(_impl).

    Plan 02 (gemm), 03 (vec), 04 (act) GREEN-fills `_impl` aliases and
    replaces the body with:

        numpy_out = get_public_fn(kernel_name)(*inputs_fp16)
        njit_out  = (FP16-cast wrapper around get_impl_fn)(*inputs_fp32)
        assert np.array_equal(numpy_out.view(np.uint16), njit_out.view(np.uint16))
    """
    pytest.skip(f"Plan 02/03/04 GREEN-fills parity body for {kernel_name}")


def test_has_numba_detection(_numba_available: bool) -> None:
    """NJIT-01 sentinel: HAS_NUMBA reflects environment numba presence."""
    from riscv.gtx._jit import HAS_NUMBA
    assert HAS_NUMBA == _numba_available, (
        f"HAS_NUMBA={HAS_NUMBA} but _numba_available={_numba_available}"
    )
