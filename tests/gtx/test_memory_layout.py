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
"""Phase 1 acceptance: GtxMemory layout invariants.

D-17: writing 0x3C00 to halfword view at L1[nest=0, spu=0, off=0] produces bytes
      [0x00, 0x3C] LE.
D-12: every named accessor returns a non-copying view (arr.base is not None).
D-11: mem.spr is a unified dict[int, int].
D-01: DDR is None at construction (lazy alloc).
"""
import numpy as np
import pytest

from riscv.gtx.memory import GtxMemory
from riscv.gtx.params import (
    GTX_NEST_NUM,
    GTX_SPU_NUM,
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
)


@pytest.fixture
def mem():
    return GtxMemory()


def test_le_byte_order_via_byte_write(mem):
    """Writing LE bytes [0x00, 0x3C] to L1 byte view appears as np.float16(1.0) in fp16 view."""
    mem.l1_byte(0, 0)[0] = 0x00
    mem.l1_byte(0, 0)[1] = 0x3C
    assert mem.l1_f16(0, 0)[0] == np.float16(1.0)


def test_le_byte_order_via_fp16_write(mem):
    """Writing np.float16(2.0) to L1 fp16 view produces LE bytes [0x00, 0x40]."""
    mem.l1_f16(0, 0)[0] = np.float16(2.0)
    assert mem.l1_byte(0, 0)[0] == 0x00
    assert mem.l1_byte(0, 0)[1] == 0x40


def test_l1_f16_view_invariant(mem):
    """D-12: l1_f16 returns a view, not a copy."""
    view = mem.l1_f16(0, 0)
    assert view.base is not None, "l1_f16 must return a view (D-12)"
    assert view.shape == (GTX_L1_SIZE_BYTES // 2,)
    assert view.dtype == np.float16


def test_l0_f16_view_invariant(mem):
    """D-12: l0_f16 returns a view, not a copy."""
    view = mem.l0_f16(0, 0)
    assert view.base is not None
    assert view.shape == (GTX_L0_SIZE_BYTES // 2,)


def test_slice_preserves_base(mem):
    """Slicing an fp16 view preserves base (no copy on slice)."""
    view = mem.l1_f16(0, 0)
    sub = view[100:200]
    assert sub.base is not None, "slice of view must remain a view"


def test_l1_shape(mem):
    """L1 dimensions match HW parameters."""
    assert mem.l1_byte(0, 0).shape == (GTX_L1_SIZE_BYTES,)
    assert mem.l1_byte(0, 0).dtype == np.uint8
    for n in range(GTX_NEST_NUM):
        for s in range(GTX_SPU_NUM):
            assert mem.l1_byte(n, s).shape == (GTX_L1_SIZE_BYTES,)


def test_spr_dict(mem):
    """D-11: mem.spr is a unified dict[int, int]."""
    assert isinstance(mem.spr, dict)
    assert len(mem.spr) == 0
    mem.spr[0x100] = 0xCAFE     # GSPR range
    mem.spr[0x500] = 0xBABE     # NSPR range
    mem.spr[0x900] = 0xF00D     # LSPR range
    assert mem.spr[0x100] == 0xCAFE
    assert mem.spr[0x500] == 0xBABE
    assert mem.spr[0x900] == 0xF00D


def test_ddr_lazy_allocation(mem):
    """D-01: DDR is None at construction."""
    assert mem._ddr_bytes is None
