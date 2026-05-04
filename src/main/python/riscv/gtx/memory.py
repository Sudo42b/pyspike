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
"""NumPy-backed memory layer for GTX NPU.

D-10: Layered API. Both raw byte views and named halfword accessors.
D-11: SPR unified dict[int, int]. GSPR/NSPR/LSPR routing by address (P2 SPR-01).
D-12: Every named accessor returns a non-copying view (arr.base is not None).
D-01: DDR is lazily allocated (see ddr.py:ensure_ddr); _ddr_bytes is None initially.
"""
from typing import Optional

import numpy as np

from .params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
)


class GtxMemory:
    """GTX NPU memory layer — L0/L1/L2 ndarray + DDR lazy alloc + SPR dict."""

    def __init__(self) -> None:
        self._l0_bytes: np.ndarray = np.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=np.uint8
        )
        self._l1_bytes: np.ndarray = np.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=np.uint8
        )
        self._l2_bytes: np.ndarray = np.zeros(
            (GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=np.uint8
        )
        self.spr: dict[int, int] = {}
        self._ddr_bytes: Optional[np.ndarray] = None

    # ----- Raw byte views (D-10 low-level) -----

    def l0_byte(self, nest: int, spu: int) -> np.ndarray:
        return self._l0_bytes[nest, spu]

    def l1_byte(self, nest: int, spu: int) -> np.ndarray:
        return self._l1_bytes[nest, spu]

    def l2_byte(self, nest: int) -> np.ndarray:
        return self._l2_bytes[nest]

    # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----

    def l0_f16(self, nest: int, spu: int) -> np.ndarray:
        view = self._l0_bytes[nest, spu].view(np.float16)
        assert view.base is not None  # D-12 tripwire
        return view

    def l1_f16(self, nest: int, spu: int) -> np.ndarray:
        view = self._l1_bytes[nest, spu].view(np.float16)
        assert view.base is not None
        return view

    def l2_f16(self, nest: int) -> np.ndarray:
        view = self._l2_bytes[nest].view(np.float16)
        assert view.base is not None
        return view

    # ----- Halfword uint16 view (rare) -----

    def l1_u16(self, nest: int, spu: int) -> np.ndarray:
        view = self._l1_bytes[nest, spu].view(np.uint16)
        assert view.base is not None
        return view
