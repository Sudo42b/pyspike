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
"""Torch-backed memory layer for GTX NPU.

Migrated 2026-05-11 from numpy → torch (CUDA-optional). Memory tensors live
on CPU; compute kernels may temporarily move slices to GPU. CPU layout keeps
byte-level DMA access cheap (no host↔device sync per instruction).

D-10: Layered API. Both raw byte views and named halfword accessors.
D-11: SPR unified dict[int, int].
D-12: Every named accessor returns a non-copying view (tensor._base is not None).
D-01: DDR is lazily allocated (see ddr.py:ensure_ddr); _ddr_bytes is None initially.
"""
from typing import Optional

import torch

from .params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
)


class GtxMemory:
    """GTX NPU memory layer — L0/L1/L2 torch tensor + DDR lazy alloc + SPR dict."""

    def __init__(self) -> None:
        self._l0_bytes: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=torch.uint8
        )
        self._l1_bytes: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=torch.uint8
        )
        self._l2_bytes: torch.Tensor = torch.zeros(
            (GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=torch.uint8
        )
        self.spr: dict[int, int] = {}
        self._ddr_bytes: Optional[torch.Tensor] = None

    # ----- Raw byte views (D-10 low-level) -----

    def l0_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self._l0_bytes[nest, spu]

    def l1_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self._l1_bytes[nest, spu]

    def l2_byte(self, nest: int) -> torch.Tensor:
        return self._l2_bytes[nest]

    # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----

    def l0_f16(self, nest: int, spu: int) -> torch.Tensor:
        # D-12: torch view(dtype) shares storage (no copy) even though
        # tensor._base is None for dtype-reinterpret views.
        return self._l0_bytes[nest, spu].view(torch.float16)

    def l1_f16(self, nest: int, spu: int) -> torch.Tensor:
        return self._l1_bytes[nest, spu].view(torch.float16)

    def l2_f16(self, nest: int) -> torch.Tensor:
        return self._l2_bytes[nest].view(torch.float16)

    # ----- Halfword uint16 view (rare) -----

    def l1_u16(self, nest: int, spu: int) -> torch.Tensor:
        return self._l1_bytes[nest, spu].view(torch.uint16)
