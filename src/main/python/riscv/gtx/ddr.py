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
"""DDR backing store — lazy allocation (D-01) + GTX_DDR_SIZE env var parsing (D-02).

Phase 1 scope: ensure_ddr() lazy growth + cap parsing.
Phase 3 fills: ddr_init_from_file / ddr_dump_to_file (with GTX_DDR_REVERSED I/O — D-03).
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .memory import GtxMemory   # avoid circular import at runtime

# D-02 default: 4 GiB
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024


def get_ddr_cap() -> int:
    """Read GTX_DDR_SIZE env var; default 4GB. Supports 'G'/'M'/'K' suffixes.

    Examples: '4G' -> 4*1024**3, '64M' -> 64*1024**2, '1024K' -> 1024*1024.
    """
    val = os.environ.get("GTX_DDR_SIZE")
    if val is None:
        return DEFAULT_DDR_SIZE
    val = val.strip().upper()
    if val.endswith("G"):
        return int(val[:-1]) * 1024 ** 3
    if val.endswith("M"):
        return int(val[:-1]) * 1024 ** 2
    if val.endswith("K"):
        return int(val[:-1]) * 1024
    return int(val)


def ensure_ddr(mem: "GtxMemory", end_offset: int) -> np.ndarray:
    """Lazy DDR alloc. Phase 1 stub: allocates exactly end_offset (no doubling).

    D-01: DDR not pre-allocated at GtxMemory construction; first ensure_ddr() call
    materializes it.
    D-02: end_offset > GTX_DDR_SIZE cap -> ValueError (explicit, not silent truncation).

    Phase 3 will replace this with the C++ doubling-grow strategy matching
    gtx_npu_t::ensure_ddr.
    """
    cap = get_ddr_cap()
    if end_offset > cap:
        raise ValueError(
            f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
            f"(set GTX_DDR_SIZE env var to raise)"
        )
    if mem._ddr_bytes is None or end_offset > mem._ddr_bytes.size:
        new_size = max(
            end_offset,
            mem._ddr_bytes.size if mem._ddr_bytes is not None else 0,
        )
        new_arr = np.zeros(new_size, dtype=np.uint8)
        if mem._ddr_bytes is not None:
            new_arr[:mem._ddr_bytes.size] = mem._ddr_bytes
        mem._ddr_bytes = new_arr
    return mem._ddr_bytes
