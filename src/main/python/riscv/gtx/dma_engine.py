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
"""DMA engine -- spike-independent DMA helpers + DeferredDdrStore + decoder.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc:25-435.

Per CONTEXT D-01: This module has NO `proc`/`insn` dependencies. It operates
on `GtxMemory` instances and pure data only -- the spike-dependent entry
points (firmware_dma @handler) live in `ops/dma.py` (Plan 02). Plans 02/04/05
all import from this module.

Phase 3 plan 01 Task 2.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .params import (
    GTX_NEST_NUM, GTX_SPU_NUM,
    GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES,
    GTX_DDR_BASE,
)
from .ddr import ensure_ddr  # Phase 3 plan 03 upgrades to doubling-grow

if TYPE_CHECKING:
    from .memory import GtxMemory


# ============================================================================
# DeferredDdrStore -- frozen dataclass, exact 7 fields per gtx_npu.h:1257-1266
# ============================================================================
@dataclass(frozen=True)
class DeferredDdrStore:
    """S-loop deferred L2->DDR store request.

    Direct port of `deferred_ddr_store_t` (gtx_npu.h:1257-1266). Field order
    is locked -- do not reorder. Pitfall 4 in 03-RESEARCH.md: adding/removing
    fields silently breaks producer/consumer (firmware_dma push site /
    flush_deferred_ddr_stores).
    """
    nest: int
    l2_off: int
    ddr_off: int
    length: int
    height: int
    l2_stride: int
    ddr_stride: int


# ============================================================================
# decode_firmware_dma_args -- direct port of gtx_npu_dma.cc:262-288
# ============================================================================
def decode_firmware_dma_args(rs1: int, rs2: int, rs3: int,
                              *, xd: int, xs1: int, xs2: int) -> dict:
    """Decode packed firmware_dma rs1/rs2/rs3 -> field dict.

    Pitfall 1 (is_copy carve-out): COPY funct3=010 uses rs1>>32 (32-bit dst)
    instead of (rs1>>27) & 0x1FFFFFFFFF (37-bit hi field). L1 is 384 KB =
    19 bits, fits within 27. C++ COPY path is L1->L1, both addrs in low half.

    Pitfall 2 (HW conventions): length_raw=0 -> 65536, height_raw=0 -> 1.

    Returns dict with keys: addr_hi, addr_lo, height, length, rd_stride,
    wr_stride, is_store, is_copy, funct3.
    """
    funct3 = (xd << 2) | (xs1 << 1) | xs2
    is_store = bool(funct3 & 1)
    is_copy = (not is_store) and bool(funct3 & 2)
    addr_hi = (rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)
    addr_lo = rs1 & 0x7FFFFFF
    height_raw = (rs2 >> 48) & 0xFFFF
    length_raw = (rs2 >> 32) & 0xFFFF
    rs2_low = rs2 & 0xFFFFFFFF
    rs3_low = rs3 & 0xFFFFFFFF

    height = 1 if height_raw == 0 else height_raw
    length = 0x10000 if length_raw == 0 else length_raw

    if is_store:
        wr_stride, rd_stride = rs2_low, rs3_low
    else:
        rd_stride, wr_stride = rs2_low, rs3_low

    return dict(addr_hi=addr_hi, addr_lo=addr_lo, height=height, length=length,
                rd_stride=rd_stride, wr_stride=wr_stride,
                is_store=is_store, is_copy=is_copy, funct3=funct3)


# ============================================================================
# exec_dma_2d -- direct port of gtx_npu_dma.cc:25-90
# ============================================================================
def exec_dma_2d(mem: 'GtxMemory', *, nest_id: int, l2_addr: int, l1_addr: int,
                width: int, height: int, is_load: bool,
                l2_stride: int = 0, spu_id: int = 0) -> int:
    """Strided 2D DMA between NEST L2 and SPU L1. Returns 0 (cycles vestigial).

    Note: `ctx` arg from C++ dropped -- was only for trace logging.
    Default l2_stride=0 means contiguous (l2_stride := width).
    """
    if nest_id >= GTX_NEST_NUM:
        return 0
    if width == 0 or height == 0:
        return 0

    if l2_stride == 0:
        l2_stride = width

    l1_buf = mem.l1_byte(nest_id, spu_id)
    l2_buf = mem.l2_byte(nest_id)

    for row in range(height):
        l2_off = (l2_addr + row * l2_stride) % GTX_L2_SIZE_BYTES
        l1_off = (l1_addr + row * width) % GTX_L1_SIZE_BYTES
        copy_len = min(width,
                        GTX_L2_SIZE_BYTES - l2_off,
                        GTX_L1_SIZE_BYTES - l1_off)
        if copy_len <= 0:
            continue
        if is_load:
            l1_buf[l1_off : l1_off + copy_len] = l2_buf[l2_off : l2_off + copy_len]
        else:
            l2_buf[l2_off : l2_off + copy_len] = l1_buf[l1_off : l1_off + copy_len]
    return 0


# ============================================================================
# exec_load_svr -- direct port of gtx_npu_dma.cc:97-113
# ============================================================================
def exec_load_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                   l1_addr: int, l0_reg: int) -> None:
    """L1 -> L0 transfer (32 bytes = one SVR register, 8 x 4-byte words)."""
    if nest_id >= GTX_NEST_NUM or spu_id >= GTX_SPU_NUM:
        return
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % GTX_L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32

    for j in range(8):
        src = (l1_off + j * 4) % GTX_L1_SIZE_BYTES
        dst = (l0_off + j * 4) % GTX_L0_SIZE_BYTES
        l0_buf[dst : dst + 4] = l1_buf[src : src + 4]


# ============================================================================
# exec_store_svr -- direct port of gtx_npu_dma.cc:118-136
# ============================================================================
def exec_store_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                    l1_addr: int, l0_reg: int) -> None:
    """L0 -> L1 transfer (32 bytes = one SVR register, 8 x 4-byte words)."""
    if nest_id >= GTX_NEST_NUM or spu_id >= GTX_SPU_NUM:
        return
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % GTX_L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32

    for j in range(8):
        src = (l0_off + j * 4) % GTX_L0_SIZE_BYTES
        dst = (l1_off + j * 4) % GTX_L1_SIZE_BYTES
        l1_buf[dst : dst + 4] = l0_buf[src : src + 4]


# ============================================================================
# exec_transpose -- direct port of gtx_npu_dma.cc:143-167
# ============================================================================
def exec_transpose(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                    rows: int, cols: int, addr_a: int, addr_r: int) -> int:
    """Matrix transpose in L1 (FP16, 2 bytes per elem).

    Note: addr_a / addr_r are passed as args (caller reads from
    spu.lspr[LSPR_SPM_ADDRA/R]) -- keeps this helper spike-independent.
    """
    if nest_id >= GTX_NEST_NUM or spu_id >= GTX_SPU_NUM:
        return 0
    if rows == 0 or cols == 0:
        return 0

    l1 = mem.l1_byte(nest_id, spu_id)
    for i in range(rows):
        for j in range(cols):
            s_off = (addr_a + (j + cols * i) * 2) % GTX_L1_SIZE_BYTES
            d_off = (addr_r + (i + rows * j) * 2) % GTX_L1_SIZE_BYTES
            l1[d_off : d_off + 2] = l1[s_off : s_off + 2].copy()
    return 0


# ============================================================================
# exec_transpose_ddr -- direct port of gtx_npu_dma.cc:175-225
# ============================================================================
def exec_transpose_ddr(mem: 'GtxMemory', *, src_addr: int, dst_addr: int,
                        dim2: int, dim1: int, dim0: int,
                        p2: int, p1: int, p0: int) -> None:
    """DDR-to-DDR 3D tensor transpose/permute (FP16).

    Reads [dim2][dim1][dim0] from src_addr, writes permuted to dst_addr.
    Permutation (p2,p1,p0): new dim k uses index from old dim[pk].
    """
    src_off = (src_addr - GTX_DDR_BASE) if src_addr >= GTX_DDR_BASE else src_addr
    dst_off = (dst_addr - GTX_DDR_BASE) if dst_addr >= GTX_DDR_BASE else dst_addr

    # HW convention: dim==0 -> 1
    if dim2 == 0:
        dim2 = 1
    if dim1 == 0:
        dim1 = 1
    if dim0 == 0:
        dim0 = 1

    old_dims = [dim0, dim1, dim2]
    new_dim1 = old_dims[p1]
    new_dim0 = old_dims[p0]

    old_s1 = dim0
    old_s2 = dim1 * dim0
    new_s1 = new_dim0
    new_s2 = new_dim1 * new_dim0

    nelem = dim2 * dim1 * dim0
    max_off = max(src_off + nelem * 2, dst_off + nelem * 2)
    ensure_ddr(mem, max_off)
    cap = mem._ddr_bytes.size  # type: ignore[union-attr]

    for i2 in range(dim2):
        for i1 in range(dim1):
            for i0 in range(dim0):
                src_idx = i2 * old_s2 + i1 * old_s1 + i0
                oi = [i0, i1, i2]
                dst_idx = oi[p2] * new_s2 + oi[p1] * new_s1 + oi[p0]
                s = src_off + src_idx * 2
                d = dst_off + dst_idx * 2
                if s + 1 < cap and d + 1 < cap:
                    mem._ddr_bytes[d : d + 2] = mem._ddr_bytes[s : s + 2].copy()  # type: ignore[union-attr]


# ============================================================================
# exec_fill -- direct port of gtx_npu_dma.cc:230-246
# ============================================================================
def exec_fill(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
               length: int, fill_val: int, addr_r: int) -> int:
    """Fill L1 region at addr_r with constant FP16 value (length elems x 2 bytes)."""
    if nest_id >= GTX_NEST_NUM or spu_id >= GTX_SPU_NUM:
        return 0

    l1 = mem.l1_byte(nest_id, spu_id)
    for i in range(length):
        off = (addr_r + i * 2) % GTX_L1_SIZE_BYTES
        l1[off] = fill_val & 0xFF
        l1[off + 1] = (fill_val >> 8) & 0xFF
    return 0


# ============================================================================
# firmware_dma_sloop_store -- direct port of gtx_npu_dma.cc:319-326
# ============================================================================
def firmware_dma_sloop_store(npu: Any, *, nest: int, addr_hi: int, addr_lo: int,
                              length: int, height: int,
                              rd_stride: int, wr_stride: int) -> int:
    """Push a DeferredDdrStore onto npu.deferred_ddr_stores (S-loop STORE branch).

    npu must expose `deferred_ddr_stores` as a list. Real flush happens later
    at end_p / credit_st_chk (Plan 05).
    """
    ddr_off = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi
    npu.deferred_ddr_stores.append(DeferredDdrStore(
        nest=nest,
        l2_off=addr_lo,
        ddr_off=ddr_off,
        length=length,
        height=height,
        l2_stride=rd_stride,
        ddr_stride=wr_stride,
    ))
    return 0


# ============================================================================
# firmware_dma_sloop_load -- direct port of gtx_npu_dma.cc:294-318 (LOAD branch)
# ============================================================================
def firmware_dma_sloop_load(mem: 'GtxMemory', *, nest: int, addr_hi: int, addr_lo: int,
                              length: int, height: int,
                              rd_stride: int, wr_stride: int) -> int:
    """S-loop LOAD: DDR -> L2 immediate row-by-row memcpy."""
    ddr_off_base = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi
    # Compute max DDR offset touched -- ensure_ddr once for whole copy
    max_off = ddr_off_base + (height - 1) * rd_stride + length
    ensure_ddr(mem, max_off)
    ddr = mem._ddr_bytes  # type: ignore[union-attr]

    l2_buf = mem.l2_byte(nest)
    for row in range(height):
        ddr_off = ddr_off_base + row * rd_stride
        l2_off = (addr_lo + row * wr_stride) % GTX_L2_SIZE_BYTES
        copy_len = min(length,
                        ddr.size - ddr_off,
                        GTX_L2_SIZE_BYTES - l2_off)
        if copy_len <= 0:
            continue
        l2_buf[l2_off : l2_off + copy_len] = ddr[ddr_off : ddr_off + copy_len]
    return 0


# ============================================================================
# firmware_dma_tloop_load_store -- direct port of gtx_npu_dma.cc:349-391
# ============================================================================
def firmware_dma_tloop_load_store(mem: 'GtxMemory', *, nest: int, spu: int,
                                    is_store: bool,
                                    addr_hi: int, addr_lo: int,
                                    length: int, height: int,
                                    rd_stride: int, wr_stride: int) -> int:
    """T-loop L2 <-> L1 strided per-row.

    LOAD: L2[addr_hi + row*rd_stride] -> L1[addr_lo + row*length]
    STORE: L1[addr_lo + row*length] -> L2[addr_hi + row*wr_stride]
    """
    l1 = mem.l1_byte(nest, spu)
    l2 = mem.l2_byte(nest)

    for row in range(height):
        if not is_store:
            hi_off = (addr_hi + row * rd_stride) % GTX_L2_SIZE_BYTES
        else:
            hi_off = (addr_hi + row * wr_stride) % GTX_L2_SIZE_BYTES
        lo_off = (addr_lo + row * length) % GTX_L1_SIZE_BYTES

        copy_len = min(length,
                        GTX_L2_SIZE_BYTES - hi_off,
                        GTX_L1_SIZE_BYTES - lo_off)
        if copy_len <= 0:
            continue
        if not is_store:
            l1[lo_off : lo_off + copy_len] = l2[hi_off : hi_off + copy_len]
        else:
            l2[hi_off : hi_off + copy_len] = l1[lo_off : lo_off + copy_len]
    return 0


# ============================================================================
# firmware_dma_tloop_copy -- direct port of gtx_npu_dma.cc:334-348
# ============================================================================
def firmware_dma_tloop_copy(mem: 'GtxMemory', *, nest: int, spu: int,
                              src_addr: int, dst_addr: int,
                              length: int, height: int) -> int:
    """T-loop L1 -> L1 same-SPU copy (matches C++ std::memmove semantics).

    `.copy()` on src slice is essential: source/dest may overlap, and numpy
    slice assignment without copy can corrupt overlapping ranges.
    """
    l1 = mem.l1_byte(nest, spu)
    for row in range(height):
        s_off = (src_addr + row * length) % GTX_L1_SIZE_BYTES
        d_off = (dst_addr + row * length) % GTX_L1_SIZE_BYTES
        copy_len = min(length,
                        GTX_L1_SIZE_BYTES - s_off,
                        GTX_L1_SIZE_BYTES - d_off)
        if copy_len <= 0:
            continue
        l1[d_off : d_off + copy_len] = l1[s_off : s_off + copy_len].copy()
    return 0
