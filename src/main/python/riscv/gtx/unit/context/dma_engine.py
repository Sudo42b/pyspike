"""DMA engine -- spike-independent DMA helpers + DeferredDdrStore + decoder.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc:25-435.

Per CONTEXT D-01: This module has NO `proc`/`insn` dependencies. It operates
on `GtxMemory` instances and pure data only -- the spike-dependent entry
points (firmware_dma @handler) live in `ops/dma.py` (Plan 02). Plans 02/04/05
all import from this module.

Phase 3 plan 01 Task 2.

Invariant policy (user decision): every per-region access verifies its
bounds with ``assert`` and then performs the whole transfer in a single
``copy_()`` or ``permute`` + ``contiguous`` op. Wrap-around and
out-of-bounds are treated as firmware bugs and surfaced via
``AssertionError`` — no silent clipping, no fallback paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import torch
from ...config_params import (
    GTX_NEST_NUM, GTX_SPU_NUM,
    GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES,
    GTX_DDR_BASE,
)

if TYPE_CHECKING:
    from ..memory import GtxMemory


def ensure_ddr(mem: 'GtxMemory', end_offset: int):
    """Compat shim for the old top-level ``ddr.ensure_ddr``.

    Delegates to :meth:`GtxMemory.ensure_ddr`, the canonical doubling-grow
    DDR allocator.
    """
    return mem.ensure_ddr(end_offset)


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
    """Strided 2D DMA between NEST L2 and SPU L1.

    Invariants (asserted): non-zero size, contiguous L2 region
    (``l2_stride == 0`` sentinel, normalised to ``width``), no L2/L1 wrap.
    With those, the whole 2D transfer is one ``copy_()`` — one CUDA launch,
    no per-row Python loop. ``height == 1`` collapses to a 1-row view,
    still a single copy.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert width > 0 and height > 0, f"width {width} or height {height} is 0"
    assert l2_stride == 0, f"l2_stride {l2_stride} != 0 not supported yet"
    l2_stride = width

    l1_buf = mem.l1_byte(nest_id, spu_id)
    l2_buf = mem.l2_byte(nest_id)

    l2_end = l2_addr + height * l2_stride
    l1_end = l1_addr + height * width
    assert l2_end <= GTX_L2_SIZE_BYTES, (
        f"L2 region [{l2_addr}, {l2_end}) wraps GTX_L2_SIZE_BYTES "
        f"{GTX_L2_SIZE_BYTES} — firmware bug"
    )
    assert l1_end <= GTX_L1_SIZE_BYTES, (
        f"L1 region [{l1_addr}, {l1_end}) wraps GTX_L1_SIZE_BYTES "
        f"{GTX_L1_SIZE_BYTES} — firmware bug"
    )

    l2_view = l2_buf[l2_addr:l2_end].view(height, l2_stride)[:, :width]
    l1_view = l1_buf[l1_addr:l1_end].view(height, width)
    if is_load:
        l1_view.copy_(l2_view)
    else:
        l2_view.copy_(l1_view)
    return 0


# ============================================================================
# exec_load_svr -- direct port of gtx_npu_dma.cc:97-113
# ============================================================================
def exec_load_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                   l1_addr: int, l0_reg: int) -> None:
    """L1 -> L0 transfer (32 bytes = one SVR register, 8 x 4-byte words).

    L0 slot ``(l0_reg & 0x1F) * 32`` is always 32-byte aligned and ends at
    +32 ≤ ``GTX_L0_SIZE_BYTES`` (1 KB), so L0 never wraps. L1 must not
    wrap either — firmware bug if it does. Single 32-byte CUDA slice
    assignment replaces the 8 × 4-byte word loop.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % GTX_L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32
    assert l1_off + 32 <= GTX_L1_SIZE_BYTES, (
        f"L1 SVR window [{l1_off}, {l1_off + 32}) wraps "
        f"GTX_L1_SIZE_BYTES {GTX_L1_SIZE_BYTES} — firmware bug"
    )
    l0_buf[l0_off:l0_off + 32] = l1_buf[l1_off:l1_off + 32]


# ============================================================================
# exec_store_svr -- direct port of gtx_npu_dma.cc:118-136
# ============================================================================
def exec_store_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                    l1_addr: int, l0_reg: int) -> None:
    """L0 -> L1 transfer (32 bytes = one SVR register, 8 x 4-byte words).

    Mirror of :func:`exec_load_svr` — L0 source slot never wraps; L1
    destination must not wrap (firmware bug if it does). Single 32-byte
    CUDA slice assignment.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    l1_buf = mem.l1_byte(nest_id, spu_id)
    l0_buf = mem.l0_byte(nest_id, spu_id)
    l1_off = l1_addr % GTX_L1_SIZE_BYTES
    l0_off = (l0_reg & 0x1F) * 32
    assert l1_off + 32 <= GTX_L1_SIZE_BYTES, (
        f"L1 SVR window [{l1_off}, {l1_off + 32}) wraps "
        f"GTX_L1_SIZE_BYTES {GTX_L1_SIZE_BYTES} — firmware bug"
    )
    l1_buf[l1_off:l1_off + 32] = l0_buf[l0_off:l0_off + 32]


# ============================================================================
# exec_transpose -- direct port of gtx_npu_dma.cc:143-167
# ============================================================================
def exec_transpose(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                    rows: int, cols: int, addr_a: int, addr_r: int) -> int:
    """In-place L1 matrix transpose (FP16, 2 bytes per elem).

    Invariants (asserted): src and dst FP16 windows both fit within L1
    without wrap. ``.contiguous()`` clones the transposed view, so
    ``src == dst`` (in-place transpose) is safe. ``rows == 1`` or
    ``cols == 1`` is a degenerate transpose that still costs one
    contiguous copy — no special case needed.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    assert rows > 0 and cols > 0, f"rows {rows} or cols {cols} is 0"

    l1_f16 = mem.l1_byte(nest_id, spu_id).view(torch.float16)
    nelem_total = l1_f16.shape[0]
    nelem = rows * cols
    a_h = (addr_a // 2) % nelem_total
    r_h = (addr_r // 2) % nelem_total

    assert a_h + nelem <= nelem_total, (
        f"src window [{a_h}, {a_h + nelem}) wraps L1 fp16 capacity "
        f"{nelem_total} — firmware bug"
    )
    assert r_h + nelem <= nelem_total, (
        f"dst window [{r_h}, {r_h + nelem}) wraps L1 fp16 capacity "
        f"{nelem_total} — firmware bug"
    )

    src_view = l1_f16[a_h:a_h + nelem].view(rows, cols)
    l1_f16[r_h:r_h + nelem] = src_view.t().contiguous().view(-1)
    return 0


# ============================================================================
# exec_transpose_ddr -- direct port of gtx_npu_dma.cc:175-225
# ============================================================================
def exec_transpose_ddr(mem: 'GtxMemory', *, src_addr: int, dst_addr: int,
                        dim2: int, dim1: int, dim0: int,
                        p2: int, p1: int, p0: int) -> None:
    """DDR-to-DDR 3D tensor transpose/permute (FP16).

    Reads ``[dim2][dim1][dim0]`` from ``src_addr``, writes the permuted
    layout to ``dst_addr``. ``(p2, p1, p0)`` selects which old axis
    drives each new axis.

    Axis mapping: src axis k holds ``dim_(2-k)`` (axis 0 = dim2, axis
    1 = dim1, axis 2 = dim0). The output shape is
    ``(old_dims[p2], old_dims[p1], old_dims[p0])`` → ``torch.permute(
    2 - p2, 2 - p1, 2 - p0)``. ``.contiguous()`` flattens row-major,
    matching the vendor ``dst_idx = oi[p2]*new_s2 + oi[p1]*new_s1 +
    oi[p0]``.
    """
    src_off = (src_addr - GTX_DDR_BASE) if src_addr >= GTX_DDR_BASE else src_addr
    dst_off = (dst_addr - GTX_DDR_BASE) if dst_addr >= GTX_DDR_BASE else dst_addr

    assert dim2 > 0 and dim1 > 0 and dim0 > 0, (
        f"dims must be positive: dim2={dim2} dim1={dim1} dim0={dim0}"
    )

    nelem = dim2 * dim1 * dim0
    max_off = max(src_off + nelem * 2, dst_off + nelem * 2)
    ensure_ddr(mem, max_off)
    cap = mem.ddr.capacity()

    assert src_off + nelem * 2 <= cap, (
        f"src region [{src_off}, {src_off + nelem * 2}) exceeds DDR "
        f"capacity {cap} — firmware bug"
    )
    assert dst_off + nelem * 2 <= cap, (
        f"dst region [{dst_off}, {dst_off + nelem * 2}) exceeds DDR "
        f"capacity {cap} — firmware bug"
    )

    src_span = mem.ddr.read(src_off, nelem * 2)
    src_3d = src_span.view(torch.float16).reshape(dim2, dim1, dim0)
    permuted = src_3d.permute(2 - p2, 2 - p1, 2 - p0).contiguous()
    mem.ddr.write(dst_off, permuted.view(torch.uint8).reshape(-1))


# ============================================================================
# exec_fill -- direct port of gtx_npu_dma.cc:230-246
# ============================================================================
def exec_fill(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
               length: int, fill_val: int, addr_r: int) -> int:
    """Fill L1 region at ``addr_r`` with constant FP16 value (``length``
    elements × 2 bytes each).

    Operates through L1's uint16 view — each ``length`` element write
    becomes a single ``fill_`` call, preserving the raw 16-bit pattern
    (no FP16 cast → no NaN/denormal re-encoding). Wrap-around splits
    the write into two contiguous fills.
    """
    assert nest_id < GTX_NEST_NUM and spu_id < GTX_SPU_NUM, (
        f"invalid nest_id {nest_id} or spu_id {spu_id}"
    )

    l1_u16 = mem.l1_byte(nest_id, spu_id).view(torch.uint16)
    nelem = l1_u16.shape[0]
    r_off = (addr_r // 2) % nelem
    fill = fill_val & 0xFFFF
    if r_off + length <= nelem:
        l1_u16[r_off:r_off + length].fill_(fill)
    else:
        head = nelem - r_off
        l1_u16[r_off:].fill_(fill)
        l1_u16[:length - head].fill_(fill)
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
    """S-loop LOAD: DDR → L2.

    Invariants (asserted): row windows fit in DDR span and in L2, strides
    are ≥ length. A single H→D snapshot of the contiguous DDR span,
    then one ``copy_()`` over (height, length) 2D views. No row loop,
    no per-row CUDA launch.
    """
    ddr_off_base = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi
    max_off = ddr_off_base + (height - 1) * rd_stride + length
    ensure_ddr(mem, max_off)
    ddr_cap = mem.ddr.capacity()

    l2_buf = mem.l2_byte(nest)
    ddr_span = mem.ddr.read(
        ddr_off_base,
        min(max_off, ddr_cap) - ddr_off_base,
    ).to(l2_buf.device)

    l2_end = addr_lo + (height - 1) * wr_stride + length
    assert rd_stride >= length, f"rd_stride {rd_stride} < length {length}"
    assert wr_stride >= length, f"wr_stride {wr_stride} < length {length}"
    assert height * rd_stride <= ddr_span.numel(), (
        f"DDR span {ddr_span.numel()} too small for height*rd_stride "
        f"{height * rd_stride} — firmware bug"
    )
    assert l2_end <= GTX_L2_SIZE_BYTES, (
        f"L2 region wraps GTX_L2_SIZE_BYTES {GTX_L2_SIZE_BYTES} — "
        f"firmware bug"
    )
    assert ddr_off_base + (height - 1) * rd_stride + length <= ddr_cap, (
        f"DDR window exceeds capacity {ddr_cap} — firmware bug"
    )

    src_2d = ddr_span[:height * rd_stride].view(height, rd_stride)[:, :length]
    dst_2d = l2_buf[addr_lo:addr_lo + height * wr_stride].view(height, wr_stride)[:, :length]
    dst_2d.copy_(src_2d)
    return 0


# ============================================================================
# firmware_dma_tloop_load_store -- direct port of gtx_npu_dma.cc:349-391
# ============================================================================
def firmware_dma_tloop_load_store(mem: 'GtxMemory', *, nest: int, spu: int,
                                    is_store: bool,
                                    addr_hi: int, addr_lo: int,
                                    length: int, height: int,
                                    rd_stride: int, wr_stride: int) -> int:
    """T-loop L2 ↔ L1 strided per-row.

    LOAD:  ``L2[addr_hi + row*rd_stride] -> L1[addr_lo + row*length]``
    STORE: ``L1[addr_lo + row*length] -> L2[addr_hi + row*wr_stride]``

    Invariants (asserted): stride ≥ length, no L2/L1 wrap. Single
    ``copy_()`` over (height, length) 2D views — no row loop.
    """
    l1 = mem.l1_byte(nest, spu)
    l2 = mem.l2_byte(nest)
    hi_stride = wr_stride if is_store else rd_stride

    hi_end = addr_hi + (height - 1) * hi_stride + length
    lo_end = addr_lo + height * length
    assert hi_stride >= length, f"hi_stride {hi_stride} < length {length}"
    assert hi_end <= GTX_L2_SIZE_BYTES, (
        f"L2 region wraps GTX_L2_SIZE_BYTES {GTX_L2_SIZE_BYTES} — "
        f"firmware bug"
    )
    assert lo_end <= GTX_L1_SIZE_BYTES, (
        f"L1 region [{addr_lo}, {lo_end}) wraps GTX_L1_SIZE_BYTES "
        f"{GTX_L1_SIZE_BYTES} — firmware bug"
    )

    l2_view = l2[addr_hi:addr_hi + height * hi_stride].view(height, hi_stride)[:, :length]
    l1_view = l1[addr_lo:lo_end].view(height, length)
    if is_store:
        l2_view.copy_(l1_view)
    else:
        l1_view.copy_(l2_view)
    return 0


# ============================================================================
# firmware_dma_tloop_copy -- direct port of gtx_npu_dma.cc:334-348
# ============================================================================
def firmware_dma_tloop_copy(mem: 'GtxMemory', *, nest: int, spu: int,
                              src_addr: int, dst_addr: int,
                              length: int, height: int) -> int:
    """T-loop L1 → L1 same-SPU copy (memmove semantics).

    Invariants (asserted): both windows stay inside L1 without wrap. One
    ``.clone()`` on the source 2D view handles src/dst overlap, then a
    single ``copy_()`` writes back.
    """
    l1 = mem.l1_byte(nest, spu)

    src_end = src_addr + height * length
    dst_end = dst_addr + height * length
    assert src_end <= GTX_L1_SIZE_BYTES, (
        f"src window [{src_addr}, {src_end}) wraps GTX_L1_SIZE_BYTES "
        f"{GTX_L1_SIZE_BYTES} — firmware bug"
    )
    assert dst_end <= GTX_L1_SIZE_BYTES, (
        f"dst window [{dst_addr}, {dst_end}) wraps GTX_L1_SIZE_BYTES "
        f"{GTX_L1_SIZE_BYTES} — firmware bug"
    )

    src_2d = l1[src_addr:src_end].view(height, length).clone()
    l1[dst_addr:dst_end].view(height, length).copy_(src_2d)
    return 0
