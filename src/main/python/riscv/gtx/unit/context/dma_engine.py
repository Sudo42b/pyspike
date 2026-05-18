"""DMA engine -- spike-independent DMA helpers + DeferredDdrStore + decoder.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc:25-435.

Per CONTEXT D-01: This module has NO `proc`/`insn` dependencies. It operates
on `GtxMemory` instances and pure data only -- the spike-dependent entry
points (firmware_dma @handler) live in `ops/dma.py` (Plan 02). Plans 02/04/05
all import from this module.

Phase 3 plan 01 Task 2.

Phase 9 Wave 5 (plan 09-02b): ported from torch to xp (numpy default,
cupy under GTX_USE_CUDA=1). All copy_/view-as-reshape/permute/cpu/clone/
fill_/to-device torch-API sites are replaced with the xp-uniform
equivalents per 09-RESEARCH Pitfall 1.
The xp.ndarray-returning shim sites at memory.py (``l0_byte``, ``ddr.read``)
are bypassed here by reading raw xp storage directly (``mem.l[012][nest, spu]``,
``mem.ddr._bytes[...]``) — same pattern Wave 2a (op-handlers) adopted; lets
this file stay torch-free even before its shim sites in memory.py are removed.

Invariant policy (user decision): every per-region access verifies its
bounds with ``assert`` and then performs the whole transfer in a single
``xp.copyto`` or ``.transpose`` + ``ascontiguousarray`` op. Wrap-around and
out-of-bounds are treated as firmware bugs and surfaced via
``AssertionError`` — no silent clipping, no fallback paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from ...config_params import (
    xp, to_host,
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
    With those, the whole 2D transfer is one ``xp.copyto`` — one CUDA launch
    (or one SIMD copy on numpy), no per-row Python loop. ``height == 1``
    collapses to a 1-row view, still a single copy.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert width > 0 and height > 0, f"width {width} or height {height} is 0"
    assert l2_stride == 0, f"l2_stride {l2_stride} != 0 not supported yet"
    l2_stride = width

    # Shim bypass — read raw xp storage directly (same pattern Wave 2a adopted).
    l1_buf = mem.l1[nest_id, spu_id]
    l2_buf = mem.l2[nest_id]

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

    # RESEARCH Pitfall 1: torch view-as-reshape ported to xp .reshape.
    # The L2 region is row-major with stride l2_stride and length width
    # per row.
    l2_view = l2_buf[l2_addr:l2_end].reshape(height, l2_stride)[:, :width]
    l1_view = l1_buf[l1_addr:l1_end].reshape(height, width)
    if is_load:
        xp.copyto(l1_view, l2_view)
    else:
        xp.copyto(l2_view, l1_view)
    return 0


# ============================================================================
# exec_load_svr -- direct port of gtx_npu_dma.cc:97-113
# ============================================================================
def exec_load_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                   l1_addr: int, l0_reg: int) -> None:
    """L1 -> L0 transfer (32 bytes = one SVR register, 8 x 4-byte words).

    L0 slot ``(l0_reg & 0x1F) * 32`` is always 32-byte aligned and ends at
    +32 ≤ ``GTX_L0_SIZE_BYTES`` (1 KB), so L0 never wraps. L1 must not
    wrap either — firmware bug if it does. Single 32-byte slice
    assignment replaces the 8 × 4-byte word loop.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    # Shim bypass — raw xp storage.
    l1_buf = mem.l1[nest_id, spu_id]
    l0_buf = mem.l0[nest_id, spu_id]
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
    slice assignment.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    # Shim bypass — raw xp storage.
    l1_buf = mem.l1[nest_id, spu_id]
    l0_buf = mem.l0[nest_id, spu_id]
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
    without wrap. ``xp.ascontiguousarray`` clones the transposed view, so
    ``src == dst`` (in-place transpose) is safe. ``rows == 1`` or
    ``cols == 1`` is a degenerate transpose that still costs one
    contiguous copy — no special case needed.
    """
    assert nest_id < GTX_NEST_NUM, f"nest_id {nest_id} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert spu_id < GTX_SPU_NUM, f"spu_id {spu_id} >= GTX_SPU_NUM {GTX_SPU_NUM}"
    assert rows > 0 and cols > 0, f"rows {rows} or cols {cols} is 0"

    # Shim bypass — raw xp storage, reinterpret as FP16 view (zero-copy
    # under numpy/cupy LE byte order).
    l1_f16 = mem.l1[nest_id, spu_id].view(xp.float16)
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

    # RESEARCH Pitfall 1: torch view-as-reshape ported to xp .reshape.
    src_view = l1_f16[a_h:a_h + nelem].reshape(rows, cols)
    # `.t()` (torch) → `.T` (numpy/cupy). `.contiguous()` → `xp.ascontiguousarray`.
    # The final `.view(-1)` flattens — numpy uses `.reshape(-1)`.
    l1_f16[r_h:r_h + nelem] = xp.ascontiguousarray(src_view.T).reshape(-1)
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
    ``(old_dims[p2], old_dims[p1], old_dims[p0])`` → ``xp.transpose(
    2 - p2, 2 - p1, 2 - p0)``. ``xp.ascontiguousarray`` flattens row-major,
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

    # Shim bypass — read raw DDR bytes; reinterpret + permute on xp.
    src_span = mem.ddr._bytes[src_off:src_off + nelem * 2]
    src_3d = src_span.view(xp.float16).reshape(dim2, dim1, dim0)
    # `.permute(axes)` → `.transpose(axes)` (same kwargs); `.contiguous()`
    # → `xp.ascontiguousarray`.
    permuted = xp.ascontiguousarray(src_3d.transpose(2 - p2, 2 - p1, 2 - p0))
    # `.view(uint8).reshape(-1)` rewritten with xp.uint8 (numpy/cupy dtype).
    mem.ddr.write(dst_off, permuted.view(xp.uint8).reshape(-1))


# ============================================================================
# exec_fill -- direct port of gtx_npu_dma.cc:230-246
# ============================================================================
def exec_fill(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
               length: int, fill_val: int, addr_r: int) -> int:
    """Fill L1 region at ``addr_r`` with constant FP16 value (``length``
    elements × 2 bytes each).

    Operates through L1's uint16 view — each ``length`` element write
    becomes a single slice assignment, preserving the raw 16-bit pattern
    (no FP16 cast → no NaN/denormal re-encoding). Wrap-around splits
    the write into two contiguous fills.
    """
    assert nest_id < GTX_NEST_NUM and spu_id < GTX_SPU_NUM, (
        f"invalid nest_id {nest_id} or spu_id {spu_id}"
    )

    # Shim bypass — raw xp storage reinterpret as uint16 view (zero-copy LE).
    l1_u16 = mem.l1[nest_id, spu_id].view(xp.uint16)
    nelem = l1_u16.shape[0]
    r_off = (addr_r // 2) % nelem
    fill = fill_val & 0xFFFF
    # `.fill_(val)` (torch in-place) → slice assign (numpy idiomatic) or
    # `.fill(val)`. Use slice-assign to avoid relying on torch-style fill_.
    if r_off + length <= nelem:
        l1_u16[r_off:r_off + length] = fill
    else:
        head = nelem - r_off
        l1_u16[r_off:] = fill
        l1_u16[:length - head] = fill
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
    are ≥ length. A single snapshot of the contiguous DDR span, then one
    ``xp.copyto`` over (height, length) 2D views. No row loop, no
    per-row launch.
    """
    ddr_off_base = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi
    max_off = ddr_off_base + (height - 1) * rd_stride + length
    ensure_ddr(mem, max_off)
    ddr_cap = mem.ddr.capacity()

    # Shim bypass — raw xp storage for both L2 and DDR.
    l2_buf = mem.l2[nest]
    # Under xp, DDR + scratchpads share a backend (D-10 unified), so the
    # cross-device `.to(l2_buf.device)` torch step is now a no-op — drop it.
    ddr_span = mem.ddr._bytes[
        ddr_off_base : min(max_off, ddr_cap)
    ]

    l2_end = addr_lo + (height - 1) * wr_stride + length
    assert rd_stride >= length, f"rd_stride {rd_stride} < length {length}"
    assert wr_stride >= length, f"wr_stride {wr_stride} < length {length}"
    # `.numel()` (torch) → `.size` (xp attribute).
    assert height * rd_stride <= ddr_span.size, (
        f"DDR span {ddr_span.size} too small for height*rd_stride "
        f"{height * rd_stride} — firmware bug"
    )
    assert l2_end <= GTX_L2_SIZE_BYTES, (
        f"L2 region wraps GTX_L2_SIZE_BYTES {GTX_L2_SIZE_BYTES} — "
        f"firmware bug"
    )
    assert ddr_off_base + (height - 1) * rd_stride + length <= ddr_cap, (
        f"DDR window exceeds capacity {ddr_cap} — firmware bug"
    )

    # RESEARCH Pitfall 1: torch view-as-reshape ported to xp .reshape;
    # the dtype-only view is preserved with .view(xp.<dtype>). Slicing
    # semantics identical.
    src_2d = ddr_span[:height * rd_stride].reshape(height, rd_stride)[:, :length]
    dst_2d = l2_buf[addr_lo:addr_lo + height * wr_stride].reshape(height, wr_stride)[:, :length]
    xp.copyto(dst_2d, src_2d)
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
    ``xp.copyto`` over (height, length) 2D views — no row loop.
    """
    # Shim bypass — raw xp storage.
    l1 = mem.l1[nest, spu]
    l2 = mem.l2[nest]
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

    l2_view = l2[addr_hi:addr_hi + height * hi_stride].reshape(height, hi_stride)[:, :length]
    l1_view = l1[addr_lo:lo_end].reshape(height, length)
    if is_store:
        xp.copyto(l2_view, l1_view)
    else:
        xp.copyto(l1_view, l2_view)
    return 0


# ============================================================================
# firmware_dma_tloop_copy -- direct port of gtx_npu_dma.cc:334-348
# ============================================================================
def firmware_dma_tloop_copy(mem: 'GtxMemory', *, nest: int, spu: int,
                              src_addr: int, dst_addr: int,
                              length: int, height: int) -> int:
    """T-loop L1 → L1 same-SPU copy (memmove semantics).

    Invariants (asserted): both windows stay inside L1 without wrap. One
    ``.copy()`` on the source 2D view handles src/dst overlap, then a
    single ``xp.copyto`` writes back.
    """
    # Shim bypass — raw xp storage.
    l1 = mem.l1[nest, spu]

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

    # torch view-as-reshape + clone ported to xp .reshape + .copy().
    src_2d = l1[src_addr:src_end].reshape(height, length).copy()
    xp.copyto(l1[dst_addr:dst_end].reshape(height, length), src_2d)
    return 0


# ============================================================================
# firmware_mcast_s2l -- direct port of
# vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273
# ============================================================================
def firmware_mcast_s2l(mem: 'GtxMemory', *, nest: int,
                        l2_addr: int, l1_addr: int,
                        height: int, length: int,
                        rd_stride: int, target_spu_mask: int) -> int:
    """L2 → L1 multicast to selected SPUs (funct7=0x42).

    Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273.

    HW conventions (vendor :248-249): height==0 → 1, length==0 → 0x10000.
    Source L2 row span is snapshotted ONCE (single 2D view), then `xp.copyto`
    into each selected SPU L1 (vendor row-loop collapses to one launch).
    """
    if height == 0:
        height = 1
    if length == 0:
        length = 0x10000
    assert nest < GTX_NEST_NUM, f"nest {nest} >= GTX_NEST_NUM {GTX_NEST_NUM}"
    assert height > 0 and length > 0, f"height={height} length={length}"
    # Vendor row-loop uses `(l2_addr + row * rd_stride) % GTX_L2_SIZE` for wrap
    # safety. Functional model asserts no-wrap (firmware bug if it would).
    # When rd_stride == 0, vendor re-reads the same row each iter — match
    # literally (the 2D view below collapses to a single-row repeat via
    # broadcast-style copy when rd_stride == length).
    effective_stride = rd_stride if rd_stride > 0 else length
    assert effective_stride >= length, (
        f"rd_stride {rd_stride} < length {length} (firmware bug)"
    )

    # Shim bypass — raw xp storage.
    l2 = mem.l2[nest]
    src_end = l2_addr + (height - 1) * effective_stride + length
    assert src_end <= GTX_L2_SIZE_BYTES, (
        f"L2 src region [{l2_addr}, {src_end}) wraps GTX_L2_SIZE_BYTES "
        f"{GTX_L2_SIZE_BYTES} — firmware bug"
    )
    src_2d = l2[l2_addr:l2_addr + height * effective_stride] \
        .reshape(height, effective_stride)[:, :length]

    l1_end = l1_addr + height * length
    assert l1_end <= GTX_L1_SIZE_BYTES, (
        f"L1 dst region [{l1_addr}, {l1_end}) wraps GTX_L1_SIZE_BYTES "
        f"{GTX_L1_SIZE_BYTES} — firmware bug"
    )
    for s in range(GTX_SPU_NUM):
        if not ((target_spu_mask >> s) & 1):
            continue
        # Shim bypass — raw xp storage.
        l1 = mem.l1[nest, s]
        xp.copyto(l1[l1_addr:l1_end].reshape(height, length), src_2d)
    return 0


# ============================================================================
# firmware_mcast_g2s -- direct port of
# vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:545-583
# ============================================================================
def firmware_mcast_g2s(mem: 'GtxMemory', *, ddr_addr: int, l2_addr: int,
                        height: int, length: int,
                        rd_stride: int, target_nest_mask: int) -> int:
    """DDR → L2 multicast to selected NESTs (funct7=0x44, funct3=0).

    Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:545-583.

    HW conventions (vendor :561): height==0 → 1, length==0 → 0x10000.
    Source DDR row span snapshotted ONCE, then `xp.copyto` into each selected
    NEST L2.

    NO zero-fill special case: vendor has none (RESEARCH Pitfall 1 — earlier
    Python docstring fiction).
    """
    if height == 0:
        height = 1
    if length == 0:
        length = 0x10000
    assert height > 0 and length > 0, f"height={height} length={length}"
    effective_stride = rd_stride if rd_stride > 0 else length
    assert effective_stride >= length, (
        f"rd_stride {rd_stride} < length {length} (firmware bug)"
    )

    ddr_off_base = (ddr_addr - GTX_DDR_BASE) if ddr_addr >= GTX_DDR_BASE else ddr_addr
    max_off = ddr_off_base + (height - 1) * effective_stride + length
    ensure_ddr(mem, max_off)
    ddr_cap = mem.ddr.capacity()
    assert max_off <= ddr_cap, (
        f"DDR window [{ddr_off_base}, {max_off}) exceeds capacity "
        f"{ddr_cap} — firmware bug"
    )

    # Snapshot the full row span once. Under xp (D-10) DDR + L2 share a
    # backend so the explicit cross-device staging is dropped.
    ddr_span = mem.ddr._bytes[ddr_off_base:max_off]
    src_2d = ddr_span[:height * effective_stride] \
        .reshape(height, effective_stride)[:, :length]

    l2_end = l2_addr + height * length
    assert l2_end <= GTX_L2_SIZE_BYTES, (
        f"L2 dst region [{l2_addr}, {l2_end}) wraps GTX_L2_SIZE_BYTES "
        f"{GTX_L2_SIZE_BYTES} — firmware bug"
    )
    for k in range(GTX_NEST_NUM):
        if not ((target_nest_mask >> k) & 1):
            continue
        # Shim bypass — raw xp storage.
        l2 = mem.l2[k]
        xp.copyto(l2[l2_addr:l2_end].reshape(height, length), src_2d)
    return 0


# ============================================================================
# firmware_mcast_s2s -- direct port of
# vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:732-762
# ============================================================================
def firmware_mcast_s2s(mem: 'GtxMemory', *, src_tmu: int,
                        src_addr: int, dst_addr: int,
                        src_stride: int, dst_stride: int,
                        length: int, height: int,
                        target_nest_mask: int) -> int:
    """L2 → L2 multicast across NESTs (funct7=0x44, funct3=2 / sub_op=0x22).

    Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:732-762.

    HW conventions (vendor :741-742): height==0 → 1, NO length normalisation.
    Per-row temp-buffer read-then-write (distinct src/dst strides — unified
    2D view is not safe).

    NO self-broadcast guard (RESEARCH Pitfall 3): vendor iterates all 4 NESTs
    and writes wherever ``target_nest_mask`` says, even if ``src_tmu == k``.
    ``src_tmu >= GTX_NEST_NUM`` clamps to 0 per vendor :740.
    """
    if height == 0:
        height = 1
    assert height > 0 and length > 0, f"height={height} length={length}"
    if src_tmu >= GTX_NEST_NUM:
        src_tmu = 0

    # Shim bypass — raw xp storage.
    src_l2 = mem.l2[src_tmu]
    for row in range(height):
        s_off = (src_addr + row * src_stride) % GTX_L2_SIZE_BYTES
        d_off = (dst_addr + row * dst_stride) % GTX_L2_SIZE_BYTES
        # Vendor :749-750 — copy_len = min(length, GTX_L2_SIZE - max(s_off, d_off))
        copy_len = min(length, GTX_L2_SIZE_BYTES - max(s_off, d_off))
        if copy_len <= 0:
            continue
        # Temp buffer for the row (`.clone()` torch → `.copy()` xp) keeps
        # overlap safety if src==dst NEST.
        tmp = src_l2[s_off:s_off + copy_len].copy()
        for k in range(GTX_NEST_NUM):
            if not ((target_nest_mask >> k) & 1):
                continue
            # Shim bypass — raw xp storage. Under xp same backend so the
            # `.to(dst_l2.device)` cross-device step is dropped.
            dst_l2 = mem.l2[k]
            xp.copyto(dst_l2[d_off:d_off + copy_len], tmp)
    return 0


# ============================================================================
# firmware_copy_mem -- direct port of
# vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:763-846
# ============================================================================
def firmware_copy_mem(npu: Any, *, nest_id: int,
                       src_addr_raw: int, dst_addr_raw: int,
                       src_stride: int, dst_stride: int,
                       length: int, height: int) -> int:
    """DDR↔DDR (and L2↔DDR, L2↔L2) memory copy (funct7=0x44, f3=3, sub_op=0x23).

    Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:763-846.

    HW conventions (vendor :777): height==0 → 1, NO length normalisation.

    DDR-vs-L2 decision (vendor :779-780): addr >= GTX_L2_SIZE_BYTES → DDR.
    DDR-touching path FIRST LINE calls npu.flush_deferred_ddr_stores()
    (vendor :784 — mandatory). L2↔L2 same-NEST branch does NOT flush
    (asymmetry preserved per RESEARCH Pitfall 2).
    """
    if height == 0:
        height = 1
    assert height > 0 and length > 0, f"height={height} length={length}"

    mem = npu.mem
    src_is_ddr = src_addr_raw >= GTX_L2_SIZE_BYTES
    dst_is_ddr = dst_addr_raw >= GTX_L2_SIZE_BYTES

    if src_is_ddr or dst_is_ddr:
        # ── DDR-touching path: mandatory flush first (vendor dispatch.cc:784) ──
        npu.flush_deferred_ddr_stores()

        # Convert raw addresses → DDR buffer offsets (vendor ddr_offset() helper).
        src_off = (src_addr_raw - GTX_DDR_BASE) if src_addr_raw >= GTX_DDR_BASE else src_addr_raw
        dst_off = (dst_addr_raw - GTX_DDR_BASE) if dst_addr_raw >= GTX_DDR_BASE else dst_addr_raw

        if src_is_ddr and dst_is_ddr:
            # DDR-to-DDR (vendor :800-810)
            max_src = src_off + (height - 1) * src_stride + length
            max_dst = dst_off + (height - 1) * dst_stride + length
            ensure_ddr(mem, max(max_src, max_dst))
            for row in range(height):
                s_base = src_off + row * src_stride
                d_base = dst_off + row * dst_stride
                # Vendor :806-807 — per-row capacity clip
                copy_len = length
                ddr_cap = mem.ddr.capacity()
                if s_base + copy_len > ddr_cap:
                    copy_len = max(0, ddr_cap - s_base)
                if d_base + copy_len > ddr_cap:
                    copy_len = max(0, ddr_cap - d_base)
                if copy_len > 0:
                    # Shim bypass — raw xp DDR storage. The intermediate
                    # buffer is a copy() to handle DDR-to-DDR overlap safely
                    # (if src/dst windows alias the same backing bytes).
                    src_bytes = mem.ddr._bytes[s_base:s_base + copy_len].copy()
                    mem.ddr.write(d_base, src_bytes)
        elif src_is_ddr and not dst_is_ddr:
            # DDR-to-L2 (vendor :812-822)
            n = nest_id if nest_id < GTX_NEST_NUM else 0
            # Shim bypass — raw xp L2 storage.
            l2 = mem.l2[n]
            max_src = src_off + (height - 1) * src_stride + length
            ensure_ddr(mem, max_src)
            for row in range(height):
                s = src_off + row * src_stride
                d = (dst_addr_raw + row * dst_stride) % GTX_L2_SIZE_BYTES
                copy_len = length
                ddr_cap = mem.ddr.capacity()
                if s + copy_len > ddr_cap:
                    copy_len = max(0, ddr_cap - s)
                if d + copy_len > GTX_L2_SIZE_BYTES:
                    copy_len = max(0, GTX_L2_SIZE_BYTES - d)
                if copy_len > 0:
                    # Shim bypass — raw xp DDR storage. Under xp (D-10)
                    # DDR + L2 share a backend so the cross-device
                    # `.to(l2.device)` step is dropped.
                    src_bytes = mem.ddr._bytes[s:s + copy_len]
                    xp.copyto(l2[d:d + copy_len], src_bytes)
        else:
            # L2-to-DDR (vendor :824-834)
            n = nest_id if nest_id < GTX_NEST_NUM else 0
            # Shim bypass — raw xp L2 storage.
            l2 = mem.l2[n]
            max_dst = dst_off + (height - 1) * dst_stride + length
            ensure_ddr(mem, max_dst)
            for row in range(height):
                s = (src_addr_raw + row * src_stride) % GTX_L2_SIZE_BYTES
                d = dst_off + row * dst_stride
                copy_len = length
                ddr_cap = mem.ddr.capacity()
                if s + copy_len > GTX_L2_SIZE_BYTES:
                    copy_len = max(0, GTX_L2_SIZE_BYTES - s)
                if d + copy_len > ddr_cap:
                    copy_len = max(0, ddr_cap - d)
                if copy_len > 0:
                    # `.cpu()` (torch) → `to_host(...)` xp helper (no-op on
                    # numpy; cp.asnumpy on cupy). DDR file-I/O boundary
                    # convention from Wave 1a.
                    mem.ddr.write(d, to_host(l2[s:s + copy_len]))
    else:
        # ── L2-to-L2 same-NEST (vendor :836-844) ──
        # NO flush (asymmetry preserved per RESEARCH Pitfall 2).
        n = nest_id if nest_id < GTX_NEST_NUM else 0
        # Shim bypass — raw xp L2 storage.
        l2 = mem.l2[n]
        for row in range(height):
            s_off = (src_addr_raw + row * src_stride) % GTX_L2_SIZE_BYTES
            d_off = (dst_addr_raw + row * dst_stride) % GTX_L2_SIZE_BYTES
            copy_len = min(length, GTX_L2_SIZE_BYTES - max(s_off, d_off))
            if copy_len <= 0:
                continue
            # Temp buffer (`.clone()` torch → `.copy()` xp) for overlap
            # safety per vendor :841.
            tmp = l2[s_off:s_off + copy_len].copy()
            xp.copyto(l2[d_off:d_off + copy_len], tmp)
    return 0
