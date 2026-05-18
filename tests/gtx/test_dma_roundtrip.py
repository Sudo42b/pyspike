"""Wave 1a DMA-pattern unit tests against memory.py — backend-agnostic.

Plan 09-01a: validates that the `DDR_MEMORY.read/write/view` API + the
scratchpad byte/f16 views support the canonical DMA traffic patterns
(DDR ↔ L1 byte transfer + FP16 reinterpret) entirely on the xp backend
(no torch).

These do NOT exercise the dma_engine.py orchestration layer (that's Wave 2);
they only confirm that the memory layer's contract is correct so Wave 2 can
build on top.
"""
from __future__ import annotations

import pytest

from riscv.gtx import config_params
from riscv.gtx.config_params import xp
from riscv.gtx.unit.memory import DDR_MEMORY, GtxMemory


def _is_xp_ndarray(arr) -> bool:
    return isinstance(arr, xp.ndarray)


def test_ddr_write_then_read_byte_exact():
    """Plain byte read/write through DDR_MEMORY.write / DDR_MEMORY.read."""
    ddr = DDR_MEMORY(size=4096)
    payload = xp.arange(256, dtype=xp.uint8)
    ddr.write(0x100, payload)

    out = ddr.read(0x100, 256)
    assert _is_xp_ndarray(out)
    assert out.dtype == xp.uint8
    # Compare via host bridge to keep test agnostic to backend.
    from riscv.gtx.config_params import to_host
    assert (to_host(out) == to_host(payload)).all(), (
        "DDR byte roundtrip diverged"
    )


def test_ddr_view_fp16_reinterpret():
    """DDR_MEMORY.view(xp.float16) reinterprets the uint8 backing as fp16
    (LE host assumption). Bytes [0x00, 0x3C, 0x00, 0xC0] → [1.0, -2.0]."""
    ddr = DDR_MEMORY(size=4096)
    ddr.write(
        0,
        xp.asarray([0x00, 0x3C, 0x00, 0xC0, 0x00, 0x40], dtype=xp.uint8),
    )
    view = ddr.view(xp.float16)
    assert _is_xp_ndarray(view)
    assert view.dtype == xp.float16
    assert float(view[0]) == 1.0
    assert float(view[1]) == -2.0
    assert float(view[2]) == 2.0


def test_ddr_to_l1_byte_transfer_pattern():
    """The canonical DMA pattern: DDR → L1 byte transfer via slice copy.

    This is the operation dma_engine.exec_load() will perform in Wave 2;
    here we confirm the memory layer's primitives compose correctly."""
    mem = GtxMemory()
    mem.reset_scratchpads()
    mem.ensure_ddr(4096)

    # Stamp 64 bytes in DDR.
    payload = xp.arange(0x10, 0x10 + 64, dtype=xp.uint8)
    mem.ddr.write(0x200, payload)

    # DMA "load" — copy 64 bytes from DDR(0x200) to L1(nest=0, spu=0, off=128).
    src = mem.ddr.read(0x200, 64)
    nest, spu = 0, 0
    off = 128
    l1 = mem.l1_byte(nest, spu)
    l1[off:off + 64] = src

    # Verify L1 contents byte-exact.
    from riscv.gtx.config_params import to_host
    got = to_host(l1[off:off + 64])
    want = to_host(payload)
    assert (got == want).all(), (
        "DDR → L1 byte transfer lost bytes"
    )


def test_l1_fp16_view_write_visible_through_byte_view():
    """Cross-view aliasing in both directions (DMA store path)."""
    mem = GtxMemory()
    mem.reset_scratchpads()

    nest, spu = 0, 0
    l1_byte = mem.l1_byte(nest, spu)
    l1_f16 = mem.l1_f16(nest, spu)

    # Write 4 FP16 values through the f16 view.
    l1_f16[:4] = xp.asarray([1.0, -2.0, 2.0, 0.5], dtype=xp.float16)

    # Read those 8 bytes through the byte view (LE order).
    # 1.0 = 0x3C00 → [0x00, 0x3C]
    # -2.0 = 0xC000 → [0x00, 0xC0]
    # 2.0 = 0x4000 → [0x00, 0x40]
    # 0.5 = 0x3800 → [0x00, 0x38]
    from riscv.gtx.config_params import to_host
    bytes_view = to_host(l1_byte[:8])
    expected = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x40, 0x00, 0x38]
    assert list(int(x) for x in bytes_view) == expected, (
        f"FP16 → byte alias view mismatch: got {[int(x) for x in bytes_view]}, "
        f"expected {expected}"
    )


def test_ddr_ensure_idempotent_when_already_large():
    """`ensure()` when `end_offset` is below `getsize()` is a no-op (no
    reallocation, same backing reference)."""
    ddr = DDR_MEMORY(size=2 * 1024 * 1024)  # 2 MiB
    before = ddr.raw()
    again = ddr.ensure(1024)  # 1 KiB — way below current capacity
    assert again is before, (
        "ensure() must not realloc when end_offset is below current size"
    )


def test_ddr_clear_zeros_backing_in_place():
    """`clear()` zeros the existing backing without reallocating."""
    ddr = DDR_MEMORY(size=256)
    backing_before = ddr.raw()
    ddr.write(0, xp.arange(64, dtype=xp.uint8))
    assert int(ddr._bytes[10]) == 10  # sanity
    ddr.clear()
    backing_after = ddr.raw()
    assert backing_after is backing_before, "clear() must not realloc"
    # All bytes zero now.
    from riscv.gtx.config_params import to_host
    assert int(to_host(backing_after).sum()) == 0
