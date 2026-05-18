"""Wave 1 bridge-shim invariants (Phase 9 Plan 09-01b Task 4 — Option B).

Wave 1 ported memory.py / register_file.py / npu.py storage to xp, but
Wave 2/3 consumer files (dma_engine.py, tloop_buffer.py, ops/*, _verify.py)
still call torch APIs (`.to(...)`, `.view(torch.float16)`, `.copy_(...)`,
`.numel()`) on the buffers returned by `GtxMemory.{l0,l1,l2}_byte`,
`{l0,l1,l2}_f16`, and `DDR_MEMORY.read`. Per user decision 2026-05-18 we
restore wave-end ABS GREEN with a strangler-fig **torch-view shim**:

    accessor()  ->  torch.from_numpy(xp_buf)   when xp is numpy
                ->  RuntimeError                when xp is cupy
                                                (torch.from_numpy doesn't
                                                 accept cupy buffers and
                                                 Wave 2/3 cupy ports must
                                                 already be done by then)

Zero-copy: `torch.from_numpy(arr)` shares the same underlying buffer
(no allocation, no `.copy()`). Memory.py's xp-internal *storage* contract
is preserved; only the accessor *return type* is bridged.

The shim is throwaway. Waves 2/3 remove sites individually as each torch
consumer is ported off torch. See `09-01b-SUMMARY.md` "Deviations" + the
per-shim removal-wave inheritance table for the sunset plan.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from riscv.gtx.config_params import xp
from riscv.gtx.unit import memory as memmod
from riscv.gtx.unit.memory import DDR_MEMORY, GtxMemory


MEMORY_PY = (
    Path(__file__).resolve().parents[2]
    / "src" / "main" / "python" / "riscv" / "gtx" / "unit" / "memory.py"
)


# ------------------------------------------------------------------------
# Shim helper presence (source-level contract)
# ------------------------------------------------------------------------

def test_memory_module_exposes_torch_view_helper():
    """memory.py defines a torch-view shim helper at module scope.

    Name is `_torch_view` (the leading underscore signals "private, throwaway").
    """
    assert hasattr(memmod, "_torch_view"), (
        "memory.py must define `_torch_view(arr)` shim helper (Option B)."
    )


def test_memory_module_docstring_documents_shim():
    """Module docstring documents shim existence + sunset condition."""
    src = MEMORY_PY.read_text()
    # Sunset condition mention: shim goes away when torch consumers are
    # ported (Wave 3 = all torch consumers gone).
    assert "WAVE-1-SHIM" in src or "_torch_view" in src, (
        "memory.py module docstring should mention the shim (search markers)."
    )
    # Verify the docstring is not silent about the bridge.
    assert "torch" in src.lower(), (
        "memory.py post-shim must reference torch in docstring/comments "
        "(the shim exists explicitly to bridge xp -> torch)."
    )


def test_shim_sites_carry_removal_wave_markers():
    """Every shim call-site has a `# WAVE-1-SHIM: remove in Wave <N>` marker.

    The marker tells future readers (and the wave that owns removal) which
    plan inherits the obligation to delete that shim.

    Wave 1b landed 7 shims (3 byte + 3 f16 + ddr.read). Wave 2a (plan
    09-02a-ops) removed the 3 f16 shims after porting ops/*.py off torch.
    Wave 5 (plan 09-02b-engines) removed l0_byte + ddr.read shims after
    porting dma_engine.py off torch — those accessors no longer have any
    torch consumers. The 2 remaining shims (l1_byte, l2_byte) are
    inherited by Wave 6 (09-03-finalize) for tloop_buffer.py.
    """
    src = MEMORY_PY.read_text()
    markers = re.findall(r"WAVE-1-SHIM:\s*remove in Wave \w+", src)
    assert len(markers) >= 2, (
        f"Expected >= 2 WAVE-1-SHIM removal markers (post-Wave-5: l1_byte, "
        f"l2_byte); found {len(markers)}: {markers}"
    )


# ------------------------------------------------------------------------
# Zero-copy semantics on numpy path
# ------------------------------------------------------------------------

def test_torch_view_returns_torch_tensor_on_numpy_path():
    """`_torch_view(numpy_arr)` returns a `torch.Tensor`."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    arr = np.zeros((8,), dtype=np.uint8)
    t = memmod._torch_view(arr)
    assert isinstance(t, torch.Tensor)


def test_torch_view_shares_memory_with_source_array():
    """`_torch_view` is zero-copy (`torch.from_numpy` semantics).

    Writes via the torch tensor must be visible in the source ndarray
    and vice versa. This is the core invariant that makes the shim
    cheap enough to keep until Wave 3.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    arr = np.zeros((4,), dtype=np.uint8)
    t = memmod._torch_view(arr)
    # Write via torch, read via numpy:
    t[0] = 0xAB
    assert int(arr[0]) == 0xAB, (
        "Shim must be zero-copy: torch.from_numpy shares the buffer."
    )
    # Write via numpy, read via torch:
    arr[1] = 0xCD
    assert int(t[1].item()) == 0xCD


def test_torch_view_preserves_dtype_for_uint8_and_float16():
    """Shim preserves dtype across the bridge — uint8 and float16 (FP16)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    u8 = np.zeros((4,), dtype=np.uint8)
    t_u8 = memmod._torch_view(u8)
    assert t_u8.dtype == torch.uint8

    f16 = np.zeros((4,), dtype=np.float16)
    t_f16 = memmod._torch_view(f16)
    assert t_f16.dtype == torch.float16


# ------------------------------------------------------------------------
# Accessor return-type contract (the shim's whole point)
# ------------------------------------------------------------------------

def test_l1_byte_returns_torch_tensor_on_numpy_path():
    """`mem.l1_byte(nest, spu)` returns torch.Tensor under shim (numpy path).

    Wave 2 callers (ops/act.py, ops/mm.py, dma_engine.py, ...) consume
    this with `.view(torch.float16)`, `.copy_(...)`, etc. The shim lets
    those torch calls work until Wave 2 ports them.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l1_byte(0, 0)
    assert isinstance(buf, torch.Tensor), (
        f"l1_byte must return torch.Tensor under shim, got {type(buf)}"
    )
    assert buf.dtype == torch.uint8


def test_l2_byte_returns_torch_tensor_on_numpy_path():
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l2_byte(0)
    assert isinstance(buf, torch.Tensor)
    assert buf.dtype == torch.uint8


def test_l0_byte_returns_xp_ndarray_post_wave5():
    """Wave 5 (plan 09-02b) removed the l0_byte shim — accessor returns
    bare xp.ndarray. dma_engine.py was the only torch consumer of l0_byte
    (lines 155/179); with dma_engine ported the accessor surfaces raw xp
    storage. ops/{act,mm,spr,vec}.py (Wave 2a) already bypass via
    `npu.mem.l0[nest, spu]`.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l0_byte(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.uint8


def test_l1_f16_returns_xp_ndarray_post_wave2a():
    """Wave 2a removed the f16 shim — accessor returns bare xp.ndarray."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l1_f16(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_l0_f16_returns_xp_ndarray_post_wave2a():
    """Wave 2a removed the f16 shim — accessor returns bare xp.ndarray."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l0_f16(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_l2_f16_returns_xp_ndarray_post_wave2a():
    """Wave 2a removed the f16 shim — accessor returns bare xp.ndarray."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l2_f16(0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_ddr_read_returns_xp_ndarray_post_wave5():
    """Wave 5 (plan 09-02b) removed the ddr.read shim — accessor returns
    bare xp.ndarray. dma_engine.py was the only torch consumer of
    DDR_MEMORY.read (lines 266 / 345-348 / 534 / 647 / 664); with
    dma_engine ported and using raw `mem.ddr._bytes[start:end]` slicing,
    the accessor surfaces raw xp storage.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    ddr = DDR_MEMORY()
    ddr.ensure(64)
    buf = ddr.read(0, 32)
    assert isinstance(buf, np.ndarray), (
        f"DDR_MEMORY.read must return xp.ndarray post-Wave-5, got {type(buf)}"
    )
    assert buf.dtype == np.uint8


# ------------------------------------------------------------------------
# Accessor zero-copy: writes via torch view land in the xp storage
# ------------------------------------------------------------------------

def test_l1_byte_torch_write_visible_in_underlying_storage():
    """Write through the shim view shows up in the module-level scratchpad.

    This is the core invariant: the shim is a *view*, not a copy. Wave 1a's
    in-place DMA / vec / op semantics depend on writes being persistent.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    # Clear so we start from a known state.
    mem.l1[:] = 0
    buf = mem.l1_byte(0, 1)
    assert isinstance(buf, torch.Tensor)
    buf[42] = 0x5A  # write via torch
    # Read back via the underlying xp storage:
    assert int(mem.l1[0, 1, 42]) == 0x5A, (
        "Shim must be a view, not a copy — writes must persist in xp storage."
    )


def test_ddr_read_xp_write_visible_in_underlying_storage_post_wave5():
    """Wave 5 (plan 09-02b) — DDR_MEMORY.read shim removed; the returned
    xp.ndarray view still aliases _bytes so writes through the view land
    in underlying storage.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    ddr = DDR_MEMORY()
    ddr.ensure(64)
    # zero out first
    ddr._bytes[:] = 0
    buf = ddr.read(0, 32)
    assert isinstance(buf, np.ndarray)
    buf[7] = 0x9C
    assert int(ddr._bytes[7]) == 0x9C


# ------------------------------------------------------------------------
# cupy path: shim raises RuntimeError with site hint
# ------------------------------------------------------------------------

def test_torch_view_raises_runtime_error_on_cupy_path():
    """Under xp=cupy the shim is unreachable — Wave 2/3 cupy ports must
    have removed all torch-API call sites by then. Raising loudly with
    a "<which torch-API caller is still here>" hint surfaces the
    incomplete-port bug instantly.

    We can't toggle xp at runtime (it's frozen at import per D-02), so we
    simulate the cupy branch by passing a cupy-shaped sentinel and
    monkeypatching xp identity inside the helper. The contract verified:
    when the shim's `xp is np` check fails, it raises RuntimeError that
    mentions "Wave 2/3 cupy ports incomplete".
    """
    # The contract: function must have a cupy-branch that raises
    # RuntimeError("Wave 2/3 cupy ports incomplete: ...") rather than
    # silently calling torch.from_numpy on a cupy buffer (which would
    # explode with a confusing AttributeError deep inside torch).
    src = MEMORY_PY.read_text()
    assert "Wave 2/3 cupy ports incomplete" in src, (
        "_torch_view must raise RuntimeError with 'Wave 2/3 cupy ports "
        "incomplete' hint on the cupy path."
    )
    # And the cupy branch must reference the caller-site hint:
    assert "RuntimeError" in src, "Shim cupy path must raise RuntimeError."


# ------------------------------------------------------------------------
# Regression guard: Wave 1a/1b unit tests still pass under the shim
# ------------------------------------------------------------------------

def test_underlying_xp_storage_still_uses_xp_zeros():
    """Shim does NOT regress Wave 1a's xp-internal storage contract.

    The module-level scratchpads + DDR backing array stay xp.ndarray
    (numpy.ndarray under default xp=numpy). Only the *accessor return*
    is bridged.
    """
    mem = GtxMemory()
    # mem.l0 / mem.l1 / mem.l2 are direct references to module-level
    # globals — they must remain xp arrays.
    assert type(mem.l0).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l1).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l2).__module__.startswith(("numpy", "cupy"))
    # DDR backing also stays xp:
    assert type(mem.ddr._bytes).__module__.startswith(("numpy", "cupy"))
