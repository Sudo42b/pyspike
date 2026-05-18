"""Wave 1 bridge-shim invariants (Phase 9 Plan 09-01b Task 4 — Option B).

Wave 1 ported memory.py / register_file.py / npu.py storage to xp via a
strangler-fig torch-view shim that wrapped accessor returns in
`torch.from_numpy(...)` so un-ported Wave 2/3 callers kept working. The
shim was throwaway; Waves 2/3 removed sites individually as each torch
consumer was ported off torch.

Phase 9 Wave 6 (plan 09-03-finalize) closed the strangler-fig: all 7
WAVE-1-SHIM sites + the `_torch_view` helper + the local `import torch`
are now removed. memory.py is torch-free; every accessor returns bare
xp.ndarray.

The tests in this file used to assert the shim's behavior; they were
flipped to assert the post-Wave-6 xp-native return contract. The shim
helper presence tests are removed (the helper no longer exists). The
zero-copy contract is now an xp.ndarray view-aliasing invariant.

See `09-01b-SUMMARY.md` "Deviations" + the per-shim removal-wave
inheritance table for the historical sunset plan.
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

from riscv.gtx.config_params import xp
from riscv.gtx.unit import memory as memmod
from riscv.gtx.unit.memory import DDR_MEMORY, GtxMemory


MEMORY_PY = (
    Path(__file__).resolve().parents[2]
    / "src" / "main" / "python" / "riscv" / "gtx" / "unit" / "memory.py"
)


# ------------------------------------------------------------------------
# Post-Wave-6 contract: memory.py is fully torch-free
# ------------------------------------------------------------------------

def test_memory_module_has_no_torch_view_helper_post_wave6():
    """Wave 6 removed `_torch_view` helper + module-level torch import."""
    assert not hasattr(memmod, "_torch_view"), (
        "memory.py must NOT define `_torch_view` after Wave 6 sunset."
    )


def test_memory_module_is_torch_free_post_wave6():
    """memory.py source has zero `import torch` statements after Wave 6."""
    src = MEMORY_PY.read_text()
    code_lines = [line for line in src.splitlines()
                  if "import torch" in line and not line.lstrip().startswith(("#", '"', "'"))]
    assert not code_lines, (
        f"memory.py still has live `import torch` statements: {code_lines}"
    )


def test_memory_module_docstring_documents_wave6_sunset():
    """Module docstring records the Wave 6 sunset of the WAVE-1-SHIM."""
    src = MEMORY_PY.read_text()
    # Sunset condition: the historical strangler-fig is documented.
    assert "WAVE-1-SHIM" in src and "Wave 6" in src, (
        "memory.py docstring should document the Wave 6 shim sunset history."
    )


# ------------------------------------------------------------------------
# Accessor return-type contract — all bare xp.ndarray post-Wave-6
# ------------------------------------------------------------------------

def test_l1_byte_returns_xp_ndarray_post_wave6():
    """Wave 6 (plan 09-03-finalize) removed the l1_byte shim — accessor
    returns bare xp.ndarray. tloop_buffer.py was the last torch consumer
    (line 483) and is now ported off torch.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l1_byte(0, 0)
    assert isinstance(buf, np.ndarray), (
        f"l1_byte must return xp.ndarray post-Wave-6, got {type(buf)}"
    )
    assert buf.dtype == np.uint8


def test_l2_byte_returns_xp_ndarray_post_wave6():
    """Wave 6 removed the l2_byte shim — accessor returns bare xp.ndarray.
    tloop_buffer.py was the only torch consumer of l2_byte (lines
    459/467/477/485); with tloop_buffer ported the accessor surfaces raw
    xp storage.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l2_byte(0)
    assert isinstance(buf, np.ndarray), (
        f"l2_byte must return xp.ndarray post-Wave-6, got {type(buf)}"
    )
    assert buf.dtype == np.uint8


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
# Accessor zero-copy: writes via the xp view land in the underlying storage
# ------------------------------------------------------------------------

def test_l1_byte_xp_write_visible_in_underlying_storage_post_wave6():
    """Wave 6 — l1_byte returns bare xp.ndarray view; writes through the
    view persist in the module-level scratchpad. Wave 1a's in-place DMA /
    vec / op semantics depend on the view-aliasing contract surviving the
    shim removal.
    """
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    # Clear so we start from a known state.
    mem.l1[:] = 0
    buf = mem.l1_byte(0, 1)
    assert isinstance(buf, np.ndarray)
    buf[42] = 0x5A  # write via the xp view
    # Read back via the underlying xp storage:
    assert int(mem.l1[0, 1, 42]) == 0x5A, (
        "Accessor must return a view, not a copy — writes must persist in xp storage."
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
# Regression guard: Wave 1a/1b xp-internal storage contract preserved
# ------------------------------------------------------------------------

def test_underlying_xp_storage_still_uses_xp_zeros():
    """Post-Wave-6 xp-internal storage contract is preserved.

    The module-level scratchpads + DDR backing array stay xp.ndarray
    (numpy.ndarray under default xp=numpy).
    """
    mem = GtxMemory()
    # mem.l0 / mem.l1 / mem.l2 are direct references to module-level
    # globals — they must remain xp arrays.
    assert type(mem.l0).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l1).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l2).__module__.startswith(("numpy", "cupy"))
    # DDR backing also stays xp:
    assert type(mem.ddr._bytes).__module__.startswith(("numpy", "cupy"))
