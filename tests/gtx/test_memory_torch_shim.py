"""memory.py is torch-free; accessors return bare xp.ndarray views.

Audit-style tests that lock in:
  * No `import torch` in memory.py source.
  * No `_torch_view` helper attribute on the module.
  * Every l{0,1,2}_byte / l{0,1,2}_f16 / ddr.read accessor returns a bare
    xp.ndarray with the expected dtype.
  * Accessor returns are views (zero-copy): writes via the view land in
    the module-level scratchpad / DDR backing storage.
"""
from __future__ import annotations

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
# memory.py source-level audit: no torch
# ------------------------------------------------------------------------

def test_memory_module_has_no_torch_view_helper():
    """memory.py must NOT expose a `_torch_view` attribute."""
    assert not hasattr(memmod, "_torch_view")


def test_memory_module_is_torch_free():
    """memory.py source has zero `import torch` statements."""
    src = MEMORY_PY.read_text()
    code_lines = [line for line in src.splitlines()
                  if "import torch" in line and not line.lstrip().startswith(("#", '"', "'"))]
    assert not code_lines, (
        f"memory.py still has live `import torch` statements: {code_lines}"
    )


# ------------------------------------------------------------------------
# Accessor return-type contract — all bare xp.ndarray
# ------------------------------------------------------------------------

def test_l1_byte_returns_xp_ndarray():
    """l1_byte returns bare xp.ndarray (uint8)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l1_byte(0, 0)
    assert isinstance(buf, np.ndarray), f"got {type(buf)}"
    assert buf.dtype == np.uint8


def test_l2_byte_returns_xp_ndarray():
    """l2_byte returns bare xp.ndarray (uint8)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l2_byte(0)
    assert isinstance(buf, np.ndarray), f"got {type(buf)}"
    assert buf.dtype == np.uint8


def test_l0_byte_returns_xp_ndarray():
    """l0_byte returns bare xp.ndarray (uint8)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l0_byte(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.uint8


def test_l1_f16_returns_xp_ndarray():
    """l1_f16 returns bare xp.ndarray (float16)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l1_f16(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_l0_f16_returns_xp_ndarray():
    """l0_f16 returns bare xp.ndarray (float16)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l0_f16(0, 0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_l2_f16_returns_xp_ndarray():
    """l2_f16 returns bare xp.ndarray (float16)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    mem = GtxMemory()
    buf = mem.l2_f16(0)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float16


def test_ddr_read_returns_xp_ndarray():
    """DDR_MEMORY.read returns bare xp.ndarray (uint8)."""
    if xp is not np:
        pytest.skip("Numpy-path test (default xp=numpy)")
    ddr = DDR_MEMORY()
    ddr.ensure(64)
    buf = ddr.read(0, 32)
    assert isinstance(buf, np.ndarray), f"got {type(buf)}"
    assert buf.dtype == np.uint8


# ------------------------------------------------------------------------
# Accessor zero-copy: writes via the xp view land in the underlying storage
# ------------------------------------------------------------------------

def test_l1_byte_xp_write_visible_in_underlying_storage():
    """l1_byte returns a view (not a copy); writes persist in the
    module-level scratchpad — required by the in-place DMA / vec / op
    semantics.
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


def test_ddr_read_xp_write_visible_in_underlying_storage():
    """DDR_MEMORY.read returns a view that aliases _bytes; writes through
    the view land in underlying storage.
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
# Regression guard: xp-internal storage contract preserved
# ------------------------------------------------------------------------

def test_underlying_xp_storage_still_uses_xp_zeros():
    """Module-level scratchpads + DDR backing array stay xp.ndarray
    (numpy.ndarray under default xp=numpy)."""
    mem = GtxMemory()
    # mem.l0 / mem.l1 / mem.l2 are direct references to module-level
    # globals — they must remain xp arrays.
    assert type(mem.l0).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l1).__module__.startswith(("numpy", "cupy"))
    assert type(mem.l2).__module__.startswith(("numpy", "cupy"))
    # DDR backing also stays xp:
    assert type(mem.ddr._bytes).__module__.startswith(("numpy", "cupy"))
