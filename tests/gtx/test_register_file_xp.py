"""RegisterFile xp-port invariants (Phase 9 Plan 09-01b Task 1).

Pins the post-Wave-1b contract: RegisterFile.tensor is an `xp.ndarray`
(numpy.ndarray by default; cupy.ndarray under GTX_USE_CUDA=1) of dtype
`xp.int64`. Constructor no longer accepts `device=` kwarg (xp is
device-implicit per D-11 / D-12).

These tests are written RED first — they will fail against the legacy
torch-backed register_file.py and pass after the xp port.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riscv.gtx.config_params import xp
from riscv.gtx.unit.csr import GSPR, LSPR
from riscv.gtx.unit.register_file import RegisterFile


def test_register_file_int64_xp_storage():
    """RegisterFile allocates `xp.zeros(shape, dtype=xp.int64)` storage."""
    rf = RegisterFile(GSPR, shape=(1024,))
    # Storage class follows xp (numpy default, cupy under GTX_USE_CUDA=1).
    assert rf.tensor.dtype == xp.int64
    assert tuple(rf.tensor.shape) == (1024,)
    # Module name proves we're not on torch any more.
    assert type(rf.tensor).__module__.startswith(("numpy", "cupy"))


def test_register_file_zero_initialized():
    """Fresh RegisterFile is all zeros."""
    rf = RegisterFile(GSPR, shape=(1024,))
    # `xp.all(... == 0)` works on numpy + cupy uniformly.
    assert bool(xp.all(rf.tensor == 0))


def test_register_file_write_read_roundtrip():
    """rf.write(addr, value) round-trips back to rf.read(addr)."""
    rf = RegisterFile(GSPR, shape=(1024,))
    # __setitem__ on int key takes raw address path.
    rf[0x010] = 0xCAFE
    # __getitem__ on 1D returns int via the raw-address path.
    assert int(rf[0x010]) == 0xCAFE


def test_register_file_int64_max_value():
    """Writing 0x7FFFFFFFFFFFFFFF (int64 max) preserves the value — no overflow."""
    rf = RegisterFile(GSPR, shape=(1024,))
    rf[0x010] = 0x7FFFFFFFFFFFFFFF
    assert int(rf[0x010]) == 0x7FFFFFFFFFFFFFFF


def test_register_file_constructor_has_no_device_kwarg():
    """RegisterFile no longer accepts `device=` (xp is device-implicit)."""
    import inspect
    sig = inspect.signature(RegisterFile.__init__)
    assert "device" not in sig.parameters, (
        f"`device` kwarg should be gone post-Wave-1b; found: {list(sig.parameters)}"
    )


def test_register_file_multidim_shape():
    """Multi-dim shapes (LSPR per-NEST × per-SPU) preserved on xp."""
    rf = RegisterFile(LSPR, shape=(4, 16, 1024))
    assert tuple(rf.tensor.shape) == (4, 16, 1024)
    assert rf.tensor.dtype == xp.int64


def test_register_file_64bit_field_broadcast_no_overflow():
    """64-bit field writes (top bit set) must not OverflowError.

    Pins the pre-existing 64-bit signed-wrap fix in __setattr__ — that
    logic must survive the torch → xp port. Uses LSPR.SGPR0.gpr which is
    a 64-bit field.
    """
    rf = RegisterFile(LSPR, shape=(4, 16, 1024))
    val_u64 = 0xCAFEBABEDEADBEEF
    rf.SGPR0.gpr = val_u64  # must not raise
    addr = 0x800 & 0x3FF
    stored = rf.tensor[..., addr]
    signed = val_u64 - (1 << 64)
    assert tuple(stored.shape) == (4, 16)
    # `xp.all` returns an xp scalar; `bool(...)` works on both numpy and cupy.
    assert bool(xp.all(stored == signed))


def test_no_torch_in_register_file_source():
    """register_file.py is torch-free (H-5 audit)."""
    path = Path(__file__).resolve().parents[2] / "src/main/python/riscv/gtx/unit/register_file.py"
    src = path.read_text()
    assert "import torch" not in src, f"register_file.py still imports torch:\n{src[:500]}"
    # tolerate the word 'torch' in source comments? Plan wants 0 — be strict.
    # Exception: SystemC-style historic comments. Strict per acceptance criteria.
    occurrences = [line for line in src.splitlines() if "torch" in line.lower() and not line.lstrip().startswith("#")]
    assert not occurrences, f"register_file.py has non-comment torch refs:\n{occurrences}"


def test_no_device_kwarg_in_register_file_source():
    """register_file.py source has zero `device=` kwarg refs."""
    path = Path(__file__).resolve().parents[2] / "src/main/python/riscv/gtx/unit/register_file.py"
    src = path.read_text()
    # match standalone kwarg pattern `device=`, allow `device:` typing if accidentally appears.
    bad_lines = [
        line for line in src.splitlines()
        if "device=" in line and not line.lstrip().startswith("#")
    ]
    assert not bad_lines, f"register_file.py still has device= refs:\n{bad_lines}"


def test_register_file_xp_import_present():
    """register_file.py imports xp from config_params."""
    path = Path(__file__).resolve().parents[2] / "src/main/python/riscv/gtx/unit/register_file.py"
    src = path.read_text()
    assert "from ..config_params import xp" in src or "from riscv.gtx.config_params import xp" in src, (
        "register_file.py must import `xp` from config_params"
    )
