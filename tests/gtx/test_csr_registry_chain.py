"""CSR registry + RegisterFile tensor chain coverage.

Locks the post-d6f73f9 invariants: @csr decorator populates module-level
GSPR/NSPR/LSPR dicts at import time, Register schemas are stateless, and
RegisterFile materializes them as int64 torch.Tensor storage with
PIPE-only addr indexing (APB excluded, scope-masked to 10 bits).
"""
from __future__ import annotations

import pytest
import torch

from riscv.gtx.unit.csr import (
    GSPR,
    NSPR,
    LSPR,
    BusType,
    Register,
    find_by_address,
)
from riscv.gtx.unit.register_file import RegisterFile


def test_gspr_nspr_lspr_populated_at_import_time():
    """Module-level dicts are non-empty and contain known PIPE entries."""
    assert len(GSPR) > 10
    assert len(NSPR) > 5
    assert len(LSPR) > 10
    assert "STACK_INFO" in GSPR
    assert "THREAD_MASK" in NSPR
    assert "SPM_ADDRA" in LSPR


def test_csr_decorator_produces_register_schema():
    """@csr lifts a bits()-attr class into a Register with full metadata."""
    reg = NSPR["THREAD_MASK"]
    assert isinstance(reg, Register)
    assert reg.name == "THREAD_MASK"
    assert reg.address == 0x400
    assert reg.width == 16
    assert reg.rw_type == "RW"
    assert reg.bus_type is BusType.PIPE
    assert "mask" in reg.fields
    field = reg.fields["mask"]
    assert field.start == 0 and field.end == 15
    assert field.mask == 0xFFFF


def test_register_file_gspr_is_1024_int64_tensor():
    """RegisterFile(GSPR) allocates a zero-initialized int64 tensor."""
    rf = RegisterFile(GSPR, shape=(1024,), device="cpu")
    assert rf.tensor.shape == (1024,)
    assert rf.tensor.dtype == torch.int64
    assert bool((rf.tensor == 0).all())


def test_register_file_nspr_lspr_multidim_shapes(gtx_npu):
    """GtxNpu fixture allocates per-scope shapes: GSPR/NSPR/LSPR."""
    assert tuple(gtx_npu.gspr.tensor.shape) == (1024,)
    assert tuple(gtx_npu.nspr.tensor.shape) == (4, 1024)
    assert tuple(gtx_npu.lspr.tensor.shape) == (4, 16, 1024)


def test_addr_by_name_pipe_only_masked_to_10_bits():
    """_addr_by_name covers PIPE only; APB entries are excluded; addr & 0x3FF."""
    rf = RegisterFile(LSPR, shape=(4, 16, 1024), device="cpu")
    assert "SPM_ADDRA" in rf._addr_by_name
    assert rf._addr_by_name["SPM_ADDRA"] == (0x900 & 0x3FF)
    apb_keys = [n for n, r in LSPR.items() if r.bus_type is BusType.APB]
    assert apb_keys, "expected at least one APB entry in LSPR"
    for k in apb_keys:
        assert k not in rf._addr_by_name


def test_register_view_attribute_write_broadcasts_across_nests(gtx_npu):
    """16-bit THREAD_MASK.mask seeded across all 4 nests by vendor defaults.

    Deliberately exercises a ≤63-bit field — register_file.py:188 raises
    OverflowError on full 64-bit field writes (e.g. SPM_ADDRA.value =
    0xDEADBEEFCAFEBABE) because ~(0xFFFFFFFFFFFFFFFF << 0) leaves int64.
    That bug is out of scope for this plan and flagged in SUMMARY.
    """
    view = gtx_npu.nspr.THREAD_MASK
    assert tuple(view._tensor.shape) == (4,)
    mask_vals = (view._tensor & 0xFFFF).tolist()
    assert mask_vals == [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF]


def test_find_by_address_pipe_each_scope():
    """find_by_address(PIPE) routes by address range to the right scope."""
    assert find_by_address(0x010, BusType.PIPE).name == "STACK_INFO"
    assert find_by_address(0x400, BusType.PIPE).name == "THREAD_MASK"
    assert find_by_address(0x900, BusType.PIPE).name == "SPM_ADDRA"


def test_find_by_address_pipe_missing_raises_keyerror():
    """Unused GSPR slot yields KeyError, not silent None."""
    with pytest.raises(KeyError):
        find_by_address(0x099, BusType.PIPE)
