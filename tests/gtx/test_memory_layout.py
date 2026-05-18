"""Wave 1a unit tests: unit/memory.py is torch-free, xp-backed.

Plan 09-01a (Wave 1 part a) — port `unit/memory.py` from torch to xp.

These tests assert the **memory layer contract** post-port:
  * Module-level scratchpads (`_L2_GLOBAL`, `_L1_GLOBAL`, `_L0_GLOBAL`) +
    `DDR_MEMORY._bytes` allocate via `xp.zeros(..., dtype=xp.uint8)`.
  * `GtxMemory.l1_byte` returns an xp.ndarray[uint8]; `l1_f16` returns
    `xp.ndarray[float16]` aliased on the same storage (write through
    uint8 view shows up in fp16 view and vice-versa — D-10 LE byte-order
    via `.view(xp.float16)` reinterpret, no copy).
  * `_DDR_DEVICE` literal removed (H-5 audit).
  * No `torch` references anywhere in `unit/memory.py`.
  * `D-10` VRAM-budget comment present near DDR_MEMORY init.

Wave 1a invariant for downstream waves:
  `mem.l{0,1,2}_byte(...)` and `mem.l{0,1,2}_f16(...)` return xp.ndarrays.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from riscv.gtx import config_params
from riscv.gtx.config_params import xp
from riscv.gtx.unit import memory as memmod
from riscv.gtx.unit.memory import DDR_MEMORY, GtxMemory


MEMORY_PY = (
    Path(__file__).resolve().parents[2]
    / "src" / "main" / "python" / "riscv" / "gtx" / "unit" / "memory.py"
)


# --------------------------------------------------------------------------
# Source-level invariants (grep-level acceptance criteria from plan 09-01a)
# --------------------------------------------------------------------------

def test_memory_py_has_no_torch_references():
    """Plan acceptance: `grep -c "import torch\\|torch\\."` returns 0."""
    src = MEMORY_PY.read_text()
    # Strip out comment-only mentions before greppage so historical doc
    # comments don't false-positive. We want NO active torch usage.
    matches = re.findall(r"\btorch\.", src)
    assert matches == [], (
        f"memory.py must be torch-free post-Wave-1a, found {len(matches)} "
        f"`torch.` references: {matches[:5]}"
    )
    assert "import torch" not in src, (
        "memory.py must not `import torch` post-Wave-1a"
    )


def test_memory_py_imports_xp_from_config_params():
    """Plan acceptance: at least 1 `from ..config_params import xp`."""
    src = MEMORY_PY.read_text()
    assert re.search(r"from\s+\.\.config_params\s+import\s+.*\bxp\b", src), (
        "memory.py must import xp from ..config_params"
    )


def test_ddr_device_literal_removed():
    """H-5 completeness: `_DDR_DEVICE` and `torch.device` removed."""
    src = MEMORY_PY.read_text()
    assert "_DDR_DEVICE" not in src, (
        "memory.py must not reference `_DDR_DEVICE` (H-5 clean cut)"
    )
    assert "torch.device" not in src, (
        "memory.py must not reference `torch.device` (H-5 clean cut)"
    )


def test_no_ddr_device_anywhere_in_gtx_package():
    """H-5 audit: `_DDR_DEVICE` removed from entire gtx/ tree."""
    gtx_root = MEMORY_PY.parent.parent  # .../riscv/gtx/
    matches = subprocess.run(
        ["grep", "-rn", "_DDR_DEVICE", "--include=*.py", str(gtx_root)],
        capture_output=True, text=True,
    )
    # grep returns 1 when no matches (success for us).
    assert matches.returncode != 0 or matches.stdout.strip() == "", (
        f"Found residual `_DDR_DEVICE` references after Wave 1a:\n"
        f"{matches.stdout}"
    )


def test_xp_zeros_used_at_least_four_times():
    """Plan acceptance: ≥ 4 `xp.zeros` calls (3 scratchpads + DDR init + grow)."""
    src = MEMORY_PY.read_text()
    count = len(re.findall(r"\bxp\.zeros\b", src))
    assert count >= 4, (
        f"memory.py must contain ≥ 4 `xp.zeros` calls (3 scratchpads + "
        f"DDR init + DDR ensure-grow), got {count}"
    )


def test_to_host_used_at_file_io_boundary():
    """Plan acceptance: ≥ 1 `to_host` call (ddr_save_to_hex boundary)."""
    src = MEMORY_PY.read_text()
    assert "to_host" in src, (
        "memory.py must use `to_host(...)` at the file-I/O boundary "
        "(ddr_save_to_hex / ddr_load_from_hex)"
    )


def test_vram_budget_comment_present():
    """Plan acceptance: D-10 VRAM-budget comment near DDR_MEMORY."""
    src = MEMORY_PY.read_text()
    assert "D-10" in src, (
        "memory.py must reference D-10 in a comment near DDR_MEMORY.__init__"
    )
    assert "GTX_DDR_SIZE" in src, (
        "memory.py D-10 comment must name the GTX_DDR_SIZE override knob"
    )


# --------------------------------------------------------------------------
# Runtime invariants (the actual port produces working xp ndarrays)
# --------------------------------------------------------------------------

def _is_xp_ndarray(arr) -> bool:
    """Return True iff arr is an xp ndarray (numpy or cupy)."""
    return isinstance(arr, xp.ndarray)


def test_module_level_scratchpads_are_xp_uint8():
    """`_L0_GLOBAL`, `_L1_GLOBAL`, `_L2_GLOBAL` are xp uint8 arrays of the
    documented shapes."""
    NN = config_params.GTX_NEST_NUM
    SN = config_params.GTX_SPU_NUM
    L0 = config_params.GTX_L0_SIZE_BYTES
    L1 = config_params.GTX_L1_SIZE_BYTES
    L2 = config_params.GTX_L2_SIZE_BYTES

    assert _is_xp_ndarray(memmod._L0_GLOBAL)
    assert _is_xp_ndarray(memmod._L1_GLOBAL)
    assert _is_xp_ndarray(memmod._L2_GLOBAL)

    assert memmod._L0_GLOBAL.dtype == xp.uint8
    assert memmod._L1_GLOBAL.dtype == xp.uint8
    assert memmod._L2_GLOBAL.dtype == xp.uint8

    assert memmod._L0_GLOBAL.shape == (NN, SN, L0)
    assert memmod._L1_GLOBAL.shape == (NN, SN, L1)
    assert memmod._L2_GLOBAL.shape == (NN, L2)


def test_l1_byte_and_l1_f16_alias_same_storage():
    """Plan Test 1 `test_memory_layout`: writing 0x3C00 to l1_byte
    produces bytes `[0x00, 0x3C]` (LE); reading the same offset through
    l1_f16 yields `float16(1.0)`."""
    mem = GtxMemory()

    # Zero the scratchpads first (module-level globals are aliased across
    # GtxMemory instances; another test could have written to them).
    mem.reset_scratchpads()

    nest, spu = 0, 0
    off = 64  # arbitrary 2-byte-aligned offset

    l1_byte = mem.l1_byte(nest, spu)
    l1_f16 = mem.l1_f16(nest, spu)

    assert _is_xp_ndarray(l1_byte), f"l1_byte must return xp.ndarray, got {type(l1_byte)}"
    assert _is_xp_ndarray(l1_f16), f"l1_f16 must return xp.ndarray, got {type(l1_f16)}"
    assert l1_byte.dtype == xp.uint8
    assert l1_f16.dtype == xp.float16

    # Write fp16(1.0) via byte view (LE = [0x00, 0x3C]).
    l1_byte[off] = 0x00
    l1_byte[off + 1] = 0x3C

    # Read back via fp16 view at the same offset (off bytes / 2 = halfword index).
    val = float(l1_f16[off // 2])
    assert val == 1.0, f"expected fp16(1.0) round-trip, got {val!r}"

    # Reverse direction: write through fp16 view, read through byte view.
    mem.reset_scratchpads()
    l1_f16[off // 2] = xp.float16(-2.0)
    # -2.0 in IEEE 754 binary16 = 0xC000 (sign=1, exp=0x10, mantissa=0)
    # LE bytes: [0x00, 0xC0].
    assert int(l1_byte[off]) == 0x00, f"got byte[off]={l1_byte[off]:#x}"
    assert int(l1_byte[off + 1]) == 0xC0, f"got byte[off+1]={l1_byte[off+1]:#x}"


def test_l0_byte_and_l0_f16_alias_same_storage():
    """Same alias contract as L1, but for L0."""
    mem = GtxMemory()
    mem.reset_scratchpads()

    nest, spu = 1, 5
    off = 16

    l0_byte = mem.l0_byte(nest, spu)
    l0_f16 = mem.l0_f16(nest, spu)
    assert _is_xp_ndarray(l0_byte)
    assert _is_xp_ndarray(l0_f16)

    l0_byte[off] = 0x00
    l0_byte[off + 1] = 0x3C
    assert float(l0_f16[off // 2]) == 1.0


def test_l2_byte_and_l2_f16_alias_same_storage():
    """L2 has no SPU axis (1 buffer per NEST)."""
    mem = GtxMemory()
    mem.reset_scratchpads()

    nest = 2
    off = 128

    l2_byte = mem.l2_byte(nest)
    l2_f16 = mem.l2_f16(nest)
    assert _is_xp_ndarray(l2_byte)
    assert _is_xp_ndarray(l2_f16)

    l2_byte[off] = 0x00
    l2_byte[off + 1] = 0x3C
    assert float(l2_f16[off // 2]) == 1.0


# --------------------------------------------------------------------------
# DDR_MEMORY runtime invariants
# --------------------------------------------------------------------------

def test_ddr_memory_init_returns_xp_uint8():
    """DDR_MEMORY allocates its backing store via `xp.zeros(..., dtype=xp.uint8)`."""
    ddr = DDR_MEMORY(size=4096)
    backing = ddr.raw()
    assert _is_xp_ndarray(backing)
    assert backing.dtype == xp.uint8
    assert backing.shape == (4096,)


def test_ddr_grow_doubling_preserves_prior_bytes():
    """Plan Test 2 `test_ddr_grow`: `ensure()` doubling-grow preserves prior
    data past the initial floor."""
    ddr = DDR_MEMORY(size=64)
    # Stamp 64 known bytes.
    for i in range(64):
        ddr._bytes[i] = i & 0xFF

    # Grow well past the initial floor (INITIAL_FLOOR = 1 MiB), forcing the
    # doubling allocator AND a slice-copy of the prior bytes.
    end_offset = 2 * 1024 * 1024  # 2 MiB
    new_buf = ddr.ensure(end_offset)
    assert _is_xp_ndarray(new_buf)
    assert new_buf is ddr._bytes  # ensure returns the live backing
    assert ddr._bytes.shape[0] >= end_offset

    # Prior bytes preserved.
    for i in range(64):
        assert int(ddr._bytes[i]) == (i & 0xFF), (
            f"byte {i} lost across grow: got {int(ddr._bytes[i]):#x}, "
            f"expected {i & 0xFF:#x}"
        )
    # Newly allocated region is zero.
    assert int(ddr._bytes[1024]) == 0


# --------------------------------------------------------------------------
# File-I/O boundary: ddr_save_to_hex / ddr_load_from_hex use to_host bridge
# --------------------------------------------------------------------------

def test_ddr_save_to_hex_writes_expected_bytes(tmp_path):
    """Plan Test 3 `test_ddr_save_to_hex_xp_aware`: file bytes are
    backend-agnostic (via to_host)."""
    mem = GtxMemory()
    mem.ensure_ddr(64)
    base_addr = config_params.GTX_DDR_BASE
    # Stamp a recognisable pattern in DDR at offset 0 (base address).
    for i in range(32):
        mem.ddr._bytes[i] = i & 0xFF

    out_path = tmp_path / "ddr.hex"
    mem.ddr_save_to_hex(str(out_path), base_addr, 32)

    text = out_path.read_text().strip()
    # The file format reverses each 32-byte chunk before hex-encoding.
    # Bytes 0..31 reversed = 31..0; hex string is high-byte first.
    expected = bytes(reversed(range(32))).hex()
    assert text == expected, (
        f"ddr_save_to_hex mismatch.\nexpected: {expected!r}\nactual:   {text!r}"
    )


def test_ddr_load_from_hex_roundtrip(tmp_path):
    """ddr_load_from_hex parses the same format ddr_save_to_hex produces."""
    mem_a = GtxMemory()
    mem_a.ensure_ddr(64)
    base_addr = config_params.GTX_DDR_BASE
    for i in range(32):
        mem_a.ddr._bytes[i] = (0xA0 + i) & 0xFF

    path = tmp_path / "ddr_rt.hex"
    mem_a.ddr_save_to_hex(str(path), base_addr, 32)

    mem_b = GtxMemory()
    mem_b.ensure_ddr(64)
    # ddr_load_from_hex starts at offset 0 (raw byte stream — no `@` header
    # in our saved file).
    mem_b.ddr_load_from_hex(str(path))

    # Read back first 32 bytes of mem_b's DDR and compare with mem_a's.
    a_bytes = bytes(int(x) for x in mem_a.ddr._bytes[:32])
    b_bytes = bytes(int(x) for x in mem_b.ddr._bytes[:32])
    assert a_bytes == b_bytes, (
        f"ddr roundtrip mismatch:\n  saved : {a_bytes.hex()}\n  loaded: {b_bytes.hex()}"
    )
