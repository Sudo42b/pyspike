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
"""Phase 3 plan 03 — DMA-04 unit tests.

Coverage:
  - ensure_ddr doubling-grow (INITIAL_FLOOR=1 MiB, doubles, preserves bytes,
    cap raises ValueError) — D-13
  - ddr_dump_to_file LTR mode (default) — D-08, D-09
  - ddr_dump_to_file REVERSED mode (GTX_DDR_REVERSED=1)
  - ddr_dump_to_file zero-pad on out-of-range
  - ddr_dump_to_file does NOT consult GTX_DDR_DUMP* env vars — D-09
  - ddr_dump_to_file reads GTX_DDR_REVERSED per-call (D-08 toggling)
  - ddr_init_from_file LTR + REVERSED + half-density + comments + @offset
  - Round-trip both modes (dump→reset→init→bit-exact)

DDR I/O is pure-python; no `_RISCV_AVAILABLE` skip required (D-07).
"""
import os
import pathlib

import numpy as np
import pytest

from riscv.gtx.memory import GtxMemory
from riscv.gtx.ddr import (
    DEFAULT_DDR_SIZE,
    INITIAL_FLOOR,
    ensure_ddr,
    get_ddr_cap,
    ddr_init_from_file,
    ddr_dump_to_file,
)


# ============================================================================
# ensure_ddr doubling-grow (D-13)
# ============================================================================

def test_ensure_ddr_initial_floor_is_1_mib():
    """First ensure_ddr call allocates exactly INITIAL_FLOOR (1 MiB)."""
    mem = GtxMemory()
    assert mem._ddr_bytes is None
    ensure_ddr(mem, 100)
    assert mem._ddr_bytes is not None
    assert INITIAL_FLOOR == 1024 * 1024
    assert mem._ddr_bytes.size == INITIAL_FLOOR


def test_ensure_ddr_doubles_on_grow():
    """Each grow doubles capacity (1 MiB -> 2 MiB -> 4 MiB)."""
    mem = GtxMemory()
    ensure_ddr(mem, 100)
    assert mem._ddr_bytes.size == INITIAL_FLOOR
    ensure_ddr(mem, INITIAL_FLOOR + 1)
    assert mem._ddr_bytes.size == 2 * INITIAL_FLOOR
    ensure_ddr(mem, 2 * INITIAL_FLOOR + 1)
    assert mem._ddr_bytes.size == 4 * INITIAL_FLOOR


def test_ensure_ddr_preserves_existing_bytes():
    """Grow keeps already-written bytes intact at their original offsets."""
    mem = GtxMemory()
    ensure_ddr(mem, 100)
    mem._ddr_bytes[0] = 42
    mem._ddr_bytes[INITIAL_FLOOR - 1] = 84
    ensure_ddr(mem, INITIAL_FLOOR + 1)
    assert mem._ddr_bytes.size == 2 * INITIAL_FLOOR
    assert mem._ddr_bytes[0] == 42
    assert mem._ddr_bytes[INITIAL_FLOOR - 1] == 84


def test_ensure_ddr_cap_exceeded_raises(monkeypatch):
    """end_offset > GTX_DDR_SIZE cap raises ValueError (D-02 preserved)."""
    monkeypatch.setenv("GTX_DDR_SIZE", "256K")
    mem = GtxMemory()
    with pytest.raises(ValueError, match="exceeds cap"):
        ensure_ddr(mem, 1024 * 1024)


def test_ensure_ddr_idempotent_when_within_capacity():
    """Calling ensure_ddr with end_offset <= current size is a no-op."""
    mem = GtxMemory()
    ensure_ddr(mem, 100)
    arr1 = mem._ddr_bytes
    ensure_ddr(mem, 50)
    # Same backing array, no realloc
    assert mem._ddr_bytes is arr1
    assert mem._ddr_bytes.size == INITIAL_FLOOR


# ============================================================================
# ddr_dump_to_file LTR / REVERSED modes
# ============================================================================

def test_ddr_dump_ltr_default_byte_order(tmp_path, monkeypatch):
    """Default LTR: line N contains bytes[N*32..(N+1)*32] in natural order."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    mem = GtxMemory()
    ensure_ddr(mem, 64)
    mem._ddr_bytes[:64] = np.arange(64, dtype=np.uint8)
    out = tmp_path / "ltr.hex"
    ddr_dump_to_file(mem, str(out), 0, 64)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert bytes.fromhex(lines[0]) == bytes(range(0, 32))
    assert bytes.fromhex(lines[1]) == bytes(range(32, 64))


def test_ddr_dump_reversed_chunk_byte_order(tmp_path, monkeypatch):
    """GTX_DDR_REVERSED=1: each 32-byte chunk written reversed."""
    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    mem = GtxMemory()
    ensure_ddr(mem, 64)
    mem._ddr_bytes[:64] = np.arange(64, dtype=np.uint8)
    out = tmp_path / "rev.hex"
    ddr_dump_to_file(mem, str(out), 0, 64)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert bytes.fromhex(lines[0]) == bytes(range(0, 32))[::-1]
    assert bytes.fromhex(lines[1]) == bytes(range(32, 64))[::-1]


def test_ddr_dump_modes_differ_and_invert(tmp_path, monkeypatch):
    """REV first line is byte-reverse of LTR first line: D-08 must-have."""
    mem = GtxMemory()
    ensure_ddr(mem, 32)
    mem._ddr_bytes[:32] = np.arange(32, dtype=np.uint8)

    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    ltr = tmp_path / "ltr.hex"
    ddr_dump_to_file(mem, str(ltr), 0, 32)

    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    rev = tmp_path / "rev.hex"
    ddr_dump_to_file(mem, str(rev), 0, 32)

    ltr_first = ltr.read_text().splitlines()[0]
    rev_first = rev.read_text().splitlines()[0]
    assert ltr_first != rev_first
    # The key DMA-04 truth: REV is byte-reverse of LTR
    assert bytes.fromhex(rev_first) == bytes.fromhex(ltr_first)[::-1]


def test_ddr_dump_zero_pad_on_out_of_range(tmp_path, monkeypatch):
    """Out-of-range bytes pad with 0x00 (matches C++ idx>=GTX_DDR_SIZE branch)."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    mem = GtxMemory()
    ensure_ddr(mem, 16)
    mem._ddr_bytes[:16] = np.arange(16, dtype=np.uint8)
    out = tmp_path / "pad.hex"
    ddr_dump_to_file(mem, str(out), 0, 32)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    chunk = bytes.fromhex(lines[0])
    assert chunk[:16] == bytes(range(16))
    assert chunk[16:] == bytes(16)  # 16 zero-pad bytes


def test_ddr_dump_ignores_dump_addr_size_env_vars(tmp_path, monkeypatch):
    """D-09 truth: ddr_dump_to_file does NOT consult GTX_DDR_DUMP* env vars."""
    monkeypatch.setenv("GTX_DDR_DUMP", str(tmp_path / "should_not_be_used.hex"))
    monkeypatch.setenv("GTX_DDR_DUMP_ADDR", "0xff")
    monkeypatch.setenv("GTX_DDR_DUMP_SIZE", "0x100")
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    mem = GtxMemory()
    ensure_ddr(mem, 32)
    mem._ddr_bytes[:32] = np.arange(32, dtype=np.uint8)
    actual = tmp_path / "actual.hex"
    ddr_dump_to_file(mem, str(actual), 0, 32)
    assert actual.exists()
    # The env-var-named file must NOT be created
    assert not (tmp_path / "should_not_be_used.hex").exists()
    # File honors function args (addr=0, size=32), not env-var addr 0xff
    assert bytes.fromhex(actual.read_text().strip()) == bytes(range(32))


def test_ddr_dump_reversed_env_read_per_call(tmp_path, monkeypatch):
    """D-08: GTX_DDR_REVERSED is read at every call, not cached at import."""
    mem = GtxMemory()
    ensure_ddr(mem, 32)
    mem._ddr_bytes[:32] = np.arange(32, dtype=np.uint8)

    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    a = tmp_path / "a.hex"
    ddr_dump_to_file(mem, str(a), 0, 32)

    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    b = tmp_path / "b.hex"
    ddr_dump_to_file(mem, str(b), 0, 32)

    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    c = tmp_path / "c.hex"
    ddr_dump_to_file(mem, str(c), 0, 32)

    # Two LTR calls produce identical output; toggling between them switches mode.
    assert a.read_text() == c.read_text()
    assert a.read_text() != b.read_text()


# ============================================================================
# ddr_init_from_file parsing
# ============================================================================

def test_ddr_init_ltr(tmp_path, monkeypatch):
    """LTR init: bytes land at offsets in natural left-to-right order."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    inp = tmp_path / "in.hex"
    inp.write_text("@10\n" + ("aa" * 32) + "\n" + ("bb" * 32) + "\n")
    mem = GtxMemory()
    ddr_init_from_file(mem, str(inp))
    assert mem._ddr_bytes is not None
    assert (mem._ddr_bytes[0x10:0x30] == 0xAA).all()
    assert (mem._ddr_bytes[0x30:0x50] == 0xBB).all()


def test_ddr_init_reversed(tmp_path, monkeypatch):
    """REVERSED init: chunk byte-reversed before storing — '00..ff..' → 'ff..00..'."""
    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    inp = tmp_path / "rev_in.hex"
    inp.write_text("@0\n" + ("00" * 16 + "ff" * 16) + "\n")
    mem = GtxMemory()
    ddr_init_from_file(mem, str(inp))
    expected = bytes(b"\xff" * 16 + b"\x00" * 16)
    assert bytes(mem._ddr_bytes[0:32]) == expected


def test_ddr_init_half_density_16_bytes(tmp_path, monkeypatch):
    """Half-density: 16-byte (32-hex-char) lines advance offset by 16, not 32."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    inp = tmp_path / "half.hex"
    inp.write_text("@0\n" + ("aa" * 16) + "\n" + ("bb" * 16) + "\n")
    mem = GtxMemory()
    ddr_init_from_file(mem, str(inp))
    assert (mem._ddr_bytes[0x00:0x10] == 0xAA).all()
    assert (mem._ddr_bytes[0x10:0x20] == 0xBB).all()


def test_ddr_init_skips_comments_and_empty(tmp_path, monkeypatch):
    """Empty lines and # comments are skipped without affecting offset."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    inp = tmp_path / "comments.hex"
    inp.write_text("# top comment\n\n@0\n# inner\n" + ("11" * 32) + "\n")
    mem = GtxMemory()
    ddr_init_from_file(mem, str(inp))
    assert (mem._ddr_bytes[0x00:0x20] == 0x11).all()


# ============================================================================
# Round-trip both modes (dump → init → bit-exact)
# ============================================================================

def test_ddr_round_trip_ltr(tmp_path, monkeypatch):
    """LTR round-trip: dump LTR, init LTR, bytes match original."""
    monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
    mem1 = GtxMemory()
    ensure_ddr(mem1, 256)
    pattern = np.arange(256, dtype=np.uint8)
    mem1._ddr_bytes[:256] = pattern
    hexf = tmp_path / "rt.hex"
    ddr_dump_to_file(mem1, str(hexf), 0, 256)
    mem2 = GtxMemory()
    ddr_init_from_file(mem2, str(hexf))
    assert bytes(mem2._ddr_bytes[:256]) == bytes(pattern)


def test_ddr_round_trip_reversed(tmp_path, monkeypatch):
    """REVERSED round-trip: dump REV, init REV, bytes match original.
    Two byte-reversals cancel, so the in-memory result equals input."""
    monkeypatch.setenv("GTX_DDR_REVERSED", "1")
    mem1 = GtxMemory()
    ensure_ddr(mem1, 256)
    pattern = np.arange(256, dtype=np.uint8)
    mem1._ddr_bytes[:256] = pattern
    hexf = tmp_path / "rt_rev.hex"
    ddr_dump_to_file(mem1, str(hexf), 0, 256)
    mem2 = GtxMemory()
    ddr_init_from_file(mem2, str(hexf))
    assert bytes(mem2._ddr_bytes[:256]) == bytes(pattern)
