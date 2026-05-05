---
phase: 03-dma-ddr-i-o
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/ddr.py
  - tests/gtx/test_ddr_modes.py
autonomous: true
requirements: [DMA-04]

must_haves:
  truths:
    - "ensure_ddr uses doubling-grow strategy with INITIAL_FLOOR=1 MiB; cap = get_ddr_cap()."
    - "ddr_init_from_file parses @offset lines, hex lines (any nbytes <= 32), with GTX_DDR_REVERSED env var read per-call (D-08)."
    - "ddr_dump_to_file writes 32 bytes/line, zero-pad on out-of-range, GTX_DDR_REVERSED reverses byte order within each 32-byte chunk."
    - "GTX_DDR_REVERSED=1 dumped file's first hex line is byte-reversed of LTR dump first line: bytes.fromhex(rev) == bytes.fromhex(ltr)[::-1]."
    - "Each mode round-trips: dump in mode X, init in mode X, bytes match original."
    - "ddr_dump_to_file accepts addr/size as args only — no env-var read for these (D-09)."
  artifacts:
    - path: "src/main/python/riscv/gtx/ddr.py"
      provides: "ensure_ddr (doubling-grow) + ddr_init_from_file + ddr_dump_to_file + INITIAL_FLOOR constant"
      contains: "INITIAL_FLOOR"
      contains_2: "def ddr_init_from_file"
      contains_3: "def ddr_dump_to_file"
      min_lines: 130
    - path: "tests/gtx/test_ddr_modes.py"
      provides: "DMA-04 unit tests: 8 tests covering ensure_ddr doubling, LTR/REVERSED dump, half-density init, round-trip both modes, ddr_dump_to_file ignores GTX_DDR_DUMP* env vars (D-09)"
      min_lines: 200
  key_links:
    - from: "ensure_ddr"
      to: "doubling-grow allocator"
      via: "new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)"
      pattern: "current_size \\* 2"
    - from: "ddr_init_from_file / ddr_dump_to_file"
      to: "os.environ.get('GTX_DDR_REVERSED')"
      via: "per-call read inside the function (D-08)"
      pattern: "os\\.environ\\.get\\(['\\\"]GTX_DDR_REVERSED['\\\"]\\)"
    - from: "ddr_dump_to_file"
      to: "32-byte chunk reversal in REVERSED mode"
      via: "chunk[::-1] before .hex()"
      pattern: "chunk\\[::-1\\]"
---

<objective>
Replace Phase 1 stub `ensure_ddr` with the C-divergent doubling-grow allocator
(per RESEARCH "ensure_ddr semantics divergence" — C++ allocates full 4 GiB, but
P3 keeps doubling-grow as a CI ergonomic). Add `ddr_init_from_file` and
`ddr_dump_to_file` (per CONTEXT D-07 location: `riscv/gtx/ddr.py`). Both DDR I/O
functions read `GTX_DDR_REVERSED` per-call (D-08), and `ddr_dump_to_file` does
NOT consult `GTX_DDR_DUMP_*` env vars (D-09 — pure library function, CLI is P6).

Purpose: Land the byte-domain DDR I/O layer with both LTR and reversed parsing
modes so subsequent DMA round-trip tests (Plan 5) and P4/P5 .elf regression
fixtures can ingest C++ golden hex dumps. This is independent of Plans 01/02 —
no shared file boundaries.

Output: `ddr.py` (~130 LOC, replacing the existing 78), populated
`test_ddr_modes.py` (Wave 0 scaffold filled).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md
@.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md
@.planning/phases/03-dma-ddr-i-o/03-VALIDATION.md

@src/main/python/riscv/gtx/ddr.py
@src/main/python/riscv/gtx/memory.py
@src/main/python/riscv/gtx/params.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc
@vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc

<interfaces>
From src/main/python/riscv/gtx/ddr.py (existing P1 — to be extended):
```python
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024
def get_ddr_cap() -> int  # reads GTX_DDR_SIZE env var; supports K/M/G suffix
def ensure_ddr(mem: GtxMemory, end_offset: int) -> np.ndarray  # P1 stub: linear grow
```

From src/main/python/riscv/gtx/params.py (Plan 01 added):
```python
GTX_DDR_BASE: int = 0x370000000
GTX_DDR_BUS_WORD_BYTES: int = 32
```

From src/main/python/riscv/gtx/memory.py:
```python
class GtxMemory:
    _ddr_bytes: Optional[np.ndarray]   # None until ensure_ddr() called
```

C++ reference signatures (gtx_npu_dma.cc:438-558):
```c++
void gtx_npu_t::ddr_init_from_file(const char *filename)
void gtx_npu_t::ddr_dump_to_file(const char *filename, uint64_t addr, uint64_t size)
```
P3 keeps `mem` first arg (D-07: pure functions), so:
```python
def ddr_init_from_file(mem: GtxMemory, filename: str) -> None
def ddr_dump_to_file(mem: GtxMemory, filename: str, addr: int, size: int) -> None
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: ensure_ddr doubling-grow + ddr_init_from_file + ddr_dump_to_file + 8 unit tests</name>
  <files>
    src/main/python/riscv/gtx/ddr.py,
    tests/gtx/test_ddr_modes.py
  </files>
  <read_first>
    - src/main/python/riscv/gtx/ddr.py (current Phase 1 stub — line-by-line; preserve license header + DEFAULT_DDR_SIZE + get_ddr_cap)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc (lines 437-502 ddr_init_from_file canonical parser; lines 504-558 ddr_dump_to_file)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc (lines 195-205 — confirm C++ ensure_ddr is single-shot 4 GiB; this DOCUMENTS the divergence)
    - 03-RESEARCH.md "Code Examples" §2 (ddr_init_from_file canonical), §3 (ddr_dump_to_file canonical), §5 (doubling-grow ensure_ddr)
    - 03-RESEARCH.md "DDR Hex I/O" — half-density edge case + zero-pad rules
    - 03-RESEARCH.md "Common Pitfalls" #5 (ensure_ddr 4 GiB note), #6 (GTX_DDR_REVERSED cache poisoning)
    - 03-CONTEXT.md D-07 (pure functions), D-08 (per-call env read), D-09 (no env-var-read for addr/size in dump), D-13 (doubling-grow)
  </read_first>
  <behavior>
    - **ensure_ddr doubling-grow tests**:
      * After `mem = GtxMemory()`, `mem._ddr_bytes is None`.
      * `ensure_ddr(mem, 100)` returns ndarray, `mem._ddr_bytes.size == 1*1024*1024` (INITIAL_FLOOR — 1 MiB).
      * `ensure_ddr(mem, 1*1024*1024 + 1)` doubles to 2 MiB (`mem._ddr_bytes.size == 2*1024*1024`).
      * `ensure_ddr(mem, 2*1024*1024 + 1)` doubles to 4 MiB.
      * Existing bytes preserved across grow: write byte at offset 0 with value 42, grow, verify byte at offset 0 still 42.
      * Cap exceeded raises ValueError.
    - **ddr_dump_to_file LTR mode** (no GTX_DDR_REVERSED):
      * Pre-populate `mem._ddr_bytes[0:64] = np.arange(64, dtype=np.uint8)`.
      * Dump 64 bytes from addr=0. File contains 2 lines, each 64 hex chars + \n. First line = bytes.fromhex(`'00'+'01'+...+'1f'`) = bytes(range(0,32)) when read back.
    - **ddr_dump_to_file REVERSED mode** (monkeypatch.setenv('GTX_DDR_REVERSED','1')):
      * Same pre-pop. First line = bytes(range(0,32))[::-1] = bytes(range(31,-1,-1)).
      * Verifies `bytes.fromhex(rev_line) == bytes.fromhex(ltr_line)[::-1]`.
    - **ddr_dump_to_file zero-pad on out-of-range**:
      * Pre-populate only 16 bytes. Dump size=32 → file has 1 line, last 16 hex chars are '00'*32.
    - **ddr_init_from_file LTR mode**:
      * Write file: `'@10\n' + ('aa'*32) + '\n' + ('bb'*32) + '\n'`. Init. Assert `mem._ddr_bytes[0x10:0x30] == 0xAA` and `[0x30:0x50] == 0xBB`.
    - **ddr_init_from_file REVERSED**:
      * Write file: `'@0\n' + ('00'*16 + 'ff'*16) + '\n'`. Init with `GTX_DDR_REVERSED=1`. Verify `mem._ddr_bytes[0:32]` is the byte-reversed chunk: `bytes(b'\xff'*16 + b'\x00'*16)`.
    - **ddr_init_from_file half-density (16-byte line)**:
      * Write line with only 32 hex chars (16 bytes). Init. Assert offset advanced by 16 not 32.
    - **ddr_init_from_file # comment + empty line skip**:
      * `'# comment\n@0\n\n01'*32 + '\n'`. Init succeeds, no crash.
    - **Round-trip both modes**:
      * Pre-pop 256 bytes pattern. Dump (LTR). Reset memory. Init from same file. Bytes match.
      * Same with REVERSED — dump REV, init REV, bytes match.
    - **ddr_dump_to_file does NOT read GTX_DDR_DUMP** (D-09):
      * Set `monkeypatch.setenv('GTX_DDR_DUMP', 'should_be_ignored.hex')` and `GTX_DDR_DUMP_ADDR='0xff'`. Call `ddr_dump_to_file(mem, '/tmp/actual.hex', addr=0, size=32)`. Verify ONLY `/tmp/actual.hex` written, file at 'should_be_ignored.hex' does not exist (or use tmp_path for both). Assert `pathlib.Path('/tmp/actual.hex').exists()` and the env-var-named file absent.
  </behavior>
  <action>
1. Replace `src/main/python/riscv/gtx/ddr.py` content (preserve license header, license comment, and the existing `DEFAULT_DDR_SIZE` + `get_ddr_cap` functions). The new file:

   ```python
   #
   # Copyright 2026 WuXi EsionTech Co., Ltd.
   # ... [keep existing license header]
   #
   """DDR backing store + hex I/O — Phase 3 fills (D-07/D-08/D-09/D-13).

   Doubling-grow ensure_ddr (P3 D-13 — diverges from C++ single-shot 4 GiB
   alloc as a CI ergonomic per 03-RESEARCH 'ensure_ddr semantics divergence'.
   Production firmware that touches the full 4 GiB triggers a single grow to
   cap; small tests stay small).
   """
   from __future__ import annotations
   import os
   from typing import TYPE_CHECKING

   import numpy as np

   if TYPE_CHECKING:
       from .memory import GtxMemory

   from .params import GTX_DDR_BASE

   # D-02 default: 4 GiB
   DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024

   # D-13: floor for doubling-grow first allocation. 1 MiB picked because:
   #   - covers minimum 32-byte bus word with headroom
   #   - small enough that CI per-test allocations are cheap
   #   - large enough that "single grow per test" is the common case
   INITIAL_FLOOR: int = 1 * 1024 * 1024


   def get_ddr_cap() -> int:
       """[unchanged from P1 — preserve verbatim]"""
       val = os.environ.get("GTX_DDR_SIZE")
       if val is None:
           return DEFAULT_DDR_SIZE
       val = val.strip().upper()
       if val.endswith("G"):
           return int(val[:-1]) * 1024 ** 3
       if val.endswith("M"):
           return int(val[:-1]) * 1024 ** 2
       if val.endswith("K"):
           return int(val[:-1]) * 1024
       return int(val)


   def ensure_ddr(mem: "GtxMemory", end_offset: int) -> np.ndarray:
       """Doubling-grow DDR allocation. P3 D-13 upgrade.

       NOTE: C++ gtx_npu_t::ensure_ddr (gtx_npu_core.cc:198-203) allocates the
       full GTX_DDR_SIZE (4 GiB) once. We use doubling-grow purely as a CI/test
       ergonomic so per-test allocations stay small. For regression tests
       touching the full 4 GiB, behavior is identical (single grow to cap).
       Cap enforced via GTX_DDR_SIZE env var.
       """
       cap = get_ddr_cap()
       if end_offset > cap:
           raise ValueError(
               f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
               f"(set GTX_DDR_SIZE env var to raise)"
           )
       current_size = mem._ddr_bytes.size if mem._ddr_bytes is not None else 0
       if end_offset > current_size:
           new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)
           new_size = min(new_size, cap)
           new_arr = np.zeros(new_size, dtype=np.uint8)
           if mem._ddr_bytes is not None:
               new_arr[:current_size] = mem._ddr_bytes
           mem._ddr_bytes = new_arr
       return mem._ddr_bytes


   def _ddr_offset(addr: int) -> int:
       """Address-to-offset helper. Direct port of C++ ddr_offset()."""
       if addr >= GTX_DDR_BASE:
           return addr - GTX_DDR_BASE
       return addr


   def ddr_init_from_file(mem: "GtxMemory", filename: str) -> None:
       """Direct port of gtx_npu_dma.cc:438-502.

       Parser rules:
         - Skip empty lines and lines starting with '#'.
         - Lines starting with '@HEX' set offset = int(rest, 16).
         - Hex data lines: nbytes = min(len(line)//2, 32). Consume exactly
           nbytes bytes and advance offset by nbytes.
         - GTX_DDR_REVERSED=1: byte at hex-position (nbytes-1-i)*2 goes to
           ddr[offset + i] (right-to-left). Default LTR.

       D-08: GTX_DDR_REVERSED read per call (no module-level cache).
       """
       reversed_mode = bool(os.environ.get("GTX_DDR_REVERSED"))

       offset = 0
       with open(filename, "r") as f:
           for raw in f:
               line = raw.strip()
               if not line or line.startswith("#"):
                   continue
               if line.startswith("@"):
                   offset = int(line[1:].strip(), 16)
                   continue
               nbytes = min(len(line) // 2, 32)
               if nbytes == 0:
                   continue
               chunk = bytes.fromhex(line[: nbytes * 2])
               if reversed_mode:
                   chunk = chunk[::-1]
               ensure_ddr(mem, offset + nbytes)
               mem._ddr_bytes[offset : offset + nbytes] = np.frombuffer(
                   chunk, dtype=np.uint8
               )
               offset += nbytes


   def ddr_dump_to_file(mem: "GtxMemory", filename: str,
                         addr: int, size: int) -> None:
       """Direct port of gtx_npu_dma.cc:509-558.

       D-09: only `addr` and `size` are accepted as args — does NOT consult
       GTX_DDR_DUMP / GTX_DDR_DUMP_ADDR / GTX_DDR_DUMP_SIZE env vars (those
       are CLI/P6 territory). D-08: GTX_DDR_REVERSED read per call.
       """
       reversed_mode = bool(os.environ.get("GTX_DDR_REVERSED"))
       off = _ddr_offset(addr)

       if mem._ddr_bytes is None:
           # Match C++ has_ddr() check — write empty file (or skip).
           # Mimic C++ which returns silently; we still create the file
           # so callers can do path checks.
           with open(filename, "w"):
               pass
           return

       ddr_size = mem._ddr_bytes.size
       with open(filename, "w") as f:
           for i in range(0, size, 32):
               chunk_off = off + i
               # Build 32-byte chunk with zero-pad on out-of-range
               chunk = bytearray(32)
               for j in range(32):
                   src_idx = chunk_off + j
                   if 0 <= src_idx < ddr_size:
                       chunk[j] = int(mem._ddr_bytes[src_idx])
                   # else: leave 0 (zero-pad)
               if reversed_mode:
                   chunk = bytes(chunk)[::-1]
               else:
                   chunk = bytes(chunk)
               f.write(chunk.hex() + "\n")
   ```

2. Populate `tests/gtx/test_ddr_modes.py`. NO `_RISCV_AVAILABLE` skipif (DDR I/O is pure-Python). Tests:

   ```python
   import os
   import pathlib
   import numpy as np
   import pytest
   from riscv.gtx.memory import GtxMemory
   from riscv.gtx.ddr import (
       DEFAULT_DDR_SIZE, INITIAL_FLOOR,
       ensure_ddr, get_ddr_cap,
       ddr_init_from_file, ddr_dump_to_file,
   )


   def test_ensure_ddr_initial_floor_is_1_mib():
       mem = GtxMemory()
       assert mem._ddr_bytes is None
       ensure_ddr(mem, 100)
       assert mem._ddr_bytes is not None
       assert mem._ddr_bytes.size == INITIAL_FLOOR == 1024 * 1024


   def test_ensure_ddr_doubles_on_grow():
       mem = GtxMemory()
       ensure_ddr(mem, 100)
       assert mem._ddr_bytes.size == INITIAL_FLOOR
       ensure_ddr(mem, INITIAL_FLOOR + 1)
       assert mem._ddr_bytes.size == 2 * INITIAL_FLOOR
       ensure_ddr(mem, 2 * INITIAL_FLOOR + 1)
       assert mem._ddr_bytes.size == 4 * INITIAL_FLOOR


   def test_ensure_ddr_preserves_existing_bytes():
       mem = GtxMemory()
       ensure_ddr(mem, 100)
       mem._ddr_bytes[0] = 42
       mem._ddr_bytes[INITIAL_FLOOR - 1] = 84
       ensure_ddr(mem, INITIAL_FLOOR + 1)
       assert mem._ddr_bytes[0] == 42
       assert mem._ddr_bytes[INITIAL_FLOOR - 1] == 84
       assert mem._ddr_bytes.size == 2 * INITIAL_FLOOR


   def test_ensure_ddr_cap_exceeded_raises(monkeypatch):
       monkeypatch.setenv("GTX_DDR_SIZE", "256K")
       mem = GtxMemory()
       with pytest.raises(ValueError, match="exceeds cap"):
           ensure_ddr(mem, 1024 * 1024)


   def test_ddr_dump_ltr_default_byte_order(tmp_path, monkeypatch):
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
       monkeypatch.setenv("GTX_DDR_REVERSED", "1")
       mem = GtxMemory()
       ensure_ddr(mem, 64)
       mem._ddr_bytes[:64] = np.arange(64, dtype=np.uint8)
       out = tmp_path / "rev.hex"
       ddr_dump_to_file(mem, str(out), 0, 64)
       lines = out.read_text().splitlines()
       assert bytes.fromhex(lines[0]) == bytes(range(0, 32))[::-1]
       assert bytes.fromhex(lines[1]) == bytes(range(32, 64))[::-1]


   def test_ddr_dump_modes_differ(tmp_path, monkeypatch):
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
       assert bytes.fromhex(rev_first) == bytes.fromhex(ltr_first)[::-1]


   def test_ddr_dump_zero_pad_on_out_of_range(tmp_path, monkeypatch):
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
       assert chunk[16:] == bytes(16)  # 16 zero bytes


   def test_ddr_init_ltr(tmp_path, monkeypatch):
       monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
       inp = tmp_path / "in.hex"
       inp.write_text("@10\n" + ("aa" * 32) + "\n" + ("bb" * 32) + "\n")
       mem = GtxMemory()
       ddr_init_from_file(mem, str(inp))
       assert mem._ddr_bytes is not None
       assert (mem._ddr_bytes[0x10:0x30] == 0xAA).all()
       assert (mem._ddr_bytes[0x30:0x50] == 0xBB).all()


   def test_ddr_init_reversed(tmp_path, monkeypatch):
       monkeypatch.setenv("GTX_DDR_REVERSED", "1")
       inp = tmp_path / "rev_in.hex"
       inp.write_text("@0\n" + ("00" * 16 + "ff" * 16) + "\n")
       mem = GtxMemory()
       ddr_init_from_file(mem, str(inp))
       expected = bytes(b"\xff" * 16 + b"\x00" * 16)  # reversed of 00*16 ff*16
       assert bytes(mem._ddr_bytes[0:32]) == expected


   def test_ddr_init_half_density_16_bytes(tmp_path, monkeypatch):
       monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
       inp = tmp_path / "half.hex"
       inp.write_text("@0\n" + ("aa" * 16) + "\n" + ("bb" * 16) + "\n")
       mem = GtxMemory()
       ddr_init_from_file(mem, str(inp))
       # First line wrote 16 bytes at 0x00..0x10; second line at 0x10..0x20
       assert (mem._ddr_bytes[0x00:0x10] == 0xAA).all()
       assert (mem._ddr_bytes[0x10:0x20] == 0xBB).all()


   def test_ddr_init_skips_comments_and_empty(tmp_path, monkeypatch):
       monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
       inp = tmp_path / "comments.hex"
       inp.write_text("# top comment\n\n@0\n# inner\n" + ("11" * 32) + "\n")
       mem = GtxMemory()
       ddr_init_from_file(mem, str(inp))
       assert (mem._ddr_bytes[0x00:0x20] == 0x11).all()


   def test_ddr_round_trip_ltr(tmp_path, monkeypatch):
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


   def test_ddr_dump_ignores_dump_addr_size_env_vars(tmp_path, monkeypatch):
       """D-09: ddr_dump_to_file does NOT read GTX_DDR_DUMP / _ADDR / _SIZE."""
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
       assert not (tmp_path / "should_not_be_used.hex").exists()
       # And file content honors the args, not env: 32 bytes from addr=0
       assert bytes.fromhex(actual.read_text().strip()) == bytes(range(32))


   def test_ddr_reversed_env_read_per_call(tmp_path, monkeypatch):
       """D-08: env read at every call. Toggle between calls; each honors current state."""
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

       assert a.read_text() == c.read_text()    # LTR matches LTR
       assert a.read_text() != b.read_text()    # LTR differs from REV
   ```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_ddr_modes.py -x --noconftest -o "addopts="</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E "INITIAL_FLOOR: int = 1 \* 1024 \* 1024" src/main/python/riscv/gtx/ddr.py` matches.
    - `grep -E "current_size \* 2" src/main/python/riscv/gtx/ddr.py` matches (doubling-grow).
    - `grep -E "def ddr_init_from_file" src/main/python/riscv/gtx/ddr.py` matches.
    - `grep -E "def ddr_dump_to_file" src/main/python/riscv/gtx/ddr.py` matches.
    - `grep -c "os.environ.get(\"GTX_DDR_REVERSED\")\|os.environ.get('GTX_DDR_REVERSED')" src/main/python/riscv/gtx/ddr.py` returns 2 (both functions read per-call).
    - `grep -c "GTX_DDR_DUMP" src/main/python/riscv/gtx/ddr.py` returns 0 (D-09: no env-var-read for dump).
    - All ~16 ddr_modes tests pass.
  </acceptance_criteria>
  <done>DDR I/O layer is live. ensure_ddr does doubling-grow with INITIAL_FLOOR=1MiB. Both DDR I/O functions honor GTX_DDR_REVERSED per-call (D-08). Dump function takes addr/size as args, ignores GTX_DDR_DUMP_* env vars (D-09). DMA-04 satisfied. Plan 5 round-trip uses these as building blocks.</done>
</task>

</tasks>

<verification>
- ensure_ddr doubling-grow strategy confirmed (1 MiB → 2 MiB → 4 MiB).
- LTR vs REVERSED dumps differ AND each round-trips.
- Half-density (16-byte) hex line parsing works.
- `@offset` line + `#` comments + empty lines all handled.
- D-08 (per-call env read) verified by toggling GTX_DDR_REVERSED between calls.
- D-09 (no env-var-read for addr/size) verified by setting bogus env values and confirming actual args win.
</verification>

<success_criteria>
- `pytest tests/gtx/test_ddr_modes.py -x --noconftest -o "addopts="` returns 0 with all ~16 tests green.
- ddr.py LOC roughly 130-150 (was 78 in P1).
- DMA-04 covered: round-trip both modes pass, modes verifiably differ.
</success_criteria>

<output>
After completion, create `.planning/phases/03-dma-ddr-i-o/03-03-SUMMARY.md` documenting:
- Final ddr.py LOC.
- INITIAL_FLOOR rationale (1 MiB picked per RESEARCH).
- Confirmed C++ divergence (doubling-grow vs single-shot 4 GiB) is documented in the docstring.
- Half-density edge case test result (P3 dumper does NOT produce half-density output, per RESEARCH §"Half-density edge case").
</output>
</content>
</invoke>