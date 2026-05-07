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
    from .memory import GtxMemory   # avoid circular import at runtime

from .params import GTX_DDR_BASE

# D-02 default: 4 GiB
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024

# D-13: floor for doubling-grow first allocation. 1 MiB picked because:
#   - covers 32-byte bus-word minimum with ample headroom
#   - small enough that CI per-test allocations are cheap
#   - large enough that "single grow per test" is the common case
INITIAL_FLOOR: int = 1 * 1024 * 1024


def get_ddr_cap() -> int:
    """Read GTX_DDR_SIZE env var; default 4GB. Supports 'G'/'M'/'K' suffixes.

    Examples: '4G' -> 4*1024**3, '64M' -> 64*1024**2, '1024K' -> 1024*1024.
    """
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

    Strategy: new_size = min(cap, max(end_offset, current_size * 2, INITIAL_FLOOR)).
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
    any dump-related env vars (those are CLI/P6 territory).
    D-08: GTX_DDR_REVERSED read per call.

    Output format: 32 bytes/line, hex-encoded, '\\n'-terminated.
    Out-of-range bytes are zero-padded (matches C++ idx>=GTX_DDR_SIZE branch).
    """
    reversed_mode = bool(os.environ.get("GTX_DDR_REVERSED"))
    off = _ddr_offset(addr)

    if mem._ddr_bytes is None:
        # Match C++ has_ddr() check — write empty file so callers can do
        # path/exists checks without special-casing.
        with open(filename, "w"):
            pass
        return

    ddr_size = mem._ddr_bytes.size
    with open(filename, "w") as f:
        for i in range(0, size, 32):
            chunk_off = off + i
            # Build 32-byte chunk with zero-pad on out-of-range
            buf = bytearray(32)
            for j in range(32):
                src_idx = chunk_off + j
                if 0 <= src_idx < ddr_size:
                    buf[j] = int(mem._ddr_bytes[src_idx])
                # else: leave 0 (zero-pad)
            chunk = bytes(buf)
            if reversed_mode:
                chunk = chunk[::-1]
            f.write(chunk.hex() + "\n")


def _atexit_ddr_dump() -> None:
    """atexit handler: vendor gtx_npu_core.cc:61-73 1:1 port (D-05).

    Triggered at Python interpreter shutdown when GTX_DDR_DUMP env var is set
    (registration is gated in __init__.py per D-04). Reads ADDR/SIZE from env
    vars and writes DDR slice to file via existing args-only ddr_dump_to_file.

    Single-NPU model (D-05 / RESEARCH §NPU Instance Lookup): looks up
    riscv.gtx.npu._LAST_NPU module global. v1 single-hart scope.

    SAFETY (RESEARCH §Pitfall 3, CPython issue #103512): this function MUST
    NOT raise SystemExit. All error paths use early `return` and
    `print(..., file=sys.stderr)`.
    """
    # Lazy-import the npu MODULE (not _LAST_NPU directly) to capture the live
    # value at hook-fire time. `from .npu import _LAST_NPU` would bind the
    # value AT IMPORT TIME (None), missing any later GtxNpu.__init__ assignment.
    from . import npu as _npu_mod
    last_npu = _npu_mod._LAST_NPU
    if last_npu is None:
        return  # No NPU was instantiated — nothing to dump
    if last_npu.mem is None:
        return  # No memory subsystem — defensive

    # P3 D-05: flush deferred S-loop stores before dumping (mirrors vendor
    # gtx_npu_core.cc:64 flush_deferred_ddr_stores()).
    last_npu.flush_deferred_ddr_stores()

    dump_file = os.environ.get('GTX_DDR_DUMP')
    if not dump_file:
        return  # Defensive — registration is gated, but env may have changed.

    # Vendor gtx_npu_core.cc:68-71: hex parse with stoull base=16; defaults
    # 0x37f000000 (= GTX_DDR_BASE + 0x0F000000) and 0x400 (= 1024 bytes).
    addr_s = os.environ.get('GTX_DDR_DUMP_ADDR')
    size_s = os.environ.get('GTX_DDR_DUMP_SIZE')
    try:
        addr = int(addr_s, 16) if addr_s else 0x37f000000
        size = int(size_s, 16) if size_s else 0x400
    except ValueError as exc:
        import sys
        print("GTX_DDR_DUMP: invalid hex int in env vars (" + str(exc) + "); skip dump",
              file=sys.stderr)
        return

    # Ensure DDR is allocated for the requested dump range. Vendor C++ pre-
    # allocates DDR in the constructor (gtx_npu_core.cc:167 ensure_ddr() in
    # reset()) so has_ddr() is always true. Our pyspike lazy-allocates per
    # P3 D-13 doubling-grow ergonomic; here we materialize the requested
    # range so a clean firmware that never touches DDR still produces a
    # zero-padded dump file (matches vendor `ddr_dump_to_file` zero-pad
    # branch at gtx_npu_dma.cc:509-558 + ensures has_ddr() parity).
    end_offset = _ddr_offset(addr) + size
    ensure_ddr(last_npu.mem, end_offset)

    ddr_dump_to_file(last_npu.mem, dump_file, addr, size)
