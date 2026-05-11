"""DDR backing store + hex I/O — Phase 3 fills (D-07/D-08/D-09/D-13).

Doubling-grow ensure_ddr (P3 D-13 — diverges from C++ single-shot 4 GiB
alloc as a CI ergonomic per 03-RESEARCH 'ensure_ddr semantics divergence'.
Production firmware that touches the full 4 GiB triggers a single grow to
cap; small tests stay small).
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

import torch

from .memory import GtxMemory
from .params import GTX_DDR_BASE, DEFAULT_DDR_SIZE, INITIAL_FLOOR

def maximum_ddr() -> int:
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

def ensure_ddr(mem: GtxMemory, end_offset: int) -> torch.Tensor:
    cap = maximum_ddr()
    if end_offset > cap:
        raise ValueError(
            f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
            f"(set GTX_DDR_SIZE env var to raise)"
        )
    current_size = len(mem._ddr_bytes) if mem._ddr_bytes is not None else 0
    if end_offset > current_size:
        new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)
        new_size = min(new_size, cap)
        new_arr = torch.zeros(new_size, dtype=torch.uint8)
        if mem._ddr_bytes is not None:
            new_arr[:current_size] = mem._ddr_bytes
        mem._ddr_bytes = new_arr
    return mem._ddr_bytes


def _ddr_offset(addr: int) -> int:
    """Address-to-offset helper. Direct port of C++ ddr_offset()."""
    if addr >= GTX_DDR_BASE:
        return addr - GTX_DDR_BASE
    return addr


def ddr_load_from_hex(mem: GtxMemory, filename: str) -> None:
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
            chunk = chunk[::-1]
            ensure_ddr(mem, offset + nbytes)
            # torch.frombuffer requires writable buffer; bytes is read-only,
            # so go via bytearray (copy is cheap for 32-byte chunks).
            mem._ddr_bytes[offset : offset + nbytes] = torch.frombuffer(
                bytearray(chunk), dtype=torch.uint8
            )
            offset += nbytes


def ddr_save_to_hex(mem: GtxMemory, filename: str,
                     addr: int, size: int) -> None:
    off = _ddr_offset(addr)

    ddr_src = mem._ddr_bytes
    ddr_size = len(ddr_src)
    with open(filename, "w") as f:
        for i in range(0, size, 32):
            chunk_off = off + i
            # Build 32-byte chunk with zero-pad on out-of-range
            buf = bytearray(32)
            for j in range(32):
                src_idx = chunk_off + j
                if 0 <= src_idx < ddr_size:
                    buf[j] = int(ddr_src[src_idx])
                # else: leave 0 (zero-pad)
            chunk = bytes(buf)
            chunk = chunk[::-1]
            f.write(chunk.hex() + "\n")
