from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, Optional

import torch

from ..config_params import GTX_DDR_BASE, DEFAULT_DDR_SIZE, INITIAL_FLOOR

from ..config_params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
)

"""DDR backing store + hex I/O — Phase 3 fills (D-07/D-08/D-09/D-13).

Doubling-grow ensure_ddr (P3 D-13 — diverges from C++ single-shot 4 GiB
alloc as a CI ergonomic per 03-RESEARCH 'ensure_ddr semantics divergence'.
Production firmware that touches the full 4 GiB triggers a single grow to
cap; small tests stay small).
"""


class MEMORY(abc.ABC):
    @abc.abstractmethod
    def getsize(self) -> int: ...

    @abc.abstractmethod
    def alloc(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
    
    @abc.abstractmethod
    def free(self) -> None: ...


class SPU_MEMORY(MEMORY):
    def __init__(self):
        self._l0_bytes = torch.zeros(GTX_L0_SIZE_BYTES, dtype=torch.uint8)
        self._l1_bytes = torch.zeros(GTX_L1_SIZE_BYTES, dtype=torch.uint8)

    def getsize(self) -> int:
        return GTX_L0_SIZE_BYTES + GTX_L1_SIZE_BYTES

    def alloc(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._l0_bytes, self._l1_bytes

    def free(self) -> None:
        self._l0_bytes.zero_()
        self._l1_bytes.zero_()


class L2_MEMORY(MEMORY):
    def __init__(self):
        self._l2_bytes = torch.zeros(GTX_L2_SIZE_BYTES, dtype=torch.uint8)

    def getsize(self) -> int:
        return GTX_L2_SIZE_BYTES

    def alloc(self) -> torch.Tensor:
        return self._l2_bytes

    def free(self) -> None:
        self._l2_bytes.zero_()


class NEST(MEMORY):
    def __init__(self):
        self.l2 = L2_MEMORY()
        self.spus = [SPU_MEMORY() for _ in range(GTX_SPU_NUM)]

    def getsize(self) -> int:
        return self.l2.getsize() + sum(spu.getsize() for spu in self.spus)

    def alloc(self, spu_id: Optional[int] = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if spu_id is None:
            return self.l2.alloc()
        return self.spus[spu_id].alloc()

    def free(self) -> None:
        self.l2.free()
        for spu in self.spus:
            spu.free()


class DDR_MEMORY(MEMORY):
    def __init__(self):
        self._ddr_bytes: Optional[torch.Tensor] = None

    @staticmethod
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

    def getsize(self) -> int:
        return len(self._ddr_bytes) if self._ddr_bytes is not None else 0

    def alloc(self, end_offset: int) -> torch.Tensor:
        cap = self.maximum_ddr()
        if end_offset > cap:
            raise ValueError(
                f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
                f"(set GTX_DDR_SIZE env var to raise)"
            )
        current_size = self.getsize()
        if end_offset > current_size:
            new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)
            new_size = min(new_size, cap)
            new_arr = torch.zeros(new_size, dtype=torch.uint8)
            if self._ddr_bytes is not None:
                new_arr[:current_size] = self._ddr_bytes
            self._ddr_bytes = new_arr
        return self._ddr_bytes

    def free(self) -> None:
        self._ddr_bytes = None


class GtxMemory(MEMORY):
    """GTX NPU memory layer — L0/L1/L2/DDR memory hierarchy."""
    def __init__(self) -> None:
        self.nests: list[NEST] = [NEST() for _ in range(GTX_NEST_NUM)]
        self.ddr = DDR_MEMORY()
        self.spr: dict[int, int] = {}

    def getsize(self) -> int:
        return sum(nest.getsize() for nest in self.nests) + self.ddr.getsize()

    def alloc(self, nest_id: int, spu_id: Optional[int] = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.nests[nest_id].alloc(spu_id)

    def free(self) -> None:
        for nest in self.nests:
            nest.free()
        self.ddr.free()
        self.spr.clear()

    @property
    def _ddr_bytes(self) -> Optional[torch.Tensor]:
        return self.ddr._ddr_bytes

    @_ddr_bytes.setter
    def _ddr_bytes(self, value: Optional[torch.Tensor]) -> None:
        self.ddr._ddr_bytes = value

    # ----- Raw byte views (D-10 low-level) -----

    def l0_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self.nests[nest].spus[spu]._l0_bytes

    def l1_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self.nests[nest].spus[spu]._l1_bytes

    def l2_byte(self, nest: int) -> torch.Tensor:
        return self.nests[nest].l2._l2_bytes

    # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----

    def l0_f16(self, nest: int, spu: int) -> torch.Tensor:
        return self.l0_byte(nest, spu).view(torch.float16)

    def l1_f16(self, nest: int, spu: int) -> torch.Tensor:
        return self.l1_byte(nest, spu).view(torch.float16)

    def l2_f16(self, nest: int) -> torch.Tensor:
        return self.l2_byte(nest).view(torch.float16)

    def ensure_ddr(self, end_offset: int) -> torch.Tensor:
        return self.ddr.alloc(end_offset)

    def _ddr_offset(self, addr: int) -> int:
        """Address-to-offset helper. Direct port of C++ ddr_offset()."""
        if addr >= GTX_DDR_BASE:
            return addr - GTX_DDR_BASE
        return addr

    def ddr_load_from_hex(self, filename: str) -> None:
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
                self.ensure_ddr(offset + nbytes)
                # torch.frombuffer requires writable buffer; bytes is read-only,
                # so go via bytearray (copy is cheap for 32-byte chunks).
                self._ddr_bytes[offset : offset + nbytes] = torch.frombuffer(
                    bytearray(chunk), dtype=torch.uint8
                )
                offset += nbytes

    def ddr_save_to_hex(self, filename: str, addr: int, size: int) -> None:
        off = self._ddr_offset(addr)

        ddr_src = self._ddr_bytes
        ddr_size = len(ddr_src) if ddr_src is not None else 0
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

    # ----- Environment-driven I/O (replaces deleted ``ddr.py`` shims) -----

    @staticmethod
    def _parse_env_int(s: str) -> int:
        s = s.strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)

    def load_via_env(self) -> bool:
        """Load DDR from a hex file pointed to by ``GTX_DDR_INIT``.

        Returns True if a load happened, False if the env var was unset/empty.
        """
        path = os.environ.get("GTX_DDR_INIT")
        if not path:
            return False
        self.ddr_load_from_hex(path)
        return True

    def dump_via_env(self) -> bool:
        """Dump a DDR region to a hex file when ``GTX_DDR_DUMP`` is set.

        Environment
            GTX_DDR_DUMP        target file path (empty/unset → no-op)
            GTX_DDR_DUMP_ADDR   starting DDR address  (default ``GTX_DDR_BASE``)
            GTX_DDR_DUMP_SIZE   number of bytes       (default current DDR size)
        """
        path = os.environ.get("GTX_DDR_DUMP")
        if not path:
            return False
        addr_str = os.environ.get("GTX_DDR_DUMP_ADDR", "")
        addr = self._parse_env_int(addr_str) if addr_str else GTX_DDR_BASE
        size_str = os.environ.get("GTX_DDR_DUMP_SIZE", "")
        size = self._parse_env_int(size_str) if size_str else self.ddr.getsize()
        if size <= 0:
            return False
        self.ddr_save_to_hex(path, addr, size)
        return True