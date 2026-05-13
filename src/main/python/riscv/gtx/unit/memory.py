from __future__ import annotations
import abc
import os
from typing import Optional

import torch

from ..config_params import GTX_DDR_BASE, DEFAULT_DDR_SIZE, INITIAL_FLOOR

from ..config_params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
    DEVICE,
)

"""GTX memory hierarchy.

L0 / L1 / L2 scratchpads live as three **module-level tensors**
allocated once at import time on ``DEVICE`` with shapes matched to
the full NEST × SPU topology::

    _L2_GLOBAL : (NEST_NUM,            GTX_L2_SIZE_BYTES)   uint8
    _L1_GLOBAL : (NEST_NUM, SPU_NUM,   GTX_L1_SIZE_BYTES)   uint8
    _L0_GLOBAL : (NEST_NUM, SPU_NUM,   GTX_L0_SIZE_BYTES)   uint8

``GtxMemory`` holds direct references to those tensors and exposes
per-(NEST, SPU) byte / fp16 views through ``lk_byte`` / ``lk_f16``
helpers. The single contiguous backing per level keeps DMA-style
cross-SPU operations to plain tensor slicing
(``mem.l1[nest, :, off:off+n]``).

``DDR_MEMORY`` keeps its own grow-on-demand path on CPU — DDR is the
RISC-V system DRAM, not a scratchpad, and the CPU residence keeps
host↔device traffic confined to the DMA boundary.
"""


# =============================================================================
# Module-level scratchpad tensors — one contiguous block per level, sized
# to the full NEST × SPU topology so per-(NEST, SPU) buffers are views.
# =============================================================================

_L2_GLOBAL: torch.Tensor = torch.zeros(
    (GTX_NEST_NUM, GTX_L2_SIZE_BYTES),
    dtype=torch.uint8, device=DEVICE,
)
_L1_GLOBAL: torch.Tensor = torch.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES),
    dtype=torch.uint8, device=DEVICE,
)
_L0_GLOBAL: torch.Tensor = torch.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES),
    dtype=torch.uint8, device=DEVICE,
)


# =============================================================================
# Abstract base
# =============================================================================

class MEMORY(abc.ABC):
    """Common interface for every memory in the GTX hierarchy."""

    @abc.abstractmethod
    def getsize(self) -> int: ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Zero out the memory (does NOT release the backing tensor)."""


# =============================================================================
# DDR — grow-on-demand backing store
# =============================================================================

_DDR_DEVICE = torch.device("cpu")


class DDR_MEMORY(MEMORY):
    """Doubling-grow DDR — RISC-V *system* DRAM, always on CPU.

    Diverges from C++ single-shot 4 GiB alloc as a CI ergonomic (see
    03-RESEARCH 'ensure_ddr semantics divergence').

    Device contract (per user-stated hierarchy intent):
      - DDR lives on **CPU** (this is system memory in the RISC-V model).
      - L0 / L1 / L2 scratchpads live on **DEVICE** (CUDA when available).
      - Crossing the boundary happens inside ``read``/``write``: a caller
        passing a CUDA slice to ``write`` triggers a single ``.cpu()``;
        the slice returned by ``read`` stays on CPU and is the caller's
        responsibility to ``.to(scratchpad.device)`` if the destination
        is a scratchpad. This puts H/D moves at the DMA boundary only,
        not scattered across every byte access.
    """

    def __init__(self, size: int = DEFAULT_DDR_SIZE) -> None:
        self._bytes: Optional[torch.Tensor] = torch.zeros(
            size, dtype=torch.uint8, device=_DDR_DEVICE,
        )

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

    # --- MEMORY abc ---------------------------------------------------------
    def getsize(self) -> int:
        return len(self._bytes) if self._bytes is not None else 0

    def clear(self) -> None:
        if self._bytes is not None:
            self._bytes.zero_()

    # --- DDR-specific lifecycle --------------------------------------------
    def free(self) -> None:
        """Release the backing tensor (distinct from ``clear`` which zeros)."""
        self._bytes = None

    def capacity(self) -> int:
        return self._bytes.numel() if self._bytes is not None else 0

    def ensure(self, end_offset: int) -> Optional[torch.Tensor]:
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
            new_arr = torch.zeros(new_size, dtype=torch.uint8, device=_DDR_DEVICE)
            if self._bytes is not None:
                new_arr[:current_size] = self._bytes
            self._bytes = new_arr
        return self._bytes

    # --- byte-level access --------------------------------------------------
    # Caller convention: ``read`` returns a CPU slice; if you need it on
    # the scratchpad device, call ``.to(<scratchpad>.device)`` once at the
    # DMA boundary. ``write`` accepts either CPU or DEVICE input — a single
    # ``.cpu()`` is applied on cross-device input.

    def read(self, addr: int, n: int) -> torch.Tensor:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        return self._bytes[addr : addr + n]

    def write(self, addr: int, data: torch.Tensor) -> None:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        if data.device != self._bytes.device:
            data = data.to(self._bytes.device)
        self._bytes[addr : addr + data.numel()] = data

    def view(self, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        return self._bytes if dtype == torch.uint8 else self._bytes.view(dtype)

    def raw(self) -> Optional[torch.Tensor]:
        return self._bytes


# =============================================================================
# Top-level facade — references the module-level scratchpad tensors and
# owns DDR + the legacy SPR dict.
# =============================================================================

class GtxMemory(MEMORY):
    """GTX NPU memory layer — L0/L1/L2/DDR hierarchy facade."""

    def __init__(self) -> None:
        # Scratchpads share the module-level globals (single contiguous
        # tensor per level — see top of file).
        self.l2: torch.Tensor = _L2_GLOBAL
        self.l1: torch.Tensor = _L1_GLOBAL
        self.l0: torch.Tensor = _L0_GLOBAL
        self.ddr = DDR_MEMORY()
        self.spr: dict[int, int] = {}
        # Pre-built per-(NEST, SPU) views — every ``l[012]_byte`` /
        # ``l[012]_f16`` lookup pulled ~1.5M live slices on the abs
        # regression alone (cProfile: ~13s in ``l1_byte``). Caching the
        # slices once at construction collapses the hot path to a tensor
        # index, and the f16 view caches reuse the same storage so the
        # ``.view(torch.float16)`` cost is paid once.

        # Storage must stay aliased to ``self.l[012]``: DMA writes through
        # the byte view, vec ops write through the fp16 view, and
        # ``clear()`` / ``reset_scratchpads()`` zero through ``self.l[012]``
        # — all three paths need to land on the same backing bytes.
        # ``torch.stack`` (copies the inputs) and ``.to(torch.float16)``
        # (dtype cast, not reinterpret) both BREAK aliasing, so use the
        # original tensors directly and ``.view(torch.float16)`` for the
        # FP16 reinterpret (zero-copy, shared storage).
        self._l0_views: torch.Tensor = self.l0
        self._l1_views: torch.Tensor = self.l1
        self._l2_views: torch.Tensor = self.l2
        self._l0_f16_views: torch.Tensor = self.l0.view(torch.float16)
        self._l1_f16_views: torch.Tensor = self.l1.view(torch.float16)
        self._l2_f16_views: torch.Tensor = self.l2.view(torch.float16)

    def getsize(self) -> int:
        return (self.l0.numel() + self.l1.numel() + self.l2.numel()
                + self.ddr.getsize())

    def clear(self) -> None:
        """Zero everything (scratchpads + DDR). Keeps backing tensors."""
        self.l0.zero_()
        self.l1.zero_()
        self.l2.zero_()
        self.ddr.clear()
        self.spr.clear()

    def reset_scratchpads(self) -> None:
        """Zero scratchpads only — DDR is preserved (loaded firmware data
        must survive a hart reset). See ``GtxNpu.reset``."""
        self.l0.zero_()
        self.l1.zero_()
        self.l2.zero_()

    def free(self) -> None:
        """Release backing tensors. Currently only DDR has a release path."""
        self.l0.zero_()
        self.l1.zero_()
        self.l2.zero_()
        self.ddr.free()
        self.spr.clear()

    # ----- Raw byte views (D-10 low-level, kept for in-place strided ops) ---

    def l0_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self._l0_views[nest, spu]

    def l1_byte(self, nest: int, spu: int) -> torch.Tensor:
        return self._l1_views[nest, spu]

    def l2_byte(self, nest: int) -> torch.Tensor:
        return self._l2_views[nest]

    # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----

    def l0_f16(self, nest: int, spu: int) -> torch.Tensor:
        return self._l0_f16_views[nest, spu]

    def l1_f16(self, nest: int, spu: int) -> torch.Tensor:
        return self._l1_f16_views[nest, spu]

    def l2_f16(self, nest: int) -> torch.Tensor:
        return self._l2_f16_views[nest]

    def ensure_ddr(self, end_offset: int) -> Optional[torch.Tensor]:
        return self.ddr.ensure(end_offset)

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
                # frombuffer is CPU-only -- transfer to the DDR tensor's device
                # so the slice assignment works on GPU backends.
                src = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                ddr_buf = self.ddr.raw()
                assert ddr_buf is not None  # ensure() should have allocated it

                if src.device != ddr_buf.device:
                    src = src.to(ddr_buf.device)
                self.ddr.write(offset, src)
                offset += nbytes

    def ddr_save_to_hex(self, filename: str, addr: int, size: int) -> None:
        """Dump ``size`` bytes from DDR at ``addr`` to a vendor hex file.

        Single CUDA→CPU snapshot + vectorised 32-byte chunking. Each chunk
        is reversed in-place and hex-encoded; the whole file is written in
        one ``f.write`` call. Out-of-range addresses are zero-padded.
        """
        off = self._ddr_offset(addr)
        ddr_src = self.ddr.raw()
        ddr_size = self.ddr.capacity()

        if ddr_src is not None and ddr_size > 0 and size > 0:
            start = max(off, 0)
            end = min(off + size, ddr_size)
            if start < end:
                region = bytes(ddr_src[start:end].detach().cpu().contiguous().numpy())
            else:
                region = b""
            pre = max(0 - off, 0)
            post = size - pre - len(region)
            data = b"\x00" * pre + region + b"\x00" * post
        else:
            data = b"\x00" * size

        # Pad up to a 32-byte boundary so every line is full-width.
        if size & 0x1F:
            data = data + b"\x00" * (32 - (size & 0x1F))
        nlines = (size + 31) // 32
        lines = [data[i * 32:(i + 1) * 32][::-1].hex() for i in range(nlines)]
        with open(filename, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")

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
