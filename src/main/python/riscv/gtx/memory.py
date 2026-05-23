from __future__ import annotations
import abc
import os
from typing import Any, Optional

import numpy as np

from .config_params import DDR_BASE, DEFAULT_DDR_SIZE, INITIAL_FLOOR

from .config_params import (
    L0_SIZE_BYTES,
    L1_SIZE_BYTES,
    L2_SIZE_BYTES,
    NEST_NUM,
    SPU_NUM,
    DEVICE,
    MX_IO_DTYPE,
)
from .context.exec_st import CXT


# DDR 32-byte bus-word reversal applied at the hex<->DDR boundary. Two corpora
# disagree on the convention, so the mode is env-selectable:
#   'byte' (default): full byte reversal (vendor "rightmost byte -> mem[0]"; the
#                     test/ corpus, big-endian hex).
#   'elem'          : reverse the fp16 element order only, keeping each element's
#                     two bytes (the GTX_ISS / ggml_ops_c corpus — little-endian
#                     fp16, no per-element byteswap; proven by ARANGE/abs goldens).
_DDR_REVERSE_MODE = os.environ.get("GTX_DDR_REVERSE_MODE", "byte").lower()


def _reverse_bus_word(chunk: bytes) -> bytes:
    """Reverse one (<=32-byte) bus word per GTX_DDR_REVERSE_MODE (see above)."""
    if _DDR_REVERSE_MODE == "elem" and len(chunk) % 2 == 0:
        return b"".join(chunk[i:i + 2] for i in range(len(chunk) - 2, -2, -2))
    return chunk[::-1]

"""GTX memory hierarchy.

L0 / L1 / L2 scratchpads live as three **module-level tensors**
allocated once at import time on ``DEVICE`` with shapes matched to
the full NEST × SPU topology::

    _L2_GLOBAL : (NEST_NUM,            L2_SIZE_BYTES)   uint8
    _L1_GLOBAL : (NEST_NUM, SPU_NUM,   L1_SIZE_BYTES)   uint8
    _L0_GLOBAL : (NEST_NUM, SPU_NUM,   L0_SIZE_BYTES)   uint8

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

_L2_GLOBAL: np.ndarray = np.zeros(
    (NEST_NUM, L2_SIZE_BYTES),
    dtype=np.uint8,
)
_L1_GLOBAL: np.ndarray = np.zeros(
    (NEST_NUM, SPU_NUM, L1_SIZE_BYTES),
    dtype=np.uint8,
)
_L0_GLOBAL: np.ndarray = np.zeros(
    (NEST_NUM, SPU_NUM, L0_SIZE_BYTES),
    dtype=np.uint8,
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

_DDR_DEVICE = "cpu"


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
        self._bytes: Optional[np.ndarray] = np.zeros(
            size, dtype=np.uint8,
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
            self._bytes.fill(0)

    # --- DDR-specific lifecycle --------------------------------------------
    def free(self) -> None:
        """Release the backing tensor (distinct from ``clear`` which zeros)."""
        self._bytes = None

    def capacity(self) -> int:
        return self._bytes.size if self._bytes is not None else 0

    def ensure(self, end_offset: int) -> Optional[np.ndarray]:
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
            new_arr = np.zeros(new_size, dtype=np.uint8,)
            if self._bytes is not None:
                new_arr[:current_size] = self._bytes
            self._bytes = new_arr
        return self._bytes

    # --- byte-level access --------------------------------------------------
    # Caller convention: ``read`` returns a CPU slice; if you need it on
    # the scratchpad device, call ``.to(<scratchpad>.device)`` once at the
    # DMA boundary. ``write`` accepts either CPU or DEVICE input — a single
    # ``.cpu()`` is applied on cross-device input.

    def read(self, addr: int, n: int) -> np.ndarray:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        return self._bytes[addr : addr + n]

    def write(self, addr: int, data: np.ndarray) -> None:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        self._bytes[addr : addr + data.size] = data

    def view(self, dtype: type = np.uint8) -> np.ndarray:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        return self._bytes if dtype == np.uint8 else self._bytes.view(dtype)

    def raw(self) -> Optional[np.ndarray]:
        return self._bytes


# Process-wide DDR — RISC-V system DRAM is shared by the CPU (reached via the
# gtx_ddr MMIO device, devices.py) and the NPU (GtxMemory). One Python-owned
# buffer: it outlives the C++ sim teardown, so the atexit DDR dump is safe.
_DDR_SINGLETON: Optional["DDR_MEMORY"] = None


def get_ddr() -> "DDR_MEMORY":
    global _DDR_SINGLETON
    if _DDR_SINGLETON is None:
        _DDR_SINGLETON = DDR_MEMORY()
    return _DDR_SINGLETON


# =============================================================================
# Top-level facade — references the module-level scratchpad tensors and
# owns DDR + the legacy SPR dict.
# =============================================================================

class GtxMemory(MEMORY):
    """GTX NPU memory layer — L0/L1/L2/DDR hierarchy facade."""

    def __init__(self) -> None:
        # Scratchpads share the module-level globals (single contiguous
        # tensor per level — see top of file).
        self.l2: np.ndarray = _L2_GLOBAL
        self.l1: np.ndarray = _L1_GLOBAL
        self.l0: np.ndarray = _L0_GLOBAL
        # DDR is process-wide system memory: one shared instance so the CPU
        # (via the gtx_ddr MMIO device) and the NPU see the same bytes.
        self.ddr = get_ddr()
        self.spr: dict[int, int] = {}
        # Pre-built per-(NEST, SPU) views — every ``l[012]_byte`` / ``l[012]_io``
        # lookup pulled ~1.5M live slices on the abs regression alone (cProfile:
        # ~13s in ``l1_byte``). Caching the slices once at construction collapses
        # the hot path to a tensor index; the io view cache reuses the same
        # storage so the ``.view(MX_IO_DTYPE)`` cost is paid once.
        #
        # Storage must stay aliased to ``self.l[012]``: DMA writes through the
        # byte view, MX ops write through the io view, and ``clear()`` /
        # ``reset_scratchpads()`` zero through ``self.l[012]`` — all paths must
        # land on the same backing bytes. ``.view(dtype)`` is a zero-copy
        # reinterpret (shared storage); ``.to(dtype)`` / ``torch.stack`` copy and
        # BREAK aliasing. (Arbitrary dtypes — e.g. FP16 GEMM operands — go
        # through ``view()``'s generic ``.view(dtype)`` branch.)
        self._l0_views: np.ndarray = self.l0
        self._l1_views: np.ndarray = self.l1
        self._l2_views: np.ndarray = self.l2
        # MX_IO_DTYPE reinterpret (zero-copy) — the numeric I/O width for MX ops
        # (FP32 default / FP16 toggle). Aliases the same bytes as the byte views.
        self._l0_io_views: np.ndarray = self.l0.view(MX_IO_DTYPE)
        self._l1_io_views: np.ndarray = self.l1.view(MX_IO_DTYPE)
        self._l2_io_views: np.ndarray = self.l2.view(MX_IO_DTYPE)

    def getsize(self) -> int:
        return (self.l0.size + self.l1.size + self.l2.size
                + self.ddr.getsize())

    def clear(self) -> None:
        """Zero everything (scratchpads + DDR). Keeps backing tensors."""
        self.l0.fill(0)
        self.l1.fill(0)
        self.l2.fill(0)
        self.ddr.clear()
        self.spr.clear()

    def reset_scratchpads(self) -> None:
        """Zero scratchpads only — DDR is preserved (loaded firmware data
        must survive a hart reset). See ``GtxNpu.reset``."""
        self.l0.fill(0)
        self.l1.fill(0)
        self.l2.fill(0)

    def free(self) -> None:
        """Release backing tensors. Currently only DDR has a release path."""
        self.l0.fill(0)
        self.l1.fill(0)
        self.l2.fill(0)
        self.ddr.free()
        self.spr.clear()

    # ----- Raw byte views (D-10 low-level, kept for in-place strided ops) ---

    def l0_byte(self, nest: int, spu: int) -> np.ndarray:
        return self._l0_views[nest, spu]

    def l1_byte(self, nest: int, spu: int) -> np.ndarray:
        return self._l1_views[nest, spu]

    def l2_byte(self, nest: int) -> np.ndarray:
        return self._l2_views[nest]

    # ----- MX_IO_DTYPE views (numeric I/O width; FP32 default, FP16 toggle) ---

    def l0_io(self, nest: int, spu: int) -> np.ndarray:
        return self._l0_io_views[nest, spu]

    def l1_io(self, nest: int, spu: int) -> np.ndarray:
        return self._l1_io_views[nest, spu]

    def l2_io(self, nest: int) -> np.ndarray:
        return self._l2_io_views[nest]

    def ensure_ddr(self, end_offset: int) -> Optional[np.ndarray]:
        return self.ddr.ensure(end_offset)

    def _ddr_offset(self, addr: int) -> int:
        """Address-to-offset helper. Direct port of C++ ddr_offset()."""
        if addr >= DDR_BASE:
            return addr - DDR_BASE
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
                chunk = _reverse_bus_word(chunk)
                self.ensure_ddr(offset + nbytes)
                # np.frombuffer requires writable buffer; bytes is read-only,
                # so go via bytearray (copy is cheap for 32-byte chunks).
                # frombuffer is CPU-only -- transfer to the DDR tensor's device
                # so the slice assignment works on GPU backends.
                src = np.frombuffer(bytearray(chunk), dtype=np.uint8)
                ddr_buf = self.ddr.raw()
                assert ddr_buf is not None  # ensure() should have allocated it
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
                region = np.ascontiguousarray(ddr_src[start:end]).tobytes()
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
        lines = [_reverse_bus_word(data[i * 32:(i + 1) * 32]).hex()
                 for i in range(nlines)]
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
            GTX_DDR_DUMP_ADDR   starting DDR address  (default ``DDR_BASE``)
            GTX_DDR_DUMP_SIZE   number of bytes       (default current DDR size)
        """
        path = os.environ.get("GTX_DDR_DUMP")
        if not path:
            return False
        addr_str = os.environ.get("GTX_DDR_DUMP_ADDR", "")
        addr = self._parse_env_int(addr_str) if addr_str else DDR_BASE
        size_str = os.environ.get("GTX_DDR_DUMP_SIZE", "")
        size = self._parse_env_int(size_str) if size_str else self.ddr.getsize()
        if size <= 0:
            return False
        self.ddr_save_to_hex(path, addr, size)
        return True

    # -------------------------------------------------------------------------
    # Context-aware scoped view resolver
    # -------------------------------------------------------------------------
    def view(self, cxt: CXT, level: str, ws: Any,
             dtype: type = np.uint8) -> np.ndarray:
        """Context-scoped view of a scratchpad ``level`` (``'l0'``/``'l1'``/``'l2'``).

        ``level`` picks WHICH memory; CONTEXT picks the (NEST, SPU) SCOPE so a
        single vectorized op broadcasts across the right instances:

            C3 → ``[nest, spu]``        (Mode 4: one SPU — hot path)
            C2 / C4 → ``[nest]``        (Mode 2/3: NEST, SPU axis kept → broadcast)
            C1 → full tensor            (Mode 1: NEST × SPU broadcast)

        ``dtype`` reinterprets the byte storage in place (zero-copy): ``float16``
        / ``float32`` views share the backing bytes, so writes land in L1/L0.
        L2 has no per-SPU axis — it is scoped per NEST (whole tensor in C1).
        """
        if dtype is np.uint8:
            base = {'l0': self._l0_views, 'l1': self._l1_views,
                    'l2': self._l2_views}[level]
        elif dtype is MX_IO_DTYPE:
            base = {'l0': self._l0_io_views, 'l1': self._l1_io_views,
                    'l2': self._l2_io_views}[level]
        else:                                   # any other dtype (e.g. FP16 GEMM
            base = {'l0': self._l0_views, 'l1': self._l1_views,   # operands) — a
                    'l2': self._l2_views}[level].view(dtype)      # zero-copy view

        if level == 'l2':                       # (NEST, BYTES) — no SPU axis
            return base if cxt is CXT.C1 else base[ws.current_nest]
        # l0 / l1 are (NEST, SPU, BYTES)
        if cxt is CXT.C3:
            return base[ws.current_nest, ws.current_spu]
        if cxt is CXT.C2 or cxt is CXT.C4:
            return base[ws.current_nest]        # (SPU, BYTES) — broadcast over SPU
        return base                             # C1 — (NEST, SPU, BYTES)