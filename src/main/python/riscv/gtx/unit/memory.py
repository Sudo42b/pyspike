from __future__ import annotations
import abc
import os
from typing import Optional

import numpy as _np

from ..config_params import xp, to_host, to_device

from ..config_params import GTX_DDR_BASE, DEFAULT_DDR_SIZE, INITIAL_FLOOR

from ..config_params import (
    GTX_L0_SIZE_BYTES,
    GTX_L1_SIZE_BYTES,
    GTX_L2_SIZE_BYTES,
    GTX_NEST_NUM,
    GTX_SPU_NUM,
)

"""GTX memory hierarchy.

L0 / L1 / L2 scratchpads live as three **module-level arrays**
allocated once at import time on the xp backend with shapes matched
to the full NEST × SPU topology::

    _L2_GLOBAL : (NEST_NUM,            GTX_L2_SIZE_BYTES)   uint8
    _L1_GLOBAL : (NEST_NUM, SPU_NUM,   GTX_L1_SIZE_BYTES)   uint8
    _L0_GLOBAL : (NEST_NUM, SPU_NUM,   GTX_L0_SIZE_BYTES)   uint8

``GtxMemory`` holds direct references to those arrays and exposes
per-(NEST, SPU) byte / fp16 views through ``lk_byte`` / ``lk_f16``
helpers. The single contiguous backing per level keeps DMA-style
cross-SPU operations to plain ndarray slicing
(``mem.l1[nest, :, off:off+n]``).

``DDR_MEMORY`` keeps its own grow-on-demand path. Under xp=numpy DDR
is host RAM; under xp=cupy (D-10) DDR is GPU VRAM and file I/O
crosses the H/D boundary via ``to_host()`` at the formatting edge.

WAVE-1-SHIM — Wave 1 → Wave 2/3 bridge (strangler-fig pattern):

    Storage layer is xp-internal (numpy default, cupy under
    GTX_USE_CUDA=1). Accessor methods that are STILL CONSUMED BY
    un-ported torch-API callers (tloop_buffer.py) wrap their returns
    in ``torch.from_numpy(...)`` via the private ``_torch_view(arr)``
    helper.

    Properties of the bridge:
      * Zero-copy on the numpy path (`torch.from_numpy` shares the
        underlying buffer with the source ndarray — writes through
        the torch tensor land in xp storage and vice versa).
      * Fail-loud on the cupy path: `torch.from_numpy` does not accept
        cupy buffers and Wave 6 cupy ports must already have removed
        the consuming torch calls by the time xp=cupy is used. The
        helper raises ``RuntimeError("Wave 2/3 cupy ports incomplete:
        ...")`` to surface the incomplete-port bug instantly instead
        of decoding a confusing torch internal AttributeError.

    Sunset condition: shim removed when ALL torch consumers are ported
    off torch (Wave 6 / plan 09-03-finalize end). The exact
    removal-wave assignment per accessor is documented in
    ``09-01b-SUMMARY.md`` "Deviations from plan: Option-B Wave 1
    bridge shim" table; each surviving call site below also carries
    a ``# WAVE-1-SHIM: remove in Wave <N>`` marker naming the plan
    that owns its removal.

    Removal log:
      * Wave 2a (plan 09-02a): l0_f16 / l1_f16 / l2_f16 shims removed
        (ops/*.py ported).
      * Wave 5 (plan 09-02b): l0_byte + ddr.read shims removed
        (dma_engine.py ported).
      * Wave 6 (plan 09-03-finalize) — pending: l1_byte + l2_byte
        shims + _torch_view helper + module-level torch import.
"""


# ---------------------------------------------------------------------------
# WAVE-1-SHIM helper — bridges xp.ndarray -> torch.Tensor at the accessor
# boundary so un-ported Wave 2/3 torch consumers keep working until they're
# ported. THROWAWAY. Removed in Wave 3 (plan 09-03-finalize) once every call
# site below has been ported off torch.
#
# This is the ONLY place memory.py imports torch (a deliberate local import
# inside the helper so the module-import path stays clean and the cupy
# path never touches torch).
# ---------------------------------------------------------------------------

def _torch_view(arr):
    """Zero-copy bridge from an xp ndarray to a torch.Tensor (numpy path).

    WAVE-1-SHIM (Option B, plan 09-01b Task 4). Sunset: Wave 3 end.

    Used by ``GtxMemory.{l0,l1,l2}_byte``, ``GtxMemory.{l0,l1,l2}_f16``, and
    ``DDR_MEMORY.read`` so that Wave 2/3 callers still expecting torch
    tensors (``.to(device)``, ``.view(torch.float16)``, ``.copy_(...)``,
    ``.numel()``, etc.) keep working unchanged until their files are ported
    off torch.

    On the numpy path: ``torch.from_numpy(arr)`` shares the same underlying
    buffer — no allocation, no ``.copy()``. Wave 1a's in-place DMA / vec /
    op semantics survive because the torch tensor and the xp ndarray
    reference the same memory.

    On the cupy path: torch.from_numpy does not accept cupy buffers, and
    by the time xp=cupy is in use the Wave 2/3 cupy ports must already
    have removed every torch-API call site. If we reach this branch it
    means a torch consumer was missed — raise loudly with a hint instead
    of letting torch decode a confusing AttributeError.
    """
    if xp is _np:
        # Local import — keep this isolated to the shim path so the cupy
        # branch never imports torch and so the helper is easy to grep +
        # delete when the shim sunsets.
        import torch  # noqa: PLC0415 (intentional local; shim sunset)
        return torch.from_numpy(arr)
    # cupy path: incomplete-port bug — surface explicitly.
    raise RuntimeError(
        "Wave 2/3 cupy ports incomplete: memory.py shim was hit on the "
        "xp=cupy path. Every torch-API caller (dma_engine.py / "
        "tloop_buffer.py / ops/*.py / _verify.py) must be ported to xp "
        "before GTX_USE_CUDA=1 is exercised. See 09-01b-SUMMARY.md "
        "'Deviations from plan' for the per-shim removal-wave table."
    )


# =============================================================================
# Module-level scratchpad arrays — one contiguous block per level, sized
# to the full NEST × SPU topology so per-(NEST, SPU) buffers are views.
# =============================================================================

_L2_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_L2_SIZE_BYTES),
    dtype=xp.uint8,
)
_L1_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES),
    dtype=xp.uint8,
)
_L0_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES),
    dtype=xp.uint8,
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
        """Zero out the memory (does NOT release the backing array)."""


# =============================================================================
# DDR — grow-on-demand backing store
# =============================================================================


class DDR_MEMORY(MEMORY):
    """Doubling-grow DDR — RISC-V *system* DRAM.

    Diverges from C++ single-shot 4 GiB alloc as a CI ergonomic (see
    03-RESEARCH 'ensure_ddr semantics divergence').

    Device contract (D-10 post-Phase-9 Wave 1a):
      - DDR lives on the same xp backend as scratchpads (numpy = host RAM,
        cupy = GPU VRAM). The legacy explicit-CPU placement constant was
        removed in Wave 1a; placement now follows xp.
      - File I/O (`ddr_load_from_hex` / `ddr_save_to_hex`) bridges through
        ``to_host()`` once, at the formatting edge — keeps H↔D traffic at
        the file boundary, not scattered across every byte access.
    """

    def __init__(self, size: int = DEFAULT_DDR_SIZE) -> None:
        # D-10: DDR follows xp. On consumer GPUs (<12 GB VRAM), set
        # `GTX_DDR_SIZE=1G` via env var to leave headroom for scratchpads
        # (~25 MB) + CUDA context overhead. See README "GPU memory budget"
        # section.
        self._bytes: Optional[xp.ndarray] = xp.zeros(size, dtype=xp.uint8)

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
        return int(self._bytes.shape[0]) if self._bytes is not None else 0

    def clear(self) -> None:
        if self._bytes is not None:
            self._bytes[:] = 0

    # --- DDR-specific lifecycle --------------------------------------------
    def free(self) -> None:
        """Release the backing array (distinct from ``clear`` which zeros)."""
        self._bytes = None

    def capacity(self) -> int:
        return int(self._bytes.size) if self._bytes is not None else 0

    def ensure(self, end_offset: int) -> Optional[xp.ndarray]:
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
            new_arr = xp.zeros(new_size, dtype=xp.uint8)
            if self._bytes is not None:
                # xp slice-assign works on both numpy and cupy.
                new_arr[:current_size] = self._bytes
            self._bytes = new_arr
        return self._bytes

    # --- byte-level access --------------------------------------------------
    # Caller convention: ``read`` returns a view on the xp backend; ``write``
    # accepts an ndarray on the same backend (caller is responsible for any
    # ``to_device``/``to_host`` conversion at DMA / file boundaries).

    def read(self, addr: int, n: int) -> xp.ndarray:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        # Wave 5 (plan 09-02b) removed the WAVE-1-SHIM at this accessor.
        # dma_engine.py was the only torch consumer of DDR_MEMORY.read; with
        # dma_engine ported, the accessor returns bare xp.ndarray.
        return self._bytes[addr : addr + n]

    def write(self, addr: int, data) -> None:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        n = int(data.size)
        self._bytes[addr : addr + n] = data

    def view(self, dtype=None) -> xp.ndarray:
        if self._bytes is None:
            raise RuntimeError("DDR not allocated — call ensure() first")
        if dtype is None or dtype == xp.uint8:
            return self._bytes
        return self._bytes.view(dtype)

    def raw(self) -> Optional[xp.ndarray]:
        return self._bytes


# =============================================================================
# Top-level facade — references the module-level scratchpad arrays and
# owns DDR + the legacy SPR dict.
# =============================================================================

class GtxMemory(MEMORY):
    """GTX NPU memory layer — L0/L1/L2/DDR hierarchy facade."""

    def __init__(self) -> None:
        # Scratchpads share the module-level globals (single contiguous
        # array per level — see top of file).
        self.l2 = _L2_GLOBAL
        self.l1 = _L1_GLOBAL
        self.l0 = _L0_GLOBAL
        self.ddr = DDR_MEMORY()
        self.spr: dict[int, int] = {}
        # Pre-built per-(NEST, SPU) views — every ``l[012]_byte`` /
        # ``l[012]_f16`` lookup pulled ~1.5M live slices on the abs
        # regression alone (cProfile: ~13s in ``l1_byte``). Caching the
        # slices once at construction collapses the hot path to an ndarray
        # index, and the f16 view caches reuse the same storage so the
        # ``.view(xp.float16)`` cost is paid once.

        # Storage must stay aliased to ``self.l[012]``: DMA writes through
        # the byte view, vec ops write through the fp16 view, and
        # ``clear()`` / ``reset_scratchpads()`` zero through ``self.l[012]``
        # — all three paths need to land on the same backing bytes.
        # ``xp.stack`` (copies the inputs) and dtype-casting (e.g.
        # ``.astype(xp.float16)``) both BREAK aliasing, so use the original
        # arrays directly and ``.view(xp.float16)`` for the FP16 reinterpret
        # (zero-copy, shared storage; LE byte order — see __init__.py
        # tripwire).
        self._l0_views = self.l0
        self._l1_views = self.l1
        self._l2_views = self.l2
        self._l0_f16_views = self.l0.view(xp.float16)
        self._l1_f16_views = self.l1.view(xp.float16)
        self._l2_f16_views = self.l2.view(xp.float16)

    def getsize(self) -> int:
        return (int(self.l0.size) + int(self.l1.size) + int(self.l2.size)
                + self.ddr.getsize())

    def clear(self) -> None:
        """Zero everything (scratchpads + DDR). Keeps backing arrays."""
        self.l0[:] = 0
        self.l1[:] = 0
        self.l2[:] = 0
        self.ddr.clear()
        self.spr.clear()

    def reset_scratchpads(self) -> None:
        """Zero scratchpads only — DDR is preserved (loaded firmware data
        must survive a hart reset). See ``GtxNpu.reset``."""
        self.l0[:] = 0
        self.l1[:] = 0
        self.l2[:] = 0

    def free(self) -> None:
        """Release backing arrays. Currently only DDR has a release path."""
        self.l0[:] = 0
        self.l1[:] = 0
        self.l2[:] = 0
        self.ddr.free()
        self.spr.clear()

    # ----- Raw byte views (D-10 low-level, kept for in-place strided ops) ---

    def l0_byte(self, nest: int, spu: int) -> xp.ndarray:
        # Wave 5 (plan 09-02b) removed the WAVE-1-SHIM at this accessor.
        # Wave 2a ported ops/{act,mm,spr,vec}.py to bypass via raw
        # `npu.mem.l0[nest, spu]`; dma_engine.py:155/179 (Wave 5) likewise
        # uses raw `mem.l0[nest, spu]`. No torch consumers remain.
        return self._l0_views[nest, spu]

    def l1_byte(self, nest: int, spu: int) -> "xp.ndarray | object":
        # WAVE-1-SHIM: remove in Wave 3 (port ops/act.py + ops/mm.py +
        # dma_engine.py:206/289 + tloop_buffer.py:483 — last torch consumer
        # is in tloop_buffer.py which is owned by Wave 3 plan 09-03-finalize).
        return _torch_view(self._l1_views[nest, spu])

    def l2_byte(self, nest: int) -> "xp.ndarray | object":
        # WAVE-1-SHIM: remove in Wave 3 (port tloop_buffer.py:459/467/477/485
        # — the only torch consumer of l2_byte; Wave 3 plan 09-03-finalize).
        return _torch_view(self._l2_views[nest])

    # ----- Halfword fp16 views (D-10 named, D-12 view guarantee) -----
    #
    # Wave 2a (plan 09-02a-ops) removed the WAVE-1-SHIM at these accessors.
    # All ops/* consumers (act.py L312/433, vec.py L124) now bypass these
    # accessors entirely and read raw xp byte storage via
    # `npu.mem.l[012][nest, spu].view(xp.float16)`. The accessors stay as
    # pure xp.ndarray returns for future external consumers.

    def l0_f16(self, nest: int, spu: int) -> "xp.ndarray":
        return self._l0_f16_views[nest, spu]

    def l1_f16(self, nest: int, spu: int) -> "xp.ndarray":
        return self._l1_f16_views[nest, spu]

    def l2_f16(self, nest: int) -> "xp.ndarray":
        return self._l2_f16_views[nest]

    def ensure_ddr(self, end_offset: int) -> Optional[xp.ndarray]:
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
                # frombuffer is a numpy-only path; on cupy we route the
                # host bytes through numpy → to_device. File I/O is the
                # host boundary by contract.
                src_host = _np.frombuffer(bytearray(chunk), dtype=_np.uint8)
                src = to_device(src_host)
                self.ddr.write(offset, src)
                offset += nbytes

    def ddr_save_to_hex(self, filename: str, addr: int, size: int) -> None:
        """Dump ``size`` bytes from DDR at ``addr`` to a vendor hex file.

        Single device→host snapshot + vectorised 32-byte chunking. Each chunk
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
                # to_host: no-op on numpy, cp.asnumpy on cupy. Slice first
                # so we only ship the needed window across the H/D boundary.
                region = bytes(to_host(ddr_src[start:end]))
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
