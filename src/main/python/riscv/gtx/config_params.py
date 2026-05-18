from __future__ import annotations
import os
import numpy as _np


def _identity(arr):
    """No-op for xp=numpy path (D-12)."""
    return arr


def _resolve_backend():
    """Resolve xp + to_host + to_device at import time (D-01, D-02).

    Default: numpy + identity helpers (no-op).
    GTX_USE_CUDA=1 (or "true"/"TRUE"): require cupy importable, else fail-loud
    with `pip install 'spike[cuda]'` hint (D-03). Silent fallback FORBIDDEN
    (260518-ffr regression precedent — torch.cuda.is_available auto-true
    flipped 5x ABS slowdown).
    """
    env = os.environ.get("GTX_USE_CUDA", "").strip()
    if env not in ("1", "true", "TRUE"):
        return _np, _identity, _identity

    try:
        import cupy as _cp
    except ImportError as exc:
        raise RuntimeError(
            "GTX_USE_CUDA=1 set but cupy is not importable. "
            "Install with: pip install 'spike[cuda]'"
        ) from exc

    return _cp, _cp.asnumpy, _cp.asarray


# Module-level eager resolution, frozen for process lifetime (D-02).
# All gtx.* modules: `from .config_params import xp, to_host, to_device`.
xp, to_host, to_device = _resolve_backend()

# Phase 9 Wave 6 D-04 clean-cut: DEVICE symbol REMOVED. The previous Wave 0
# deferred string alias (`DEVICE: str = "cpu" if xp is _np else "cuda"`) is
# gone. All downstream consumers ported in Waves 1/2/5 (memory.py,
# register_file.py, npu.py, dma_engine.py, ops/*.py). `from
# riscv.gtx.config_params import DEVICE` now raises ImportError —
# behavior verified by tests/gtx/test_xp_alias.py.


# NEST x SPU topology
GTX_NEST_NUM: int = 4
GTX_SPU_NUM: int = 16          # SPUs per NEST
GTX_SPUS_PER_NEST: int = GTX_SPU_NUM   # alias for clarity
# D-02 default: 4 GiB
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024
# D-13: floor for doubling-grow first allocation. 1 MiB picked because:
#   - covers 32-byte bus-word minimum with ample headroom
#   - small enough that CI per-test allocations are cheap
#   - large enough that "single grow per test" is the common case
INITIAL_FLOOR: int = 1 * 1024 * 1024

# Memory sizes (bytes)
GTX_L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
GTX_L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
GTX_L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

# DDR (D-02: capped by GTX_DDR_SIZE env var; default below)
GTX_DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

# DDR I/O (D-03)
GTX_DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for GTX_DDR_REVERSED reversal

# DDR base physical address (firmware GTX_MAIN_BASE -- gtx_params.h:24)
GTX_DDR_BASE: int = 0x370000000

# SPR address ranges — source of truth is `unit/csr/__init__.py:42-47`.
# Don't redefine the GSPR_BASE / NSPR_BASE / LSPR_BASE constants here;
# import them from `riscv.gtx.unit.csr` instead.
