"""GTX NPU hardware subpackage.

Modules
    memory.py         storage classes (SPU_MEMORY, L2_MEMORY, DDR_MEMORY,
                      GtxMemory) — backing tensors for L0/L1/L2/DDR.
    register_file.py  name-indexed wrapper over the live SPR int-dict;
                      uses typed defs from ``csr/``.
    csr/              typed @csr register declarations (GSPR / NSPR / LSPR).
    context/          NPU context FSM + warp state + DMA op handlers.
    ins/              instruction encoding, engines, ops, disasm.
"""
from . import memory
from .memory import (
    MEMORY,
    SPU_MEMORY,
    L2_MEMORY,
    DDR_MEMORY,
    GtxMemory,
)
from .register_file import RegisterFile
from .context.warp_state import WarpState
from .context import NpuContext, INITIAL_CONTEXT
from . import csr

__all__ = [
    "memory",
    "MEMORY",
    "SPU_MEMORY",
    "L2_MEMORY",
    "DDR_MEMORY",
    "GtxMemory",
    "RegisterFile",
    "csr",
    "WarpState",
    "NpuContext",
    "INITIAL_CONTEXT",
]
