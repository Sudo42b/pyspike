

from __future__ import annotations

import enum
from typing import Dict, FrozenSet, Optional, Tuple


from enum import Enum


from ..config_params import DDR_BASE, L0_SIZE_BYTES, NEST_NUM, SPU_NUM
from .exec_st import CXT


class WarpState:
    """Minimal warp/loop state — just the two routed IDs plus the WSPLIT sentinel.

    The loop *flags* are not stored: ``is_ploop`` / ``is_sloop`` / ``is_tloop``
    are derived from ``npu.CONTEXT`` (the single source of truth for nesting),
    so there is no boolean state to keep in sync.

        C1 ⇒ plan outside        → is_ploop=False
        C4 ⇒ plan inside (S/T off)→ is_ploop=True
        C2 ⇒ shared inside       → is_ploop, is_sloop
        C3 ⇒ thread inside       → is_ploop, is_tloop

    ``current_nest`` is set by ``start.p`` (NEST id); ``current_spu`` by
    ``start.t`` (SPU id) or ``start.s`` (GDMAC id). ``wsplit_seen`` is a
    process-lifetime sentinel set by WSPLIT and intentionally NOT cleared by
    ``reset`` (vendor ``gtx_npu.h`` field initializer).
    """

    def __init__(self, npu) -> None:
        self._npu = npu
        self.current_nest: int = 0
        self.current_spu: int = 0
        self.wsplit_seen: bool = False

    @property
    def is_ploop(self) -> bool:
        return self._npu.CONTEXT is not CXT.C1

    @property
    def is_sloop(self) -> bool:
        return self._npu.CONTEXT is CXT.C2

    @property
    def is_tloop(self) -> bool:
        return self._npu.CONTEXT is CXT.C3

    def reset(self) -> None:
        self.current_nest = 0
        self.current_spu = 0
        # wsplit_seen survives reset (process-lifetime sentinel).


def _resolve_nest_spu(npu) -> tuple[int, int]:
    """Representative (nest, spu) for the active context (C3 hot path).

    Broadcast scope (C1/C4 over multiple SPUs/NESTs) is handled by the
    memory resolver ``GtxMemory.view`` — this just returns the addressing
    anchor used to read uniform SPM addresses.
    """
    nest = npu.warp.current_nest if npu.warp.is_ploop else 0
    spu = npu.warp.current_spu if npu.warp.is_tloop else 0
    if nest >= NEST_NUM:
        nest = 0
    if spu >= SPU_NUM:
        spu = 0
    return nest, spu
