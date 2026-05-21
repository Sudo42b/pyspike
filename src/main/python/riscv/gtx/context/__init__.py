

from __future__ import annotations

import enum
from typing import Dict, FrozenSet, Optional, Tuple


from enum import Enum


from ..config_params import DDR_BASE, L0_SIZE_BYTES, NEST_NUM, SPU_NUM
from ..csr import GSPR
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


# ---------------------------------------------------------------------------
# Scalar source selection (r2_sel) — shared by the MX scalar/vector handlers.
# Port of SystemC ``SPU::rs_select`` / vendor gtx_npu_custom0.cc:320-333.
# ---------------------------------------------------------------------------

# r2_sel = source_sel[8:7] — picks where the scalar (rs2) operand comes from.
# Staged in OPERAND5 (op_sel) by opset(slot=1); persists until the next
# opset(1, ...) — it is not consumed per instruction.
SRC_SEL_ZERO = 0b10   # operand forced to 0
SRC_SEL_SVR  = 0b11   # read from the SPU's L0 SVR scratchpad
                      # 0b00 / 0b01 -> gpr: use the rs2 GPR value as-is


def operand3(npu, default: int = 0) -> int:
    """rs3 staging word — GSPR_GTX_OPERAND3 (set by ``opset(slot=0)``).

    The result SVR addr lives in bits [4:0]; callers mask with ``& 0x1F`` (or
    test ``<= 0x1F`` for the in-place sentinel, where an unstaged OPERAND3 means
    "write back to the input register").
    """
    return int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, default))


def op_sel(npu) -> int:
    """source_sel / r2_sel — the rs4 staging word (OPERAND5[15:0])."""
    return npu.gspr.get(GSPR['GSPR_GTX_OPERAND5'].address, 0) & 0xFFFF


def svr_word(npu, nest: int, spu: int, svr_addr: int, svr_sub_addr: int) -> int:
    """64-bit little-endian load from L0 SVR[svr_addr] sub-block svr_sub_addr.

    An SVR register is 32 B; each of its four sub-blocks is 8 B (== the SystemC
    ``temp = (b << 32) | a`` two-word read).
    """
    off = (svr_addr * 32 + svr_sub_addr * 8) % L0_SIZE_BYTES
    word = npu.mem.l0_byte(nest, spu)[off:off + 8]
    return int.from_bytes(bytes(word.tolist()), "little")


def rs_select(npu, nest: int, spu: int, gpr_value: int) -> int:
    """Resolve a source operand per r2_sel — port of SystemC ``SPU::rs_select``.

    ``op_sel(npu)`` is the source_sel word; bits [8:7] pick the source:

        0b10  zero -> 0
        0b11  svr  -> L0 SVR[svr_addr][svr_sub_addr]
                      svr_addr     = source_sel[6:2]
                      svr_sub_addr = source_sel[1:0]
        else  gpr  -> gpr_value unchanged
    """
    source_sel = op_sel(npu)
    src = (source_sel >> 7) & 0b11
    if src == SRC_SEL_ZERO:
        return 0
    if src == SRC_SEL_SVR:
        svr_addr     = (source_sel >> 2) & 0x1F
        svr_sub_addr = source_sel & 0x3
        return svr_word(npu, nest, spu, svr_addr, svr_sub_addr)
    return gpr_value
