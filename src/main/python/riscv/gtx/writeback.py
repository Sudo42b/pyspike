"""WRITEBACK state — OPSET staging clear + NPU context transition.

(1) Vendor-parity OPSET staging clear (custom0 only):
    Every non-OPSET custom0 instruction clears ``GSPR_GTX_OPERAND3``
    and ``GSPR_GTX_OPERAND5`` so stale staging values do not leak
    across unrelated instructions. Source: ``gtx_npu_custom0.cc:1042-1058``.

(2) NPU context transition (any kind):
    If the just-executed instruction is a warp marker (START_P/S/T or
    END_P/S/T), apply the corresponding context change so the NEXT
    instruction sees the new context. SPLIT/JOIN are markers but cause
    no transition. Illegal transitions are silently ignored (lenient
    mode — see :func:`unit.context.apply_transition`).
"""
from __future__ import annotations

from .fsm import NpuState
from .unit.context import apply_transition, is_warp_marker
from .unit.ins.encoding import (
    GSPR_GTX_OPERAND3,
    GSPR_GTX_OPERAND5,
    GTX_ISS_F7_OPSET,
)


def state_writeback(npu) -> NpuState:
    """OPSET staging clear + warp-marker context transition."""
    # (1) OPSET staging clear (vendor parity)
    if (npu._ctx["kind"] == "custom0"
            and npu._ctx["funct7"] != GTX_ISS_F7_OPSET):
        npu.gspr[GSPR_GTX_OPERAND3] = 0
        npu.gspr[GSPR_GTX_OPERAND5] = 0

    # (2) Context transition (warp markers only)
    mnemonic = npu._ctx.get("mnemonic")
    if mnemonic is not None and is_warp_marker(mnemonic):
        npu._context = apply_transition(npu._context, mnemonic)

    return NpuState.IDLE
