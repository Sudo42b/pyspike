#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
"""WRITEBACK state — OPSET staging clear + NPU context transition.

(1) Vendor-parity OPSET staging clear (custom0 only):
    Every non-OPSET custom0 instruction clears `gspr[0x003]` (OPERAND3)
    and `gspr[0x005]` (OPERAND4) so stale staging values do not leak
    across unrelated instructions. Source: gtx_npu_custom0.cc:1042-1058.

(2) NPU context transition (any kind):
    If the just-executed instruction is a warp marker (START_P/S/T or
    END_P/S/T), apply the corresponding context change so the NEXT
    instruction sees the new context. SPLIT/JOIN are markers but cause
    no transition. Illegal transitions are silently ignored (lenient
    mode — see context.apply_transition).
"""
from __future__ import annotations


# OPSET funct7 (kept here rather than in ins/encoding.py to avoid an
# import cycle: writeback.py is on the FSM hot path and should not pull
# op-id constants from the instruction subpackage).
_OPSET_FUNCT7 = 0x4A
# GSPR addresses cleared after every non-OPSET dispatch.
_GSPR_OPERAND3 = 0x003
_GSPR_OPERAND4 = 0x005


def state_writeback(npu):
    """OPSET staging clear + warp-marker context transition."""
    from .npu import _NpuState
    from .context import is_warp_marker, apply_transition

    # (1) OPSET staging clear (vendor parity)
    if npu._ctx["kind"] == "custom0" and npu._ctx["funct7"] != _OPSET_FUNCT7:
        npu.gspr[_GSPR_OPERAND3] = 0
        npu.gspr[_GSPR_OPERAND4] = 0

    # (2) Context transition (warp markers only)
    mnemonic = npu._ctx.get("mnemonic")
    if mnemonic is not None and is_warp_marker(mnemonic):
        npu._context = apply_transition(npu._context, mnemonic)

    return _NpuState.IDLE
