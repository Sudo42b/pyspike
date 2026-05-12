#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
"""EXECUTE state — invoke the resolved handler.

Miss (handler=None — funct7/context/funct3 unmapped) leaves rd=0
(silent NOP), matching pre-FSM dispatch semantics and vendor parity.
"""
from __future__ import annotations


def state_execute(npu):
    """Call ctx['handler'](proc, insn, xs1, xs2); store result in ctx['rd']."""
    from .npu import _NpuState
    handler = npu._ctx["handler"]
    if handler is not None:
        npu._ctx["rd"] = handler(
            npu._ctx["proc"],
            npu._ctx["insn"],
            npu._ctx["xs1"],
            npu._ctx["xs2"],
        )
    return _NpuState.WRITEBACK
