#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
"""DECODE state — extract funct7 / funct3 from the RoCC R-type instruction.

Pure function. Reads `npu._ctx["insn"]`, writes `npu._ctx["funct7"]` and
`npu._ctx["funct3"]`. Returns the next FSM state (DISPATCH).
"""
from __future__ import annotations


def state_decode(npu):
    """Extract dispatch keys from the current instruction.

      funct7 = insn.funct
      funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    Source: RoCC R-type instruction layout. funct3 synthesis mirrors the
    pre-FSM dispatcher.
    """
    # Lazy import — _NpuState lives in npu.py, which imports this module,
    # so a module-top import would be circular.
    from .npu import _NpuState
    insn = npu._ctx["insn"]
    npu._ctx["funct7"] = insn.funct
    npu._ctx["funct3"] = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
    return _NpuState.DISPATCH
