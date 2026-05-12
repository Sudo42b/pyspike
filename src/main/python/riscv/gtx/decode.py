"""DECODE state — extract funct7 / funct3 from the RoCC R-type instruction.

Pure function. Reads ``npu._ctx["insn"]``, writes ``npu._ctx["funct7"]``
and ``npu._ctx["funct3"]``. Returns :attr:`NpuState.DISPATCH`.
"""
from __future__ import annotations

from .fsm import NpuState


def state_decode(npu) -> NpuState:
    """Extract dispatch keys from the current instruction.

      ``funct7 = insn.funct``
      ``funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2``

    Source: RoCC R-type instruction layout. ``funct3`` synthesis mirrors
    the pre-FSM dispatcher.
    """
    insn = npu._ctx["insn"]
    npu._ctx["funct7"] = insn.funct
    npu._ctx["funct3"] = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
    return NpuState.DISPATCH
