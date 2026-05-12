"""EXECUTE state — invoke the resolved handler.

Miss (``handler=None`` — funct7/context/funct3 unmapped) leaves
``rd=0`` (silent NOP), matching pre-FSM dispatch semantics and vendor
parity.
"""
from __future__ import annotations

from .fsm import NpuState


def state_execute(npu) -> NpuState:
    """Call ``ctx['handler'](proc, insn, xs1, xs2)``; store result in
    ``ctx['rd']``. Returns :attr:`NpuState.WRITEBACK`.
    """
    handler = npu._ctx["handler"]
    if handler is not None:
        npu._ctx["rd"] = handler(
            npu._ctx["proc"],
            npu._ctx["insn"],
            npu._ctx["xs1"],
            npu._ctx["xs2"],
        )
    return NpuState.WRITEBACK
