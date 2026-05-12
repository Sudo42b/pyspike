"""One-instruction FSM for ``GtxNpu.custom0`` / ``custom1`` dispatch.

Pipeline (per pyspike functional model — not cycle-accurate):

    IDLE → DECODE → DISPATCH → EXECUTE → WRITEBACK → IDLE

The FSM exists for structural clarity (state-by-state debugging,
future cycle-accurate hook insertion); behaviour is identical to the
pre-FSM 2-level dispatch.

Files
    fsm.py            — this module: enum + transition table + driver
    decode.py         — DECODE
    dispatch_state.py — DISPATCH
    execute.py        — EXECUTE
    writeback.py      — WRITEBACK
    idle.py           — IDLE (sentinel only)

``NpuContext`` (C1/C2/C3/C4) is a *persistent* execution context that
spans multiple instructions and is handled by DISPATCH and WRITEBACK —
do not confuse it with the per-instruction ``NpuState`` declared here.
"""
from __future__ import annotations

import enum
from typing import Callable, Dict


class NpuState(enum.Enum):
    """One-instruction FSM states for ``GtxNpu`` dispatch.

    Transition order is fixed: ``IDLE → DECODE → DISPATCH → EXECUTE →
    WRITEBACK → IDLE``. ``IDLE`` is the loop-exit sentinel and is not
    a key in :data:`STATE_TRANSITIONS`.
    """

    IDLE = enum.auto()
    DECODE = enum.auto()
    DISPATCH = enum.auto()
    EXECUTE = enum.auto()
    WRITEBACK = enum.auto()


# State-transition table: NpuState → callable taking the GtxNpu instance
# and returning the next NpuState. Built lazily to avoid import-time
# cycles between this module and the per-state modules.

_TRANSITIONS: Dict[NpuState, Callable] = {}


def _build_transitions() -> Dict[NpuState, Callable]:
    """Wire the per-state functions into a single dispatch table."""
    from .decode import state_decode
    from .dispatch_state import state_dispatch
    from .execute import state_execute
    from .writeback import state_writeback

    return {
        NpuState.DECODE:    state_decode,
        NpuState.DISPATCH:  state_dispatch,
        NpuState.EXECUTE:   state_execute,
        NpuState.WRITEBACK: state_writeback,
    }


def get_transitions() -> Dict[NpuState, Callable]:
    """Return the (lazy-built) state-transition table.

    The table is cached after the first call; per-state modules are
    imported on demand so ``fsm.py`` itself stays cycle-free.
    """
    if not _TRANSITIONS:
        _TRANSITIONS.update(_build_transitions())
    return _TRANSITIONS


def step(npu) -> None:
    """Advance the FSM by one state transition.

    Looks up the current state's transition function in
    :data:`STATE_TRANSITIONS` and applies its return value to
    ``npu._state``. ``IDLE`` is not a valid current state inside step —
    :func:`run_pipeline` exits its loop before reaching IDLE here.
    """
    table = get_transitions()
    fn = table.get(npu._state)
    if fn is None:
        raise RuntimeError(f"GtxNpu FSM: unreachable state {npu._state!r}")
    npu._state = fn(npu)


def run_pipeline(npu, kind: str, proc, insn, xs1, xs2) -> int:
    """Single-instruction FSM driver.

    Seeds ``npu._ctx`` with the per-instruction tuple, drives the FSM
    until it returns to :attr:`NpuState.IDLE`, and returns ``ctx['rd']``
    (the handler's return value, or 0 on dispatch miss).
    """
    npu._ctx = {
        "kind": kind,
        "proc": proc,
        "insn": insn,
        "xs1": xs1,
        "xs2": xs2,
        "rd": 0,
    }
    npu._state = NpuState.DECODE
    while npu._state is not NpuState.IDLE:
        step(npu)
    return npu._ctx["rd"]
