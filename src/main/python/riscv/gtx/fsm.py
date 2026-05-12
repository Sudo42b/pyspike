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
from typing import Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .npu import GtxNpu


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


def step(npu: 'GtxNpu') -> None:
    """Advance the FSM by one state transition.

    Kept for tests / future cycle-accurate hooks. :func:`run_pipeline`
    no longer goes through ``step`` on the hot path — see the inline
    fast path there.
    """
    table = get_transitions()
    fn = table.get(npu._state)
    if fn is None:
        raise RuntimeError(f"GtxNpu FSM: unreachable state {npu._state!r}")
    npu._state = fn(npu)


# Module-level cache of the four state functions. Resolved on first
# ``run_pipeline`` call (per-state modules import ``from .fsm import
# NpuState`` at module load, so fsm.py cannot import them at module load
# without a cycle). After resolution every subsequent instruction calls
# them through these globals — no dict lookup, no method indirection.
_STATE_DECODE: Optional[Callable] = None
_STATE_DISPATCH: Optional[Callable] = None
_STATE_EXECUTE: Optional[Callable] = None
_STATE_WRITEBACK: Optional[Callable] = None


def _ensure_state_fns() -> None:
    """Bind the four per-state callables to module globals.

    Called once on the first :func:`run_pipeline` invocation. By that
    point :mod:`decode` / :mod:`dispatch_state` / :mod:`execute` /
    :mod:`writeback` are already loaded (they were pulled in by
    ``riscv.gtx`` package init), so importing them here is just a
    namespace bind.
    """
    global _STATE_DECODE, _STATE_DISPATCH, _STATE_EXECUTE, _STATE_WRITEBACK
    from .decode import state_decode
    from .dispatch_state import state_dispatch
    from .execute import state_execute
    from .writeback import state_writeback
    _STATE_DECODE = state_decode
    _STATE_DISPATCH = state_dispatch
    _STATE_EXECUTE = state_execute
    _STATE_WRITEBACK = state_writeback


def run_pipeline(npu: 'GtxNpu', kind: str, proc, insn, xs1, xs2) -> int:
    """Single-instruction pipeline driver — inlined fast path.

    The four state functions execute strictly linearly
    (DECODE → DISPATCH → EXECUTE → WRITEBACK → IDLE) — there is no
    branching in the cycle, so the original ``while npu._state is not
    IDLE: step(npu)`` loop was burning ~5 s on ABS over the 1.98 M
    instructions × (table lookup + function pointer chase + state
    enum store) per iteration. Direct calls collapse it to four jumps
    with no per-cycle bookkeeping; per-state modules stay intact for
    testability and the legacy :func:`step` API stays exported.
    """
    if _STATE_DECODE is None:
        _ensure_state_fns()
    npu._ctx = {
        "kind": kind,
        "proc": proc,
        "insn": insn,
        "xs1": xs1,
        "xs2": xs2,
        "rd": 0,
    }
    _STATE_DECODE(npu)
    _STATE_DISPATCH(npu)
    _STATE_EXECUTE(npu)
    _STATE_WRITEBACK(npu)
    return npu._ctx["rd"]
