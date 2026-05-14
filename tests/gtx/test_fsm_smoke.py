"""FSM smoke -- ORDER.md state-machine wiring exists and is callable.

Per CONTEXT.md D-SMOKE-SPLIT: this test owns the *structural* signal
(NpuState enum members + 4 state functions exist + writeback returns IDLE).
Real dispatch correctness is the job of follow-up regression tasks.
"""
from __future__ import annotations

from riscv.gtx.fsm import NpuState


def test_npu_state_enum_has_five_members():
    """ORDER.md FSM: IDLE -> DECODE -> DISPATCH -> EXECUTE -> WRITEBACK."""
    expected = {"IDLE", "DECODE", "DISPATCH", "EXECUTE", "WRITEBACK"}
    assert {m.name for m in NpuState} == expected


def test_state_functions_are_importable_callables():
    """Each non-IDLE state has a transition function with signature
    ``state_xxx(npu) -> NpuState``."""
    from riscv.gtx.decode import state_decode
    from riscv.gtx.dispatch_state import state_dispatch
    from riscv.gtx.execute import state_execute
    from riscv.gtx.writeback import state_writeback

    for fn in (state_decode, state_dispatch, state_execute, state_writeback):
        assert callable(fn), f"{fn.__name__} is not callable"


def test_state_writeback_returns_idle(gtx_npu):
    """Post-R1 (writeback.py:18-19): writeback unconditionally returns IDLE."""
    from riscv.gtx.writeback import state_writeback

    next_state = state_writeback(gtx_npu)
    assert next_state is NpuState.IDLE
