"""WRITEBACK state — no-op transition back to IDLE.

Per ORDER.md FSM simplification (d6f73f9):

- OPSET staging clear is owned by the ``custom0`` fast-path in
  :mod:`gtx.npu` (zeroes ``GSPR_GTX_OPERAND3/5`` on every bufferable
  T-loop snapshot).
- NPU context transition is dropped: all 122 ``@handler`` registrations
  are universal (``context=None``), so :func:`dispatch.resolve_for_context`
  always falls through to the universal table — no per-instruction
  re-flatten is needed. ORDER.md's C1/C2/C3/C4 table is retained as
  documentation for future context-aware dispatch, but the runtime
  state machine treats every cycle as C1.
"""
from .fsm import NpuState


def state_writeback(npu) -> NpuState:
    return NpuState.IDLE
