"""custom0 entry smoke -- RoCC dispatch reaches GtxNpu and returns int.

Per CONTEXT.md D-SMOKE-SPLIT: drives the *integration* signal --
@isa.register('gtx') wiring, GtxNpu.__init__ + CUDA tensor allocation,
fast-path bypass (warp.is_tloop=False -> falls through to run_pipeline),
and run_pipeline returning rd as int.

Note: with funct=0 / xs1=xs2=xd=0 (default DummyInsn), run_pipeline
dispatches to an illegal/missing handler -- state_dispatch / state_execute
handle this without raising (handler resolution returns None -> fallback).
We only assert the RETURN TYPE (int), not the value. Real dispatch
correctness lives in regression tasks per CONTEXT.md.
"""
from __future__ import annotations


def test_custom0_returns_int(gtx_npu, mock_proc, dummy_insn):
    """GtxNpu.custom0(proc, insn, xs1, xs2) -> int."""
    result = gtx_npu.custom0(mock_proc, dummy_insn, 0, 0)
    assert isinstance(result, int), (
        f"custom0 must return int (RoCC reg_t), got {type(result).__name__}"
    )


def test_custom1_returns_int(gtx_npu, mock_proc, dummy_insn):
    """GtxNpu.custom1 also drives run_pipeline -- same contract."""
    result = gtx_npu.custom1(mock_proc, dummy_insn, 0, 0)
    assert isinstance(result, int)
