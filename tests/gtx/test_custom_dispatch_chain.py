"""custom0/custom1 dispatch chain coverage: DECODE→DISPATCH→EXECUTE→WRITEBACK.

Locks the post-d6f73f9 invariant that every @handler registers with
``context=None`` (universal) and resolves through the 3-level/2-level
build_*_table → resolve_for_context → state_dispatch → state_execute
chain. Existing 5-test smoke set guards the entry points; this suite
covers the in-between transitions one stage at a time.
"""
from __future__ import annotations

# Triggers @handler decorators across all op modules.
from riscv.gtx.unit.ins import ops as _ops  # noqa: F401
from riscv.gtx._registry import _HANDLER_REGISTRY, collect_for_kind
from riscv.gtx.unit.ins.encoding import (
    GTX_F7_WRSPR,
    GTX_ISS_F7_OPSET,
    WARP_F3_END_P,
)

from ._mocks import DummyInsn


def test_handler_registry_populated_after_ops_import():
    """Importing riscv.gtx.unit.ins.ops fires every @handler decorator."""
    assert 50 <= len(_HANDLER_REGISTRY) <= 200, (
        f"_HANDLER_REGISTRY size {len(_HANDLER_REGISTRY)} out of expected "
        f"range [50, 200] — ops import incomplete or registry leaked"
    )


def test_all_handlers_registered_with_universal_context():
    """d6f73f9 invariant: every @handler call uses context=None (universal)."""
    non_universal = [e for e in _HANDLER_REGISTRY if e["context"] is not None]
    assert non_universal == [], (
        f"found {len(non_universal)} per-context entries; first 3: "
        f"{non_universal[:3]}"
    )


def test_collect_for_kind_custom0_is_3level_dict():
    """collect_for_kind('custom0') → Dict[funct7, Dict[ctx, Dict[f3, fn]]]."""
    c0 = collect_for_kind("custom0")
    assert isinstance(c0, dict)
    assert len(c0) > 5
    assert GTX_F7_WRSPR in c0
    inner = c0[GTX_F7_WRSPR]
    assert isinstance(inner, dict)
    assert None in inner  # universal-context key
    inner_inner = inner[None]
    assert isinstance(inner_inner, dict)


def test_collect_for_kind_custom1_is_2level_dict():
    """collect_for_kind('custom1') → Dict[funct3, Dict[ctx, fn]]."""
    c1 = collect_for_kind("custom1")
    assert isinstance(c1, dict)
    assert len(c1) >= 4
    assert WARP_F3_END_P in c1
    assert None in c1[WARP_F3_END_P]


def test_build_custom0_table_binds_npu_and_propagates_mnemonic(gtx_npu):
    """_bind wraps fn(npu, ...) and carries .gtx_mnemonic onto the closure."""
    c0 = gtx_npu._custom0
    assert GTX_F7_WRSPR in c0
    inner = c0[GTX_F7_WRSPR][None]
    bound = next(iter(inner.values()))
    assert callable(bound)
    assert hasattr(bound, "gtx_mnemonic")
    c1_bound = next(iter(gtx_npu._custom1[WARP_F3_END_P].values()))
    assert hasattr(c1_bound, "gtx_mnemonic")


def test_resolve_for_context_flattens_to_per_context_table(gtx_npu):
    """refresh_dispatch_cache(INITIAL_CONTEXT) flattens 3-level/2-level tables."""
    r0 = gtx_npu._custom0_resolved
    assert GTX_F7_WRSPR in r0
    assert isinstance(r0[GTX_F7_WRSPR], dict)
    r1 = gtx_npu._custom1_resolved
    assert WARP_F3_END_P in r1
    assert callable(r1[WARP_F3_END_P])


def test_state_decode_extracts_funct7_funct3(gtx_npu, mock_proc):
    """state_decode reads insn into ctx['funct7'] / ctx['funct3'] = (xd<<2)|(xs1<<1)|xs2."""
    from riscv.gtx.decode import state_decode

    insn = DummyInsn(funct=0x4A, xd=1, xs1=0, xs2=1)
    gtx_npu._ctx = {
        "kind": "custom0", "proc": mock_proc, "insn": insn,
        "xs1": 0, "xs2": 0, "rd": 0,
    }
    next_state = state_decode(gtx_npu)
    assert gtx_npu._ctx["funct7"] == 0x4A
    assert gtx_npu._ctx["funct3"] == (1 << 2) | (0 << 1) | 1  # 0b101 = 5
    assert next_state.name == "DISPATCH"


def test_state_dispatch_resolves_handler_or_none(gtx_npu, mock_proc, dummy_insn):
    """state_dispatch: known funct7 → handler bound; unknown funct7 → None."""
    from riscv.gtx.dispatch_state import state_dispatch

    # Case A: known WRSPR funct7 resolves to a real handler.
    gtx_npu._ctx = {
        "kind": "custom0", "funct7": GTX_F7_WRSPR, "funct3": 0,
        "proc": mock_proc, "insn": dummy_insn, "xs1": 0, "xs2": 0, "rd": 0,
    }
    state_dispatch(gtx_npu)
    assert gtx_npu._ctx["handler"] is not None

    # Case B: unknown funct7 yields handler=None (silent NOP path).
    gtx_npu._ctx = {
        "kind": "custom0", "funct7": 0x7F, "funct3": 0,
        "proc": mock_proc, "insn": dummy_insn, "xs1": 0, "xs2": 0, "rd": 0,
    }
    state_dispatch(gtx_npu)
    assert gtx_npu._ctx["handler"] is None


def test_state_execute_handler_none_is_silent_nop_rd_zero(
    gtx_npu, mock_proc, dummy_insn,
):
    """Miss path: handler=None → rd untouched, next state = WRITEBACK.

    Default fixture has _tloop_buf=None so the T-loop buffering branch is
    bypassed (avoiding the npu.py:238/240/264/265 missing _GSPR_OP3/_GSPR_OP5
    NameError — out of scope, flagged in SUMMARY).
    """
    from riscv.gtx.execute import state_execute

    gtx_npu._ctx = {
        "handler": None, "rd": 0,
        "proc": mock_proc, "insn": dummy_insn, "xs1": 0, "xs2": 0,
        "mnemonic": None, "kind": "custom0",
    }
    next_state = state_execute(gtx_npu)
    assert next_state.name == "WRITEBACK"
    assert gtx_npu._ctx["rd"] == 0


def test_end_to_end_custom0_and_custom1_return_int(gtx_npu, mock_proc):
    """Full chain via GtxNpu.custom0/custom1 returns int (RoCC reg_t contract).

    custom0 funct=GTX_ISS_F7_OPSET routes through run_pipeline (not the
    T-loop fast-path; warp.is_tloop=False on fresh fixture). custom1 funct3
    = WARP_F3_END_P picks up the end_p handler bound during __init__.
    """
    insn0 = DummyInsn(funct=GTX_ISS_F7_OPSET, rs1=2, rs2=3, xs1=1, xs2=1)
    rc0 = gtx_npu.custom0(mock_proc, insn0, 0, 0)
    assert isinstance(rc0, int)

    # funct3 reconstructed as (xd<<2)|(xs1<<1)|xs2 = (1<<2)|(1<<1)|1 = 7 = END_P
    insn1 = DummyInsn(funct=0, xd=1, xs1=1, xs2=1)
    rc1 = gtx_npu.custom1(mock_proc, insn1, 0, 0)
    assert isinstance(rc1, int)
