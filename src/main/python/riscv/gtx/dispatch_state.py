"""DISPATCH state — context-aware handler lookup.

Distinct from :mod:`dispatch` (which is the table BUILDER invoked at
``GtxNpu.__init__``). This module is the per-instruction DISPATCH state
function that resolves a handler via the pre-built tables.

Table layout the builders produce::

    custom0:  funct7 → context → {funct3-or-None: handler}   (3-level)
    custom1:  funct3 → context → handler                     (2-level)

For each layer keyed by :class:`~unit.context.NpuContext`, prefer the
current-context entry; on miss fall back to the universal ``None`` key
(where legacy ``@handler`` calls without ``context=`` land).
"""
from __future__ import annotations

from .fsm import NpuState


def state_dispatch(npu) -> NpuState:
    """Resolve handler in current NPU context.

    Writes ``ctx['handler']`` and ``ctx['mnemonic']``; returns
    :attr:`NpuState.EXECUTE`.
    """
    kind = npu._ctx["kind"]
    f7 = npu._ctx["funct7"]
    f3 = npu._ctx["funct3"]
    ctx_now = npu._context

    handler = None
    if kind == "custom0":
        sub_table = npu._custom0.get(f7)
        if sub_table is not None:
            # Middle-level context lookup: prefer current context, fall
            # back to universal (None).
            ctx_table = sub_table.get(ctx_now)
            if ctx_table is None:
                ctx_table = sub_table.get(None)
            if ctx_table is not None:
                # Inner-level: P2 back-compat — None key (no funct3
                # decomp) tried before funct3 key (mask_funct3=True).
                handler = ctx_table.get(None)
                if handler is None:
                    handler = ctx_table.get(f3)
    elif kind == "custom1":
        inner = npu._custom1.get(f3)
        if inner is not None:
            handler = inner.get(ctx_now) or inner.get(None)
    # else: custom2/3 are inherited NOPs; should never reach here.

    npu._ctx["handler"] = handler
    npu._ctx["mnemonic"] = (
        getattr(handler, "gtx_mnemonic", None) if handler is not None else None
    )
    return NpuState.EXECUTE
