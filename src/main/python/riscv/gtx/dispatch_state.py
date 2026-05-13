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

from typing import TYPE_CHECKING

from .fsm import NpuState

if TYPE_CHECKING:
    # Same cycle as gtx.dispatch — gtx.npu instantiates from here, so the
    # runtime symbol can't be imported at module load.
    from .npu import GtxNpu


def state_dispatch(npu: GtxNpu) -> NpuState:
    """Resolve handler in current NPU context.

    Walks the pre-flattened ``npu._custom0_resolved`` / ``_custom1_resolved``
    tables (see :func:`gtx.dispatch.resolve_for_context`) — one
    ``dict.get`` per kind on the hot path, no per-instruction
    :class:`NpuContext` enum hashing. The flattened tables are kept in
    sync with ``npu._context`` by :mod:`gtx.writeback` on every warp-
    marker transition.

    Writes ``ctx['handler']`` and ``ctx['mnemonic']``; returns
    :attr:`NpuState.EXECUTE`.
    """
    kind = npu._ctx["kind"]
    f3 = npu._ctx["funct3"]

    handler = None
    if kind == "custom0":
        inner = npu._custom0_resolved.get(npu._ctx["funct7"])
        if inner is not None:
            # P2 back-compat — None key (no funct3 decomp) tried before
            # funct3 key (mask_funct3=True).
            handler = inner.get(None)
            if handler is None:
                handler = inner.get(f3)
    elif kind == "custom1":
        handler = npu._custom1_resolved.get(f3)
    # else: custom2/3 are inherited NOPs; should never reach here.

    npu._ctx["handler"] = handler
    npu._ctx["mnemonic"] = (
        getattr(handler, "gtx_mnemonic", None) if handler is not None else None
    )
    return NpuState.EXECUTE
