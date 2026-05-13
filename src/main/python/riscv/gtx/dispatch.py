from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional
from . import _registry
from .unit.context import NpuContext

# ``dispatch`` and ``npu`` form a circular import (``npu.py`` calls
# ``build_custom0_table`` at construction), so the ``GtxNpu`` reference
# must live behind ``TYPE_CHECKING`` — the runtime symbol comes through
# closure capture in ``_bind``, not the type annotation. ``from __future__
# import annotations`` keeps the per-function ``npu: GtxNpu`` hints as
# unresolved strings so they don't blow up at module load.
if TYPE_CHECKING:
    from .npu import GtxNpu

def resolve_for_context(custom0_3level: Dict[int, Dict],
                         custom1_2level: Dict[int, Dict],
                         context: NpuContext):
    """Pre-flatten the 3-level / 2-level dispatch tables for one context.

    The runtime dispatcher used to do per-instruction
    ``sub_table.get(current_context)`` + universal fallback +
    funct3 fallback — four ``dict.get`` calls hashing an
    :class:`NpuContext` enum every cycle (~1.5 s on ABS over 9.9 M
    hashes). Flattening once per context change collapses dispatch to
    a single outer lookup + at most one funct3 fallback, no enum hash
    on the hot path.

    Semantics match :mod:`dispatch_state` exactly: for each
    ``funct7`` (custom0) or ``funct3`` (custom1) entry, prefer the
    context-specific inner table; fall back to the universal
    (``None`` key) table only when no context-specific override
    exists. *Within* a chosen inner table the original
    ``ctx_table.get(None)`` / ``ctx_table.get(f3)`` priority is
    preserved because both keys live side by side after flattening.

    Returns
    -------
    (resolved_custom0, resolved_custom1)
        Two 2-level dicts:
          custom0: ``Dict[funct7, Dict[Optional[funct3], handler]]``
          custom1: ``Dict[funct3, handler]``
    """
    resolved_c0: Dict[int, Dict[Optional[int], Callable]] = {}
    for f7, ctx_table in custom0_3level.items():
        inner = ctx_table.get(context)
        if inner is None:
            inner = ctx_table.get(None)
        if inner is not None:
            resolved_c0[f7] = inner

    resolved_c1: Dict[int, Callable] = {}
    for f3, ctx_table in custom1_2level.items():
        h = ctx_table.get(context)
        if h is None:
            h = ctx_table.get(None)
        if h is not None:
            resolved_c1[f3] = h
    return resolved_c0, resolved_c1


def build_custom0_table(npu: GtxNpu) -> Dict[int, Dict]:
    """Build funct7 → context → {funct3-or-None: bound-handler} 3-level dict.

    Closure-binds npu so handlers can read npu.warp / npu.gspr / npu.mem.

    Levels:
      L1 (outer)  funct7 (int)
      L2 (middle) NpuContext or None
                  None = universal (handler valid in any context, matches
                  legacy @handler calls without context=).
                  NpuContext.Cx = per-context override.
      L3 (inner)  funct3 (int) when mask_funct3=True, else None (P2 back-compat
                  sentinel — dispatcher tries None first, then funct3).
    """
    raw = _registry.collect_for_kind("custom0")
    return {
        f7: {
            ctx_key: {f3: _bind(fn, npu) for f3, fn in inner.items()}
            for ctx_key, inner in ctx_table.items()
        }
        for f7, ctx_table in raw.items()
    }


def build_custom1_table(npu: GtxNpu) -> Dict[int, Dict]:
    """Build funct3 → context → bound-handler 2-level dict.

    Levels:
      L1 (outer)  funct3 (int)
      L2 (inner)  NpuContext or None (universal).
    """
    raw = _registry.collect_for_kind("custom1")
    return {
        f3: {ctx_key: _bind(fn, npu) for ctx_key, fn in inner.items()}
        for f3, inner in raw.items()
    }


def _bind(fn: Callable, npu: GtxNpu) -> Callable:
    def wrapped(proc, insn, xs1, xs2):
        return fn(npu, proc, insn, xs1, xs2)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    # Propagate mnemonic (set by _registry.handler decorator on fn) so
    # npu._state_dispatch can extract it for warp-marker detection.
    wrapped.gtx_mnemonic = getattr(fn, "gtx_mnemonic", None)  # type: ignore[attr-defined]
    return wrapped


# NOTE: the legacy `dispatch_4mode` router has been retired — its warp-state
# broadcast logic is subsumed by the FSM's DISPATCH stage plus the per-handler
# warp routing helpers in unit/context/{control,dma}.py.
