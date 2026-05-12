#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""@handler decorator + dispatch-table builders.

Extended with `context=` parameter (NpuContext | tuple | None) for Style C
per-context dispatch — see ORDER.md and npu_context.py.
"""
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .context import NpuContext

_HANDLER_REGISTRY: List[dict] = []

# Type alias: a @handler `context=` argument
ContextArg = Union[None, NpuContext, Iterable[NpuContext]]


def _normalize_context(context: ContextArg) -> Tuple[Optional[NpuContext], ...]:
    """Expand context= into a tuple of context keys to register under.

    Returns:
      (None,)                -- "all contexts" (default; current behavior)
      (NpuContext.Cx,)       -- single context
      (Cx, Cy, ...)          -- multi-context (registers under each)
    """
    if context is None:
        return (None,)
    if isinstance(context, NpuContext):
        return (context,)
    # Iterable of NpuContext
    result: List[NpuContext] = []
    for c in context:
        if not isinstance(c, NpuContext):
            raise TypeError(
                f"@handler context= must be NpuContext or iterable thereof; "
                f"got element {c!r}"
            )
        result.append(c)
    if not result:
        raise ValueError("@handler context= iterable is empty")
    return tuple(result)


def handler(*, kind: str, funct7: Optional[int] = None,
            funct3: Optional[int] = None, mnemonic: Optional[str] = None,
            mask_funct3: bool = False,
            context: ContextArg = None):
    """Register a handler function + (optional) disasm entry.

    kind: 'custom0' or 'custom1'
    funct7: required for custom0 dispatch key
    funct3: required for custom1 dispatch key (and custom0 mask_funct3 case)
    mnemonic: if provided, contributes a disasm_insn_t entry (built in plan 04)
    mask_funct3: custom0 with funct3 sub-variant (P4 MM only -- false in P2)
    context: NpuContext | Iterable[NpuContext] | None
        None (default) registers under the universal key — dispatcher uses
        this handler in any context where no context-specific override
        exists. Pass a single NpuContext value (or tuple thereof) to
        register a per-context handler — required for the ~12 instructions
        whose semantics differ across C1/C2/C3/C4 (see context_map.yaml
        notes; e.g. GTX_LOAD differs between C2 DDR→L2SPM and C3 L2SPM→L1SPM).
    """
    if kind not in ("custom0", "custom1"):
        raise ValueError(f"@handler kind must be 'custom0' or 'custom1', got {kind!r}")

    context_keys = _normalize_context(context)

    def decorator(fn: Callable) -> Callable:
        # Attach mnemonic on the function itself so dispatch.py:_bind can
        # propagate it onto the wrapped binding for npu._state_dispatch to
        # read during DISPATCH (used for warp-marker detection and trace).
        if mnemonic is not None:
            fn.gtx_mnemonic = mnemonic  # type: ignore[attr-defined]
        for ctx_key in context_keys:
            _HANDLER_REGISTRY.append({
                "fn": fn,
                "kind": kind,
                "funct7": funct7,
                "funct3": funct3,
                "mnemonic": mnemonic,
                "mask_funct3": mask_funct3,
                "context": ctx_key,  # None or NpuContext
            })
        return fn
    return decorator


def collect_for_kind(kind: str):
    """Build dispatch dict for a given kind, with context-aware inner layer.

    For 'custom0': returns 3-level dict
        Dict[funct7, Dict[ContextKey, Dict[Optional[funct3], Callable]]]
        where ContextKey = NpuContext or None ("all contexts" / universal).
        Outermost: funct7.
        Middle:    current NpuContext at dispatch time; None means universal
                   fallback (used when no context-specific override exists).
        Innermost: funct3 (when mask_funct3=True) or None (when False, P2
                   backwards-compat — dispatcher tries None first then funct3).
    For 'custom1': returns 2-level dict
        Dict[funct3, Dict[ContextKey, Callable]]
        where ContextKey = NpuContext or None.

    Dispatcher (npu._state_dispatch) tries current_context first, falls back
    to None on miss. Existing handlers registered without context= land in
    the None bucket — universal handlers, unchanged behavior.
    """
    if kind == "custom1":
        out_2level: Dict[int, Dict[Optional[NpuContext], Callable]] = {}
        for entry in _HANDLER_REGISTRY:
            if entry["kind"] != "custom1":
                continue
            f3 = entry["funct3"]
            if f3 is None:
                raise ValueError(
                    f"@handler custom1 missing funct3: mnemonic={entry['mnemonic']}"
                )
            ctx_key = entry["context"]
            sub = out_2level.setdefault(f3, {})
            if ctx_key in sub:
                raise ValueError(
                    f"duplicate custom1 handler: funct3={f3} context={ctx_key}"
                )
            sub[ctx_key] = entry["fn"]
        return out_2level

    if kind != "custom0":
        raise ValueError(f"unknown kind: {kind!r}")

    # 3-level for custom0: funct7 → context → {funct3-or-None: callable}
    out_3level: Dict[int, Dict[Optional[NpuContext], Dict]] = {}
    for entry in _HANDLER_REGISTRY:
        if entry["kind"] != "custom0":
            continue
        f7 = entry["funct7"]
        if f7 is None:
            raise ValueError(
                f"@handler custom0 missing funct7: mnemonic={entry['mnemonic']}"
            )
        ctx_key = entry["context"]
        inner_key: Optional[int] = (
            entry["funct3"] if entry.get("mask_funct3") else None
        )
        ctx_table = out_3level.setdefault(f7, {}).setdefault(ctx_key, {})
        if inner_key in ctx_table:
            raise ValueError(
                f"duplicate custom0 handler: funct7=0x{f7:02x} "
                f"context={ctx_key} funct3={inner_key}"
            )
        ctx_table[inner_key] = entry["fn"]
    return out_3level


def collect_disasms() -> list:
    """Build disasm_insn_t list from all entries with mnemonic != None.

    Walks _HANDLER_REGISTRY once, dispatching each mnemonic'd entry to the
    appropriate helper in disasm.py:
        - kind='custom0', mask_funct3=False -> add_r_custom0(mnemonic, funct7)
        - kind='custom0', mask_funct3=True  -> add_rf3_custom0(mnemonic, funct7, funct3)
        - kind='custom1'                    -> add_warp(mnemonic, funct3)

    Lazy-imports add_r_custom0 / add_rf3_custom0 / add_warp from .disasm to
    avoid load-order issues (disasm.py imports encoding.py; _registry.py is
    imported earlier in the dependency chain via dispatch.py).
    """
    from .disasm import add_r_custom0, add_rf3_custom0, add_warp

    out: list = []
    seen: set = set()  # dedupe by (kind, funct7, funct3, mnemonic) — context
                       # multi-registrations share one disasm entry.
    for entry in _HANDLER_REGISTRY:
        mnemonic = entry.get("mnemonic")
        if not mnemonic:
            continue
        kind = entry["kind"]
        key = (kind, entry.get("funct7"), entry.get("funct3"), mnemonic)
        if key in seen:
            continue
        seen.add(key)
        if kind == "custom0" and not entry.get("mask_funct3"):
            out.append(add_r_custom0(mnemonic, entry["funct7"]))
        elif kind == "custom0" and entry.get("mask_funct3"):
            out.append(add_rf3_custom0(mnemonic, entry["funct7"], entry["funct3"]))
        elif kind == "custom1":
            out.append(add_warp(mnemonic, entry["funct3"]))
        # else: silently skip (defensive -- handler() already validates kind)
    return out
