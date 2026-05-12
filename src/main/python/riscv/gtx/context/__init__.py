"""GTX context state definitions.

NPU context FSM — C1/C2/C3/C4 contexts, transitions, validity.

MANUALLY SYNCED from src/main/python/riscv/context_map.yaml on 2026-05-11.
When the YAML changes, update this file. (No runtime PyYAML dependency
per CLAUDE.md "NumPy 외부 추가 런타임 의존성 신규 도입 금지".)

NPU contexts (persistent across instructions):
  C1: plan outside     — before START_P (initial / after END_P)
  C2: shared inside    — inside START_P + START_S
  C3: thread inside    — inside START_P + START_T
  C4: plan inside only — inside START_P, outside S/T

States:
	C1 (PLAN_OUTSIDE):
		Plan outside state. This is the state before ``START_P`` and the state
		restored after ``END_P``.

	C4 (PLAN_INSIDE):
		Plan-inside state with both shared and thread scopes inactive. This is
		the state inside ``START_P`` and outside ``START_S`` and ``START_T``.

	C2 (SHARED):
		Plan-inside and shared-inside state. This is the state reached inside
		``START_P`` and ``START_S``.

	C3 (THREAD):
		Plan-inside and thread-inside state. This is the state reached inside
		``START_P`` and ``START_T``.

Public API:
  NpuContext                  — Enum (C1/C2/C3/C4)
  INITIAL_CONTEXT             — NpuContext.C1
  get_group(mnemonic)         — group name or None
  is_valid_in_context(mn, ctx)— bool
  is_warp_marker(mnemonic)    — bool
  apply_transition(ctx, mn)   — next context (or unchanged)
  is_legal_transition(ctx, mn)— strict check

"""

from __future__ import annotations

import enum
from typing import Dict, FrozenSet, Optional, Tuple


# ===========================================================================
# Context enum
# ===========================================================================

class NpuContext(enum.Enum):
    """GTX NPU execution context (persistent across instructions).

    Determined by surrounding warp markers (`GTX_WARP_START_P/S/T` and
    matching `END_*`). Each context restricts which instruction groups
    may execute (see _CONTEXT_VALID_GROUPS below).
    """
    C1 = "C1"  # plan outside — before START_P
    C2 = "C2"  # plan inside + shared inside (START_P + START_S)
    C3 = "C3"  # plan inside + thread inside (START_P + START_T)
    C4 = "C4"  # plan inside, shared/thread outside


INITIAL_CONTEXT: NpuContext = NpuContext.C1


# ===========================================================================
# Warp marker transitions
# ===========================================================================
# Mnemonic-keyed. SPLIT/JOIN are markers but cause no context change.
# Format: marker_mnemonic → (from_context, to_context)

WARP_TRANSITIONS: Dict[str, Tuple[NpuContext, NpuContext]] = {
    "GTX_WARP_START_P": (NpuContext.C1, NpuContext.C4),
    "GTX_WARP_END_P":   (NpuContext.C4, NpuContext.C1),
    "GTX_WARP_START_S": (NpuContext.C4, NpuContext.C2),
    "GTX_WARP_END_S":   (NpuContext.C2, NpuContext.C4),
    "GTX_WARP_START_T": (NpuContext.C4, NpuContext.C3),
    "GTX_WARP_END_T":   (NpuContext.C3, NpuContext.C4),
}

WARP_MARKERS_NO_TRANSITION: FrozenSet[str] = frozenset({
    "GTX_WARP_SPLIT",
    "GTX_WARP_JOIN",
})



# ===========================================================================
# Context → valid groups
# ===========================================================================

_CONTEXT_VALID_GROUPS: Dict[NpuContext, FrozenSet[str]] = {
    NpuContext.C1: frozenset({
        "type_a", "c1_only", "tpose_fill", "shared_mcast", "all_context",
    }),
    NpuContext.C2: frozenset({
        "dma", "tpose_fill", "shared_mcast", "credit_ld",
        "all_context", "credit_ld_chk",
    }),
    NpuContext.C3: frozenset({
        "type_a", "dma", "credit_ld", "all_context", "credit_ld_chk",
    }),
    NpuContext.C4: frozenset({
        "type_a", "tpose_fill", "shared_mcast", "all_context",
    }),
}


# ===========================================================================
# Excluded from context filtering
# ===========================================================================
# Warp markers + control/sync are NOT subject to context validity.
# Markers define context boundaries; control/sync excluded from random
# generation upstream (per YAML excluded_from_context note).

EXCLUDED_FROM_CONTEXT: FrozenSet[str] = frozenset({
    # Warp markers
    "GTX_WARP_START_T", "GTX_WARP_END_T",
    "GTX_WARP_START_S", "GTX_WARP_END_S",
    "GTX_WARP_START_P", "GTX_WARP_END_P",
    "GTX_WARP_SPLIT",   "GTX_WARP_JOIN",
    # Control/Sync
    "GTX_MSYNC", "GTX_EOM", "GTX_BAR", "GTX_WAIT", "GTX_HALT",
    "GTX_INTR", "GTX_FLUSH", "GTX_MEXEC", "GTX_MBAR",
})

from . import control
from . import dma
__all__ = [
    "NpuContext",
    "INITIAL_CONTEXT",
    "WARP_TRANSITIONS",
    "WARP_MARKERS_NO_TRANSITION"
    
]
