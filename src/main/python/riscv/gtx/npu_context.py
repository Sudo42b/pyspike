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
"""NPU context FSM — C1/C2/C3/C4 contexts, transitions, validity.

MANUALLY SYNCED from src/main/python/riscv/context_map.yaml on 2026-05-11.
When the YAML changes, update this file. (No runtime PyYAML dependency
per CLAUDE.md "NumPy 외부 추가 런타임 의존성 신규 도입 금지".)

NPU contexts (persistent across instructions):
  C1: plan outside     — before START_P (initial / after END_P)
  C2: shared inside    — inside START_P + START_S
  C3: thread inside    — inside START_P + START_T
  C4: plan inside only — inside START_P, outside S/T

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
# Instruction groups (9 groups, 132 total instructions)
# ===========================================================================
# Each tuple mirrors a `groups.<name>.instructions` list in context_map.yaml.
# Keep ordering identical to YAML for diff-friendly manual sync.

_GROUP_TYPE_A: Tuple[str, ...] = (
    # Matrix Multiplication (10)
    "GTX_MM", "GTX_MM_S", "GTX_MM_O", "GTX_MM_V", "GTX_MM_T",
    "GTX_MMC", "GTX_MMC_S", "GTX_MMC_O", "GTX_MMC_V", "GTX_MMC_T",
    # Convolution (2)
    "GTX_IM2COL_N", "GTX_IM2COL_D",
    # Scalar Calculation FP (7)
    "GTX_ADD_VS", "GTX_SUB_VS", "GTX_MUL_VS", "GTX_DIV_VS",
    "GTX_FMADD_VS", "GTX_MAX_VS", "GTX_MIN_VS",
    # Scalar Calculation INT (7)
    "GTX_ADD_IS", "GTX_SUB_IS", "GTX_MUL_IS", "GTX_DIV_IS",
    "GTX_FMADD_IS", "GTX_MAX_IS", "GTX_MIN_IS",
    # Vector Calculation FP (22)
    "GTX_ADD_VV", "GTX_SUB_VV", "GTX_MUL_VV", "GTX_DIV_VV",
    "GTX_DOT_VVS", "GTX_FMADD_VVV", "GTX_SUM_VS",
    "GTX_SQRT_V", "GTX_EXP_V", "GTX_LN_V",
    "GTX_ABS_V", "GTX_NEG_V", "GTX_SIGN_V", "GTX_STEP_V",
    "GTX_CEIL_V", "GTX_TRUNC_V", "GTX_FLOOR_V", "GTX_RNE_V",
    "GTX_CLAMP_MIN_V", "GTX_CLAMP_MAX_V",
    "GTX_ACCUM_V", "GTX_ARANGE_V",
    # Vector Calculation INT (22)
    "GTX_ADD_II", "GTX_SUB_II", "GTX_MUL_II", "GTX_DIV_II",
    "GTX_DOT_IIS", "GTX_FMADD_III", "GTX_SUM_IS",
    "GTX_SQRT_I", "GTX_EXP_I", "GTX_LN_I",
    "GTX_ABS_I", "GTX_NEG_I", "GTX_SIGN_I", "GTX_STEP_I",
    "GTX_CEIL_I", "GTX_TRUNC_I", "GTX_FLOOR_I", "GTX_RNE_I",
    "GTX_AND_II", "GTX_OR_II", "GTX_NOT_I", "GTX_SHIFT_I",
    # Format Conversion SPM (5)
    "GTX_SCVT_QH", "GTX_SCVT_HQ", "GTX_SCVT_IH", "GTX_SCVT_HI",
    "GTX_SCVT_HN",
    # Activation FP (4)
    "GTX_PRELU", "GTX_GELU", "GTX_TANH", "GTX_SIGM",
    # Activation IMM (4)
    "GTX_PRELU_IMM", "GTX_GELU_IMM", "GTX_TANH_IMM", "GTX_SIGM_IMM",
    # Softmax (4)
    "GTX_ESUM", "GTX_SOFTMAX", "GTX_ESUM_I", "GTX_SOFTMAX_I",
    # Pooling (2)
    "GTX_POOL_M", "GTX_POOL_A",
    # DMA SVR (2)
    "GTX_LOAD_SVR", "GTX_STORE_SVR",
    # SPR (4) — RDSPR is C1/C3/C4 valid per ISS confirmation
    "GTX_RDSPR", "GTX_WRSPR", "GTX_CPSVR", "GTX_MVSVR",
)

_GROUP_C1_ONLY: Tuple[str, ...] = (
    "GTX_FCVT_SH", "GTX_FCVT_HS", "GTX_FCVT_DH", "GTX_FCVT_HD",
    "GTX_MCAST_G2S", "GTX_COPY_MEM",
)

_GROUP_DMA: Tuple[str, ...] = (
    "GTX_LOAD", "GTX_STORE", "GTX_COPY",
    "GTX_LOAD_3D", "GTX_STORE_3D",
    "GTX_CREDIT_ST", "GTX_CREDIT_ST_CHK",
)

_GROUP_TPOSE_FILL: Tuple[str, ...] = ("GTX_TPOSE", "GTX_FILL")

_GROUP_SHARED_MCAST: Tuple[str, ...] = ("GTX_MCAST_S2L", "GTX_MCAST_S2S")

_GROUP_CREDIT_LD: Tuple[str, ...] = ("GTX_CREDIT_LD",)

_GROUP_ALL_CONTEXT: Tuple[str, ...] = ("GTX_OPSET",)

_GROUP_CREDIT_LD_CHK: Tuple[str, ...] = ("GTX_CREDIT_LD_CHK",)


GROUPS: Dict[str, Tuple[str, ...]] = {
    "type_a":         _GROUP_TYPE_A,
    "c1_only":        _GROUP_C1_ONLY,
    "dma":            _GROUP_DMA,
    "tpose_fill":     _GROUP_TPOSE_FILL,
    "shared_mcast":   _GROUP_SHARED_MCAST,
    "credit_ld":      _GROUP_CREDIT_LD,
    "all_context":    _GROUP_ALL_CONTEXT,
    "credit_ld_chk":  _GROUP_CREDIT_LD_CHK,
}

# Reverse index: mnemonic → group_name (built once at module import).
_INSTR_TO_GROUP: Dict[str, str] = {
    mnem: group_name
    for group_name, instructions in GROUPS.items()
    for mnem in instructions
}


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


# ===========================================================================
# Public API
# ===========================================================================

def get_group(mnemonic: str) -> Optional[str]:
    """Return validity group name (e.g., 'type_a', 'dma') or None."""
    return _INSTR_TO_GROUP.get(mnemonic)


def is_valid_in_context(mnemonic: str, context: NpuContext) -> bool:
    """True if `mnemonic` may execute in `context`.

    Warp markers and control/sync are always allowed (per
    context_map.yaml excluded_from_context).

    Unknown mnemonics default to True (lenient). Caller may upgrade
    to strict mode by checking `get_group(mnemonic) is not None` first.
    """
    if mnemonic in EXCLUDED_FROM_CONTEXT:
        return True
    group = _INSTR_TO_GROUP.get(mnemonic)
    if group is None:
        return True  # unknown → lenient
    return group in _CONTEXT_VALID_GROUPS[context]


def is_warp_marker(mnemonic: str) -> bool:
    """True iff mnemonic is a warp marker (may or may not transition)."""
    return mnemonic in WARP_TRANSITIONS or mnemonic in WARP_MARKERS_NO_TRANSITION


def apply_transition(current: NpuContext, mnemonic: str) -> NpuContext:
    """Apply context transition for a warp marker. Non-markers unchanged.

    SPLIT/JOIN are markers but cause no transition (structural only).
    Illegal transitions (current ≠ expected source) are silently ignored
    in lenient mode — caller can pre-check with `is_legal_transition`.
    """
    if mnemonic not in WARP_TRANSITIONS:
        return current
    src, dst = WARP_TRANSITIONS[mnemonic]
    if current is not src:
        return current
    return dst


def is_legal_transition(current: NpuContext, mnemonic: str) -> bool:
    """True if `mnemonic` is a legal warp transition from `current`.

    Returns True for SPLIT/JOIN (no-op markers).
    Returns False for non-marker mnemonics.
    """
    if mnemonic in WARP_MARKERS_NO_TRANSITION:
        return True
    if mnemonic not in WARP_TRANSITIONS:
        return False
    src, _dst = WARP_TRANSITIONS[mnemonic]
    return current is src


__all__ = [
    "NpuContext",
    "INITIAL_CONTEXT",
    "WARP_TRANSITIONS",
    "WARP_MARKERS_NO_TRANSITION",
    "GROUPS",
    "EXCLUDED_FROM_CONTEXT",
    "get_group",
    "is_valid_in_context",
    "is_warp_marker",
    "apply_transition",
    "is_legal_transition",
]
