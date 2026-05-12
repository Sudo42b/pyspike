"""GTX instruction FSM reference.

Each ``custom0`` or ``custom1`` invocation runs a single instruction pipeline
cycle and returns to ``IDLE``.

Instruction FSM:
    1. IDLE
        Waiting state before and after a single instruction is processed.

    2. DECODE
        Extract dispatch keys from the instruction.

        - ``funct7 = insn.funct``
        - ``funct3 = (xd << 2) | (xs1 << 1) | xs2``

        The mnemonic is then resolved through the instruction registry.

    3. DISPATCH
        Resolve the handler and validate the instruction against the current
        context.

        For ``custom0``, dispatch uses a two-level lookup:

        - first ``sub_table[None]``
        - then ``sub_table[funct3]`` when needed

        For ``custom1``, dispatch uses ``self._custom1[funct3]``.

        This stage also:

        - maps the mnemonic to a group in ``context_map.yaml``
        - checks whether the group is valid for ``npu._context``
        - records a pending context transition if the instruction is a warp
          marker

    4. EXECUTE
        Invoke ``handler(proc, insn, xs1, xs2)`` and store the return value in
        ``ctx[\"rd\"]``. Warp markers are typically no-op handlers whose main
        effect is the deferred context transition.

    5. WRITEBACK
        Apply the pending context transition for warp markers and clear OPSET
        staging when required.

        For ``custom0`` instructions with ``funct7 != 0x4A``:

        - ``gspr[0x003] = 0``
        - ``gspr[0x005] = 0``

        The OPSET instruction itself, identified by ``funct7 == 0x4A``, keeps
        staging intact for vendor parity.

    6. IDLE
        Return ``ctx[\"rd\"]`` and wait for the next instruction.
"""

from __future__ import annotations

import enum
from typing import Dict, FrozenSet, Optional, Tuple

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
# Public API
# ===========================================================================

def get_group(mnemonic: str) -> Optional[str]:
    """Return validity group name (e.g., 'type_a', 'dma') or None."""
    return _INSTR_TO_GROUP.get(mnemonic)

