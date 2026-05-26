"""ggml op enum → pyspike .elf kernel registry.

Source-of-truth tables come from upstream ggml/include/ggml.h (enums) and
from compare_all_ops.py (test/<OP>/n1s16/n1s16_<kernel>.elf inventory).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# --- ggml op enum (ggml.h: enum ggml_op) ---
GGML_OP_NONE = 0
GGML_OP_DUP = 1
GGML_OP_ADD = 2
GGML_OP_ADD_ID = 3
GGML_OP_ADD1 = 4
GGML_OP_ACC = 5
GGML_OP_SUB = 6
GGML_OP_MUL = 7
GGML_OP_DIV = 8
GGML_OP_SQR = 9
GGML_OP_SQRT = 10
GGML_OP_LOG = 11
GGML_OP_SIN = 12
GGML_OP_COS = 13
GGML_OP_SUM = 14
GGML_OP_SUM_ROWS = 15
GGML_OP_CUMSUM = 16
GGML_OP_MEAN = 17
GGML_OP_ARGMAX = 18
GGML_OP_COUNT_EQUAL = 19
GGML_OP_REPEAT = 20
GGML_OP_REPEAT_BACK = 21
GGML_OP_CONCAT = 22
GGML_OP_SILU_BACK = 23
GGML_OP_NORM = 24
GGML_OP_RMS_NORM = 25
GGML_OP_RMS_NORM_BACK = 26
GGML_OP_GROUP_NORM = 27
GGML_OP_L2_NORM = 28
GGML_OP_MUL_MAT = 29
GGML_OP_MUL_MAT_ID = 30
GGML_OP_OUT_PROD = 31
GGML_OP_SCALE = 32
GGML_OP_SET = 33
GGML_OP_CPY = 34
GGML_OP_CONT = 35
GGML_OP_RESHAPE = 36
GGML_OP_VIEW = 37
GGML_OP_PERMUTE = 38
GGML_OP_TRANSPOSE = 39
GGML_OP_GET_ROWS = 40
GGML_OP_GET_ROWS_BACK = 41
GGML_OP_SET_ROWS = 42
GGML_OP_DIAG = 43
GGML_OP_DIAG_MASK_INF = 44
GGML_OP_DIAG_MASK_ZERO = 45
GGML_OP_SOFT_MAX = 46
GGML_OP_SOFT_MAX_BACK = 47
GGML_OP_ROPE = 48
GGML_OP_ROPE_BACK = 49
GGML_OP_CLAMP = 50
GGML_OP_CONV_TRANSPOSE_1D = 51
GGML_OP_IM2COL = 52
GGML_OP_IM2COL_BACK = 53
GGML_OP_IM2COL_3D = 54
GGML_OP_CONV_2D = 55
GGML_OP_CONV_3D = 56
GGML_OP_CONV_2D_DW = 57
GGML_OP_CONV_TRANSPOSE_2D = 58
GGML_OP_POOL_1D = 59
GGML_OP_POOL_2D = 60
GGML_OP_POOL_2D_BACK = 61
GGML_OP_UPSCALE = 62
GGML_OP_PAD = 63
GGML_OP_PAD_REFLECT_1D = 64
GGML_OP_ROLL = 65
GGML_OP_ARANGE = 66
GGML_OP_TIMESTEP_EMBEDDING = 67
GGML_OP_ARGSORT = 68
GGML_OP_TOP_K = 69
GGML_OP_LEAKY_RELU = 70
GGML_OP_TRI = 71
GGML_OP_FILL = 72
GGML_OP_FLASH_ATTN_EXT = 73
GGML_OP_FLASH_ATTN_BACK = 74
GGML_OP_SSM_CONV = 75
GGML_OP_SSM_SCAN = 76
GGML_OP_WIN_PART = 77
GGML_OP_WIN_UNPART = 78
GGML_OP_GET_REL_POS = 79
GGML_OP_ADD_REL_POS = 80
GGML_OP_RWKV_WKV6 = 81
GGML_OP_GATED_LINEAR_ATTN = 82
GGML_OP_RWKV_WKV7 = 83
GGML_OP_SOLVE_TRI = 84
GGML_OP_UNARY = 85
GGML_OP_MAP_CUSTOM1 = 86
GGML_OP_MAP_CUSTOM2 = 87
GGML_OP_MAP_CUSTOM3 = 88
GGML_OP_CUSTOM = 89
GGML_OP_CROSS_ENTROPY_LOSS = 90
GGML_OP_CROSS_ENTROPY_LOSS_BACK = 91
GGML_OP_OPT_STEP_ADAMW = 92
GGML_OP_OPT_STEP_SGD = 93
GGML_OP_GLU = 94
GGML_OP_COUNT = 95

# --- ggml_unary_op (sub-op when op == GGML_OP_UNARY, read from op_params[0]) ---
GGML_UNARY_OP_ABS = 0
GGML_UNARY_OP_SGN = 1
GGML_UNARY_OP_NEG = 2
GGML_UNARY_OP_STEP = 3
GGML_UNARY_OP_TANH = 4
GGML_UNARY_OP_ELU = 5
GGML_UNARY_OP_RELU = 6
GGML_UNARY_OP_SIGMOID = 7
GGML_UNARY_OP_GELU = 8
GGML_UNARY_OP_GELU_QUICK = 9
GGML_UNARY_OP_SILU = 10
GGML_UNARY_OP_HARDSWISH = 11
GGML_UNARY_OP_HARDSIGMOID = 12
GGML_UNARY_OP_EXP = 13
GGML_UNARY_OP_EXPM1 = 14
GGML_UNARY_OP_SOFTPLUS = 15
GGML_UNARY_OP_GELU_ERF = 16
GGML_UNARY_OP_XIELU = 17
GGML_UNARY_OP_FLOOR = 18
GGML_UNARY_OP_CEIL = 19
GGML_UNARY_OP_ROUND = 20
GGML_UNARY_OP_TRUNC = 21

# --- ggml_glu_op (sub-op when op == GGML_OP_GLU, op_params[0]) ---
GGML_GLU_OP_REGLU = 0
GGML_GLU_OP_GEGLU = 1
GGML_GLU_OP_SWIGLU = 2
GGML_GLU_OP_SWIGLU_OAI = 3
GGML_GLU_OP_GEGLU_ERF = 4
GGML_GLU_OP_GEGLU_QUICK = 5


@dataclass(frozen=True)
class KernelInfo:
    name: str            # 'abs', 'add_vv', ...
    test_dir: str        # 'ABS/n1s16'
    elf_basename: str    # 'n1s16_abs'
    type: str            # 'unary' | 'binary' | 'param' | 'norm' | 'glu' | 'complex' | ...

    def elf_path(self, test_root: str) -> str:
        return os.path.join(test_root, self.test_dir, f"{self.elf_basename}.elf")


# --- direct ggml_op → kernel ---
GGML_OP_TO_KERNEL: dict[int, KernelInfo] = {
    GGML_OP_DUP:          KernelInfo("dup",          "DUP/n1s16",          "n1s16_dup",          "data_move"),
    GGML_OP_ADD:          KernelInfo("add_vv",       "ADD/n1s16",          "n1s16_add_vv",       "binary"),
    GGML_OP_ADD_ID:       KernelInfo("add_id",       "ADD_ID/n1s16",       "n1s16_add_id",       "data_move"),
    GGML_OP_ADD1:         KernelInfo("add1",         "ADD1/n1s16",         "n1s16_add1",         "param"),
    GGML_OP_ACC:          KernelInfo("acc",          "ACC/n1s16",          "n1s16_acc",          "binary"),
    GGML_OP_SUB:          KernelInfo("sub_vv",       "SUB/n1s16",          "n1s16_sub_vv",       "binary"),
    GGML_OP_DIV:          KernelInfo("div_vv",       "DIV/n1s16",          "n1s16_div_vv",       "binary"),
    GGML_OP_SQR:          KernelInfo("sqr",          "SQR/n1s16",          "n1s16_sqr",          "unary"),
    GGML_OP_SQRT:         KernelInfo("sqrt",         "SQRT/n1s16",         "n1s16_sqrt",         "unary"),
    GGML_OP_LOG:          KernelInfo("log",          "LOG/n1s16",          "n1s16_log",          "unary"),
    GGML_OP_SIN:          KernelInfo("sin",          "SIN/n1s16",          "n1s16_sin",          "unary"),
    GGML_OP_COS:          KernelInfo("cos",          "COS/n1s16",          "n1s16_cos",          "unary"),
    GGML_OP_SUM:          KernelInfo("sum",          "SUM/n1s16",          "n1s16_sum",          "reduction"),
    GGML_OP_SUM_ROWS:     KernelInfo("sum_rows",     "SUM_ROWS/n1s16",     "n1s16_sum_rows",     "reduction"),
    GGML_OP_CUMSUM:       KernelInfo("cumsum",       "CUMSUM/n1s16",       "n1s16_cumsum",       "special"),
    GGML_OP_MEAN:         KernelInfo("mean",         "MEAN/n1s16",         "n1s16_mean",         "reduction"),
    GGML_OP_ARGMAX:       KernelInfo("argmax",       "ARGMAX/n1s16",       "n1s16_argmax",       "complex"),
    GGML_OP_COUNT_EQUAL:  KernelInfo("count_equal",  "COUNT_EQUAL/n1s16",  "n1s16_count_equal",  "complex"),
    GGML_OP_REPEAT:       KernelInfo("repeat",       "REPEAT/n1s16",       "n1s16_repeat",       "data_move"),
    GGML_OP_CONCAT:       KernelInfo("concat",       "CONCAT/n1s16",       "n1s16_concat",       "data_move"),
    GGML_OP_NORM:         KernelInfo("norm",         "NORM/n1s16",         "n1s16_norm",         "norm"),
    GGML_OP_RMS_NORM:     KernelInfo("rms_norm",     "RMS_NORM/n1s16",     "n1s16_rms_norm",     "norm"),
    GGML_OP_GROUP_NORM:   KernelInfo("group_norm",   "GROUP_NORM/n1s16",   "n1s16_group_norm",   "norm"),
    GGML_OP_L2_NORM:      KernelInfo("l2_norm",      "L2_NORM/n1s16",      "n1s16_l2_norm",      "norm"),
    GGML_OP_MUL_MAT_ID:   KernelInfo("mul_mat_id",   "MUL_MAT_ID/n1s16",   "n1s16_mul_mat_id",   "complex"),
    GGML_OP_OUT_PROD:     KernelInfo("out_prod",     "OUT_PROD/n1s16",     "n1s16_out_prod",     "complex"),
    GGML_OP_SCALE:        KernelInfo("scale",        "SCALE/n1s16",        "n1s16_scale",        "param"),
    GGML_OP_SET:          KernelInfo("set",          "SET/n1s16",          "n1s16_set",          "data_move"),
    GGML_OP_CPY:          KernelInfo("cpy",          "CPY/n1s16",          "n1s16_cpy",          "data_move"),
    GGML_OP_GET_ROWS:     KernelInfo("get_rows",     "GET_ROWS/n1s16",     "n1s16_get_rows",     "data_move"),
    GGML_OP_SET_ROWS:     KernelInfo("set_rows",     "SET_ROWS/n1s16",     "n1s16_set_rows",     "data_move"),
    GGML_OP_DIAG:         KernelInfo("diag",         "DIAG/n1s16",         "n1s16_diag",         "complex"),
    GGML_OP_DIAG_MASK_INF:  KernelInfo("diag_mask_inf",  "DIAG_MASK_INF/n1s16",  "n1s16_diag_mask_inf",  "special"),
    GGML_OP_DIAG_MASK_ZERO: KernelInfo("diag_mask_zero", "DIAG_MASK_ZERO/n1s16", "n1s16_diag_mask_zero", "special"),
    GGML_OP_SOFT_MAX:     KernelInfo("softmax",      "SOFT_MAX/n1s16",     "n1s16_softmax",      "norm"),
    GGML_OP_CLAMP:        KernelInfo("clamp",        "CLAMP/n1s16",        "n1s16_clamp",        "param"),
    GGML_OP_IM2COL:       KernelInfo("im2col",       "IM2COL/n1s16",       "n1s16_im2col",       "complex"),
    GGML_OP_IM2COL_3D:    KernelInfo("im2col_3d",    "IM2COL_3D/n1s16",    "n1s16_im2col_3d",    "complex"),
    GGML_OP_CONV_2D:      KernelInfo("conv_2d",      "CONV_2D/n1s16",      "n1s16_conv_2d",      "complex"),
    GGML_OP_CONV_3D:      KernelInfo("conv_3d",      "CONV_3D/n1s16",      "n1s16_conv_3d",      "complex"),
    GGML_OP_CONV_2D_DW:   KernelInfo("conv_2d_dw",   "CONV_2D_DW/n1s16",   "n1s16_conv_2d_dw",   "complex"),
    GGML_OP_CONV_TRANSPOSE_2D: KernelInfo("conv_transpose_2d", "CONV_TRANSPOSE_2D/n1s16", "n1s16_conv_transpose_2d", "complex"),
    GGML_OP_POOL_1D:      KernelInfo("pool_1d",      "POOL_1D/n1s16",      "n1s16_pool_1d",      "complex"),
    GGML_OP_POOL_2D:      KernelInfo("pool_2d",      "POOL_2D/n1s16",      "n1s16_pool_2d",      "complex"),
    GGML_OP_UPSCALE:      KernelInfo("upscale",      "UPSCALE/n1s16",      "n1s16_upscale",      "data_move"),
    GGML_OP_PAD:          KernelInfo("pad",          "PAD/n1s16",          "n1s16_pad",          "data_move"),
    GGML_OP_PAD_REFLECT_1D: KernelInfo("pad_reflect_1d", "PAD_REFLECT_1D/n1s16", "n1s16_pad_reflect_1d", "complex"),
    GGML_OP_ROLL:         KernelInfo("roll",         "ROLL/n1s16",         "n1s16_roll",         "data_move"),
    GGML_OP_ARANGE:       KernelInfo("arange",       "ARANGE/n1s16",       "n1s16_arange",       "special"),
    GGML_OP_ARGSORT:      KernelInfo("argsort",      "ARGSORT/n1s16",      "n1s16_argsort",      "complex"),
    GGML_OP_TOP_K:        KernelInfo("top_k",        "TOP_K/n1s16",        "n1s16_top_k",        "complex"),
    GGML_OP_LEAKY_RELU:   KernelInfo("leaky_relu",   "LEAKY_RELU/n1s16",   "n1s16_leaky_relu",   "unary"),
    GGML_OP_TRI:          KernelInfo("tri",          "TRI/n1s16",          "n1s16_tri",          "complex"),
    GGML_OP_FILL:         KernelInfo("fill",         "FILL/n1s16",         "n1s16_fill",         "param"),
    GGML_OP_FLASH_ATTN_EXT: KernelInfo("flash_attn_ext", "FLASH_ATTN_EXT/n1s16", "n1s16_flash_attn_ext", "complex"),
    GGML_OP_RWKV_WKV6:    KernelInfo("rwkv_wkv6",    "RWKV_WKV6/n1s16",    "n1s16_rwkv_wkv6",    "complex"),
    GGML_OP_GATED_LINEAR_ATTN: KernelInfo("gated_linear_attn", "GATED_LINEAR_ATTN/n1s16", "n1s16_gated_linear_attn", "complex"),
    GGML_OP_RWKV_WKV7:    KernelInfo("rwkv_wkv7",    "RWKV_WKV7/n1s16",    "n1s16_rwkv_wkv7",    "complex"),
    GGML_OP_SOLVE_TRI:    KernelInfo("solve_tri",    "SOLVE_TRI/n1s16",    "n1s16_solve_tri",    "complex"),
}

# --- GGML_OP_UNARY sub-op → kernel ---
GGML_UNARY_TO_KERNEL: dict[int, KernelInfo] = {
    GGML_UNARY_OP_ABS:         KernelInfo("abs",         "ABS/n1s16",         "n1s16_abs",         "unary"),
    GGML_UNARY_OP_SGN:         KernelInfo("sgn",         "SGN/n1s16",         "n1s16_sgn",         "unary"),
    GGML_UNARY_OP_NEG:         KernelInfo("neg",         "NEG/n1s16",         "n1s16_neg",         "unary"),
    GGML_UNARY_OP_STEP:        KernelInfo("step",        "STEP/n1s16",        "n1s16_step",        "unary"),
    GGML_UNARY_OP_TANH:        KernelInfo("tanh",        "TANH/n1s16",        "n1s16_tanh",        "unary"),
    GGML_UNARY_OP_ELU:         KernelInfo("elu",         "ELU/n1s16",         "n1s16_elu",         "unary"),
    GGML_UNARY_OP_RELU:        KernelInfo("relu",        "RELU/n1s16",        "n1s16_relu",        "unary"),
    GGML_UNARY_OP_SIGMOID:     KernelInfo("sigmoid",     "SIGMOID/n1s16",     "n1s16_sigmoid",     "unary"),
    GGML_UNARY_OP_GELU:        KernelInfo("gelu",        "GELU/n1s16",        "n1s16_gelu",        "unary"),
    GGML_UNARY_OP_GELU_QUICK:  KernelInfo("gelu_quick",  "GELU_QUICK/n1s16",  "n1s16_gelu_quick",  "unary"),
    GGML_UNARY_OP_SILU:        KernelInfo("silu",        "SILU/n1s16",        "n1s16_silu",        "unary"),
    GGML_UNARY_OP_HARDSWISH:   KernelInfo("hardswish",   "HARDSWISH/n1s16",   "n1s16_hardswish",   "unary"),
    GGML_UNARY_OP_HARDSIGMOID: KernelInfo("hardsigmoid", "HARDSIGMOID/n1s16", "n1s16_hardsigmoid", "unary"),
    GGML_UNARY_OP_EXP:         KernelInfo("exp",         "EXP/n1s16",         "n1s16_exp",         "unary"),
    GGML_UNARY_OP_EXPM1:       KernelInfo("expm1",       "EXPM1/n1s16",       "n1s16_expm1",       "unary"),
    GGML_UNARY_OP_SOFTPLUS:    KernelInfo("softplus",    "SOFTPLUS/n1s16",    "n1s16_softplus",    "unary"),
    GGML_UNARY_OP_GELU_ERF:    KernelInfo("gelu_erf",    "GELU_ERF/n1s16",    "n1s16_gelu_erf",    "unary"),
    GGML_UNARY_OP_XIELU:       KernelInfo("xielu",       "XIELU/n1s16",       "n1s16_xielu",       "unary"),
    GGML_UNARY_OP_FLOOR:       KernelInfo("floor",       "FLOOR/n1s16",       "n1s16_floor",       "unary"),
    GGML_UNARY_OP_CEIL:        KernelInfo("ceil",        "CEIL/n1s16",        "n1s16_ceil",        "unary"),
    GGML_UNARY_OP_ROUND:       KernelInfo("round",       "ROUND/n1s16",       "n1s16_round",       "unary"),
    GGML_UNARY_OP_TRUNC:       KernelInfo("trunc",       "TRUNC/n1s16",       "n1s16_trunc",       "unary"),
}

# --- GGML_OP_GLU sub-op → kernel ---
GGML_GLU_TO_KERNEL: dict[int, KernelInfo] = {
    GGML_GLU_OP_REGLU:       KernelInfo("reglu",       "REGLU/n1s16",       "n1s16_reglu",       "glu"),
    GGML_GLU_OP_GEGLU:       KernelInfo("geglu",       "GEGLU/n1s16",       "n1s16_geglu",       "glu"),
    GGML_GLU_OP_SWIGLU_OAI:  KernelInfo("swiglu_oai",  "SWIGLU_OAI/n1s16",  "n1s16_swiglu_oai",  "glu"),
    GGML_GLU_OP_GEGLU_ERF:   KernelInfo("geglu_erf",   "GEGLU_ERF/n1s16",   "n1s16_geglu_erf",   "glu"),
    GGML_GLU_OP_GEGLU_QUICK: KernelInfo("geglu_quick", "GEGLU_QUICK/n1s16", "n1s16_geglu_quick", "glu"),
}


def resolve(ggml_op: int, op_params: Optional[tuple] = None) -> Optional[KernelInfo]:
    """Map a ggml_op (+ optional op_params) to a KernelInfo, or None if unsupported.

    For GGML_OP_UNARY / GGML_OP_GLU, op_params[0] selects the sub-op.
    """
    if ggml_op == GGML_OP_UNARY:
        sub = op_params[0] if op_params else 0
        return GGML_UNARY_TO_KERNEL.get(sub)
    if ggml_op == GGML_OP_GLU:
        sub = op_params[0] if op_params else 0
        return GGML_GLU_TO_KERNEL.get(sub)
    return GGML_OP_TO_KERNEL.get(ggml_op)


# Ops that are pure metadata reshuffles on the buffer — no kernel work needed.
NO_OP_GRAPH_OPS = frozenset({
    GGML_OP_NONE,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_CONT,
})
