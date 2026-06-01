"""Vector arithmetic handlers — port of gtx_npu_vec.cc / gtx_npu_custom0.cc.

Owner of the VECTOR (VV/II) funct7 set: 0x18 (arith), 0x19 (vector FMADD),
0x1C-0x1E (math/sign/round), 0x1F (clamp/accum/arange + L0 bitwise logic).
The SCALAR (_VS/_IS) funct7 family — 0x10 SASMD, 0x11 FMADD_S, 0x13 MINMAX_S —
lives in :mod:`scalar`. The registry keys solely on ``(funct7, funct3)``, so the
two modules must own disjoint funct7 sets; both are imported for side-effect by
``npu.py``.

Dispatch model: the registry has *already* resolved ``(funct7, funct3)`` to a
specific handler, so each handler calls its leaf directly with the sub-op
(an :class:`enum.IntEnum` member) baked in — there is no second switch on the
instruction bits. The element-wise kernels themselves are selected through
dispatch tables (``enum → callable``) rather than if/elif ladders. ``.v``/``.vv``
handlers take the L1 (VV) path; ``.i``/``.ii`` take the L0 SVR (II) path.
"""
from __future__ import annotations

import enum

import numpy as np

from ...inst_handler import inst_register

from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM, MX_IO_DTYPE, MX_IO_BYTES
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu, operand3
# Shared MX I/O-width helpers (FP32 default / FP16 toggle) — the single
# definitions live in the package __init__; numeric VV/II ops route through them.
from . import (
    _io_low, _io_high, _fp32_low32,
    _l1_view_addr_io as _l1_view_addr,
    _l0_block_view_io as _l0_block_view,
    _l0_block_view_uint, _IO_UINT, _IO_MASK,
)

# funct7 opcodes for the unary families — passed by the handlers to the unary
# leaves so the right kernel table is chosen (and for debug tracing).
F7_MATH_V = 0b0011100       # 0x1C  sqrt / exp / ln
F7_SIGN_V = 0b0011101       # 0x1D  abs / neg / sign / step
F7_ROUND_V = 0b0011110      # 0x1E  ceil / trunc / floor / rne


# Sub-op selectors (funct3 & 3). IntEnum so a member is still a plain int where
# one is needed, while reading as ``Arith.ADD`` / ``Logic.SHIFT`` at the callsite.
class Arith(enum.IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


class Math(enum.IntEnum):
    SQRT = 0
    EXP = 1
    LN = 2


class Sign(enum.IntEnum):
    ABS = 0
    NEG = 1
    SIGN = 2
    STEP = 3


class Round(enum.IntEnum):
    CEIL = 0
    TRUNC = 1
    FLOOR = 2
    RNE = 3


class Clamp(enum.IntEnum):
    MIN = 0
    MAX = 1
    ACCUM = 2
    ARANGE = 3


class Logic(enum.IntEnum):
    AND = 0
    OR = 1
    NOT = 2
    SHIFT = 3


# exp/ln base selection from rs2[1:0] (mode), per HW spec — torch-native:
#   exp.{v,i}: 00 e^A | 01 2^A | 10 3^A | 11 5^A
#   ln.{v,i} : 00 ln  | 01 log2 | 10 log10  (11 → ln)
_EXP_FN = (np.exp, 
           np.exp2,
           lambda x: np.power(3.0, x), 
           lambda x: np.power(5.0, x))
_LN_FN = (np.log, 
          np.log2, 
          np.log10, 
          np.log)


# =============================================================================
# 1. Vector kernels (FP32 internal, FP16 output) + dispatch tables
# =============================================================================
def _as_fp32(a) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.astype(np.float32)
    return np.asarray(a, dtype=np.float32)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise divide; div-by-zero → 0.0 (vendor gtx_npu_vec.cc:333)."""
    safe = np.where(b == 0.0, np.ones_like(b), b)
    return np.where(b == 0.0, np.zeros_like(a), a / safe)


_ARITH_FN = {
    Arith.ADD: np.add,
    Arith.SUB: np.subtract,
    Arith.MUL: np.multiply,
    Arith.DIV: _div,
}


def sasmd_kernel(a, b, op: Arith) -> np.ndarray:
    """SASMD element-wise FP32 internal, MX_IO_DTYPE output. ``b`` scalar/array."""
    a_f32 = _as_fp32(a)
    if isinstance(b, np.ndarray) and b.ndim > 0:
        b_f32 = b.astype(np.float32)
    elif hasattr(b, 'shape') and getattr(b, 'shape', ()):
        b_f32 = np.asarray(b, dtype=np.float32)
    else:
        b_f32 = np.full_like(a_f32, float(b))
    return _ARITH_FN[op](a_f32, b_f32).astype(MX_IO_DTYPE)


def clamp_min_kernel(a, scalar) -> np.ndarray:
    """``out[i] = max(a[i], scalar)``."""
    return np.clip(_as_fp32(a), min=float(scalar)).astype(MX_IO_DTYPE)


def clamp_max_kernel(a, scalar) -> np.ndarray:
    """``out[i] = min(a[i], scalar)``."""
    return np.clip(_as_fp32(a), max=float(scalar)).astype(MX_IO_DTYPE)


def accum_kernel(a) -> np.ndarray:
    """Prefix sum: FP32 accumulator across whole vec, per-element MX_IO cast.

    ``torch.cumsum`` is the left-to-right vectorised form of the Python
    accumulator loop — same numerical order, no per-element kernel launch.
    """
    return np.cumsum(_as_fp32(a).reshape(-1), axis=0).astype(MX_IO_DTYPE)


def arange_kernel(n: int, start, step) -> np.ndarray:
    """``out[i] = start + i*step`` (FP32 internal)."""
    from ....config_params import DEVICE
    idx = np.arange(int(n), dtype=np.float32, device=DEVICE)
    return (float(start) + idx * float(step)).astype(MX_IO_DTYPE)


# CLAMP_V (0x1F, L1) — each kernel takes (view, rs2, vec_size); only the ones
# that need a scalar / size touch rs2 / vec_size.
_CLAMP_FN = {
    Clamp.MIN: lambda view, rs2, n: clamp_min_kernel(view, _io_low(rs2)),
    Clamp.MAX: lambda view, rs2, n: clamp_max_kernel(view, _io_low(rs2)),
    Clamp.ACCUM: lambda view, rs2, n: accum_kernel(view),
    Clamp.ARANGE: lambda view, rs2, n: arange_kernel(n, _io_low(rs2),
                                                     _io_high(rs2)),
}


def _shift(a_uint: np.ndarray, rs2: int) -> np.ndarray:
    """rs2 = shift_num[3:0], shift_mode[4] (0 → ``>>``, 1 → ``<<``).

    Matches firmware ``__shift_i`` packing ``rs2 = (shift_mode << 4) | shift_num``
    (intrin_level1.c) and the ISS decode. The earlier shift_num[4:0]/mode[5]
    layout swallowed the mode bit into the amount and read the direction from the
    wrong bit, so every left shift became a right shift by 16+num — breaking
    COUNT_EQUAL's popcount fold. Operates on the raw bit pattern at the MX I/O
    width; promote to int64 (values non-negative, so ``>>`` is logical) and mask
    back to the I/O width on left-shift overflow.
    """
    amt = rs2 & 0xF
    a = a_uint.astype(np.int64)
    shifted = ((a << amt) & _IO_MASK) if (rs2 & 0x10) else (a >> amt)
    return shifted.astype(_IO_UINT)


# LOGIC (0x1F, L0) on FP16 raw bits (vendor exec_bitwise_imm). AND/OR use the
# second register's bits ``b``; NOT/SHIFT ignore it (SHIFT reads raw rs2).
_LOGIC_FN = {
    Logic.AND: lambda a, b, rs2: a & b,
    Logic.OR: lambda a, b, rs2: a | b,
    Logic.NOT: lambda a, b, rs2: a ^ _IO_UINT(_IO_MASK),
    Logic.SHIFT: lambda a, b, rs2: _shift(a, rs2),
}


# =============================================================================
# 2. Helpers — the numeric I/O views (_l1_view_addr / _l0_block_view), operand
# decoders (_io_low / _io_high) and the raw-bits LOGIC view (_l0_block_view_uint)
# are all imported above from the package __init__ so their width tracks
# config_params.MX_IO_DTYPE (FP32 → 8×uint32 / FP16 → 16×uint16).
# =============================================================================
# 3. Unary apply (MATH / SIGN / ROUND) — dispatch tables, no if/elif ladder
# =============================================================================
# SIGN / ROUND act on the I/O view directly (a sign bit / integer rounding
# doesn't benefit from FP32 promotion). MATH promotes to FP32 for precision.
_SIGN_FN = {
    Sign.ABS: np.abs,
    Sign.NEG: np.negative,
    Sign.SIGN: np.sign,
    Sign.STEP: lambda v: (v > 0.0).astype(MX_IO_DTYPE),
}
_ROUND_FN = {
    Round.CEIL: np.ceil,
    Round.TRUNC: np.trunc,
    Round.FLOOR: np.floor,
    Round.RNE: np.round,
}


def _ln(f32: np.ndarray, mode: int) -> np.ndarray:
    """ln / log2 / log10 (mode[1:0]); non-positive inputs → 0."""
    tiny = np.finfo(np.float32).tiny
    ln_a = _LN_FN[mode & 3](np.maximum(f32, tiny))
    return np.where(f32 > 0.0, ln_a, np.zeros_like(f32))


_MATH_FN = {
    Math.SQRT: lambda f32, 
    mode: np.sqrt(f32),
    Math.EXP: lambda f32, 
    mode: _EXP_FN[mode & 3](f32),   # e^A/2^A/3^A/5^A
    Math.LN: _ln,
}


def _math(view: np.ndarray, sub, mode: int) -> np.ndarray:
    return _MATH_FN[sub](view.astype(np.float32), mode).astype(MX_IO_DTYPE)


def _sign(view: np.ndarray, sub, mode: int) -> np.ndarray:
    return _SIGN_FN[sub](view)


def _round(view: np.ndarray, sub, mode: int) -> np.ndarray:
    return _ROUND_FN[sub](view)


_UNARY_FAMILY = {F7_MATH_V: _math, F7_SIGN_V: _sign, F7_ROUND_V: _round}


def _apply_unary(funct7: int, sub, view: np.ndarray,
                 mode: int = 0) -> np.ndarray:
    """Element-wise unary kernel for funct7 0x1C/0x1D/0x1E (``mode`` = rs2[1:0],
    consumed only by MATH exp/ln for base selection)."""
    return _UNARY_FAMILY[funct7](view, sub, mode)


# =============================================================================
# 4. Preamble + leaf dispatchers — one per (family, L1/L0). The registry has
#    already keyed on (funct7, funct3), so each handler calls its leaf directly
#    with the sub-op baked in — no re-dispatch on instruction bits.
# =============================================================================
def _prep(npu, proc, inst) -> tuple:
    """Shared vector-handler preamble: resolve (nest, spu), decode rs1/rs2,
    stage OPERAND2. Returns ``(nest, spu, rs1, rs2, vec_size)``."""
    rs1 = int(proc.state.XPR[inst.rs1])
    rs2 = int(proc.state.XPR[inst.rs2])
    vec_size = (rs1 & 0xFFFF) or 0x10000
    nest, spu = _resolve_nest_spu(npu)
    assert nest < NEST_NUM, f"NEST id {nest} >= NEST_NUM={NEST_NUM}"
    assert spu < SPU_NUM, f"SPU id {spu} >= SPU_NUM={SPU_NUM}"
    npu.gspr[GSPR['GSPR_GTX_OPERAND2'].address] = rs2
    return nest, spu, rs1, rs2, vec_size


def _l1_addrs(npu, nest: int, spu: int) -> tuple:
    """``(ADDRA, ADDRB, ADDRR)`` for (nest, spu) — the L1 operand anchors."""
    lspr = npu.lspr[nest][spu]
    return (lspr.get(LSPR['SPM_ADDRA'].address, 0),
            lspr.get(LSPR['SPM_ADDRB'].address, 0),
            lspr.get(LSPR['SPM_ADDRR'].address, 0))


# ----- ARITH (0x18) ----------------------------------------------------------
def _arith_vv(npu, proc, inst, sub: Arith) -> int:
    """L1 vector-vector ADD/SUB/MUL/DIV (A,B,R from SPM_ADDR*)."""
    nest, spu, _rs1, _rs2, vsz = _prep(npu, proc, inst)
    addr_a, addr_b, addr_r = _l1_addrs(npu, nest, spu)
    va = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    vb = _l1_view_addr(npu, nest, spu, addr_b, vsz)
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = sasmd_kernel(va, vb, op=sub)
    return 0


def _arith_ii(npu, proc, inst, sub: Arith) -> int:
    """L0 SVR ADD/SUB/MUL/DIV. a=rs1[4:0], b=rs2[4:0], r=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    va = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    vb = _l0_block_view(npu, nest, spu, rs2 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _l0_block_view(npu, nest, spu, r_reg)[...] = sasmd_kernel(va, vb, op=sub)
    return 0


def _fmadd_iii(npu, proc, inst) -> int:
    """L0 a*b + c on SVR regs. a=rs1[4:0], b=rs2[4:0], c=rs2[9:5], r=OPERAND3."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    va = _as_fp32(_l0_block_view(npu, nest, spu, rs1 & 0x1F))
    vb = _as_fp32(_l0_block_view(npu, nest, spu, rs2 & 0x1F))
    vc = _as_fp32(_l0_block_view(npu, nest, spu, (rs2 >> 5) & 0x1F))
    r_reg = operand3(npu, inst.rd) & 0x1F
    _l0_block_view(npu, nest, spu, r_reg)[...] = (va * vb + vc).astype(MX_IO_DTYPE)
    return 0


# ----- MATH / SIGN / ROUND (0x1C / 0x1D / 0x1E) ------------------------------
def _unary_v(npu, proc, inst, funct7: int, sub) -> int:
    """L1 unary A → R. rs2[1:0] selects exp/ln base (MATH only)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a, _addr_b, addr_r = _l1_addrs(npu, nest, spu)
    view = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    result = _apply_unary(funct7, sub, view, rs2 & 0x3)
    import os as _os, sys as _sys
    if _os.environ.get("GTX_DEBUG_UNARY"):
        print(f"[UNARY] n{nest}s{spu} f7=0x{funct7:02x} sub={sub} "
              f"vs={vsz} addr_a=0x{addr_a:x} addr_r=0x{addr_r:x} "
              f"in0={float(view.reshape(-1)[0]):.3f} out0={float(result.reshape(-1)[0]):.3f}",
              file=_sys.stderr, flush=True)
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = result
    return 0


def _unary_i(npu, proc, inst, funct7: int, sub) -> int:
    """L0 unary SVR → SVR. input=rs1[4:0]; result=OPERAND3[4:0] or in-place."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    input_reg = rs1 & 0x1F
    op3_raw = operand3(npu, 0xFFFFFFFF)
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub, view, rs2 & 0x3)   # rs2[1:0] = exp/ln mode
    _l0_block_view(npu, nest, spu, result_reg)[...] = result
    return 0


# ----- CLAMP / ACCUM / ARANGE (0x1F, L1) -------------------------------------
def _clamp_v(npu, proc, inst, sub: Clamp) -> int:
    """L1 clamp.min / clamp.max / accum / arange (scalar args in rs2)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a, _addr_b, addr_r = _l1_addrs(npu, nest, spu)
    view = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    result = _CLAMP_FN[sub](view, rs2, vsz)
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = result
    return 0


# ----- LOGIC (0x1F, L0) ------------------------------------------------------
def _logic_i(npu, proc, inst, sub: Logic) -> int:
    """L0 bitwise AND/OR/NOT/SHIFT on FP16 raw bits (vendor exec_bitwise_imm).

    a=rs1[4:0]; AND/OR read b=rs2[4:0]; NOT/SHIFT ignore it (SHIFT uses raw rs2).
    Result reg from OPERAND3[4:0].
    """
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    va = _l0_block_view_uint(npu, nest, spu, rs1 & 0x1F)
    vb = _l0_block_view_uint(npu, nest, spu, rs2 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _l0_block_view_uint(npu, nest, spu, r_reg)[...] = _LOGIC_FN[sub](va, vb, rs2)
    return 0


# =============================================================================
# 5. Handlers — each names its op directly; the registry routes (funct7, funct3).
# =============================================================================
# ----- Vector arith (funct7=0x18): VV funct3=0..3, II funct3=4..7 ------------
@inst_register.custom0(name='add.vv', funct7=0b0011000, funct3=0b000)
def add_vv(npu, proc, inst, cxt) -> int:
    return _arith_vv(npu, proc, inst, Arith.ADD)


@inst_register.custom0(name='sub.vv', funct7=0b0011000, funct3=0b001)
def sub_vv(npu, proc, inst, cxt) -> int:
    return _arith_vv(npu, proc, inst, Arith.SUB)


@inst_register.custom0(name='mul.vv', funct7=0b0011000, funct3=0b010)
def mul_vv(npu, proc, inst, cxt) -> int:
    return _arith_vv(npu, proc, inst, Arith.MUL)


@inst_register.custom0(name='div.vv', funct7=0b0011000, funct3=0b011)
def div_vv(npu, proc, inst, cxt) -> int:
    return _arith_vv(npu, proc, inst, Arith.DIV)


@inst_register.custom0(name='fmadd.vvv', funct7=0b0011001, funct3=0b000)
def fmadd_vvv(npu, proc, inst, cxt) -> int:
    """L1 A*B + C → R, FP32 internal (vendor GTX_VEC_FMADD, gtx_npu_vec.cc:96
    ``rd16(addr_a)*rd16(addr_b)+rd16(addr_c)``). The addend is the C bank
    (SPM_ADDRC), NOT the R bank — __set_spm_addr's 2nd arg is ADDR_C."""
    nest, spu, _rs1, _rs2, vsz = _prep(npu, proc, inst)
    addr_a, addr_b, addr_r = _l1_addrs(npu, nest, spu)
    addr_c = npu.lspr[nest][spu].get(LSPR['SPM_ADDRC'].address, 0)
    va = _as_fp32(_l1_view_addr(npu, nest, spu, addr_a, vsz))
    vb = _as_fp32(_l1_view_addr(npu, nest, spu, addr_b, vsz))
    vc = _as_fp32(_l1_view_addr(npu, nest, spu, addr_c, vsz))
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = (va * vb + vc).astype(MX_IO_DTYPE)
    return 0

# ----- Vector reductions (funct7=0x1A) ---------------------------------------
# ISA v2.0.0d moved sum/dot off the matrix unit (the old mm.o/mm.v reduction
# path) onto the vector unit. result[31:0] = reduce(A[, B]) + accumulated_data,
# seeded by OPERAND2[31:0] (FP32) and written to the SVR slot staged in OPERAND3
# (same result-SVR convention as scalar max.vs/min.vs). FP32 internal accumulate.
def _reduce_write(npu, nest, spu, addr_r, value: np.ndarray) -> None:
    """Write a single reduction scalar to the OPERAND3-staged SVR slot, element
    0, rest of the 32-byte block zeroed (mirrors scalar.max_vs)."""
    dst = _l0_block_view(npu, nest, spu, operand3(npu, addr_r) & 0x1F)
    dst.fill(0)
    dst[0] = value.astype(MX_IO_DTYPE)


@inst_register.custom0(name='sum.v', funct7=0b0011010, funct3=0b000)
def sum_v(npu, proc, inst, cxt) -> int:
    """L1 sum(A) + accumulated_data -> result[31:0] at the OPERAND3 SVR slot."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a, _addr_b, addr_r = _l1_addrs(npu, nest, spu)
    va = _as_fp32(_l1_view_addr(npu, nest, spu, addr_a, vsz)).reshape(-1)
    seed = np.float32(_fp32_low32(rs2))
    _reduce_write(npu, nest, spu, addr_r, va.sum(dtype=np.float32) + seed)
    return 0


@inst_register.custom0(name='dot.vv', funct7=0b0011010, funct3=0b001)
def dot_vv(npu, proc, inst, cxt) -> int:
    """L1 dot(A, B) + accumulated_data -> result[31:0] at the OPERAND3 SVR slot.
    B from SPM_ADDRB."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a, addr_b, addr_r = _l1_addrs(npu, nest, spu)
    va = _as_fp32(_l1_view_addr(npu, nest, spu, addr_a, vsz)).reshape(-1)
    vb = _as_fp32(_l1_view_addr(npu, nest, spu, addr_b, vsz)).reshape(-1)
    seed = np.float32(_fp32_low32(rs2))
    _reduce_write(npu, nest, spu, addr_r,
                  np.float32(np.dot(va, vb)) + seed)
    return 0


@inst_register.custom0(name='sum.i', funct7=0b0011010, funct3=0b100)
def sum_i(npu, proc, inst, cxt) -> int:
    """L0 sum(SVR_A) + SVR_ACC[0] -> result[31:0] at the OPERAND3 SVR slot.
    A=rs1[4:0], ACC=rs2[9:5]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    va = _as_fp32(_l0_block_view(npu, nest, spu, rs1 & 0x1F)).reshape(-1)
    acc = _as_fp32(_l0_block_view(npu, nest, spu, (rs2 >> 5) & 0x1F)).reshape(-1)
    r_reg = operand3(npu, inst.rd) & 0x1F
    dst = _l0_block_view(npu, nest, spu, r_reg)
    dst.fill(0)
    dst[0] = (va.sum(dtype=np.float32) + np.float32(acc[0])).astype(MX_IO_DTYPE)
    return 0


@inst_register.custom0(name='dot.ii', funct7=0b0011010, funct3=0b101)
def dot_ii(npu, proc, inst, cxt) -> int:
    """L0 dot(SVR_A, SVR_B) + SVR_ACC[0] -> result[31:0] at the OPERAND3 SVR
    slot. A=rs1[4:0], B=rs2[4:0], ACC=rs2[9:5]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    va = _as_fp32(_l0_block_view(npu, nest, spu, rs1 & 0x1F)).reshape(-1)
    vb = _as_fp32(_l0_block_view(npu, nest, spu, rs2 & 0x1F)).reshape(-1)
    acc = _as_fp32(_l0_block_view(npu, nest, spu, (rs2 >> 5) & 0x1F)).reshape(-1)
    r_reg = operand3(npu, inst.rd) & 0x1F
    dst = _l0_block_view(npu, nest, spu, r_reg)
    dst.fill(0)
    dst[0] = (np.float32(np.dot(va, vb)) + np.float32(acc[0])).astype(MX_IO_DTYPE)
    return 0

@inst_register.custom0(name='sqrt.v', funct7=0b0011100, funct3=0b000)
def sqrt_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_MATH_V, Math.SQRT)


@inst_register.custom0(name='exp.v', funct7=0b0011100, funct3=0b001)
def exp_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_MATH_V, Math.EXP)


@inst_register.custom0(name='ln.v', funct7=0b0011100, funct3=0b010)
def ln_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_MATH_V, Math.LN)


@inst_register.custom0(name='abs.v', funct7=0b0011101, funct3=0b000)
def abs_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_SIGN_V, Sign.ABS)


@inst_register.custom0(name='neg.v', funct7=0b0011101, funct3=0b001)
def neg_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_SIGN_V, Sign.NEG)


@inst_register.custom0(name='sign.v', funct7=0b0011101, funct3=0b010)
def sign_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_SIGN_V, Sign.SIGN)


@inst_register.custom0(name='step.v', funct7=0b0011101, funct3=0b011)
def step_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_SIGN_V, Sign.STEP)


@inst_register.custom0(name='ceil.v', funct7=0b0011110, funct3=0b000)
def ceil_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_ROUND_V, Round.CEIL)


@inst_register.custom0(name='trunc.v', funct7=0b0011110, funct3=0b001)
def trunc_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_ROUND_V, Round.TRUNC)


@inst_register.custom0(name='floor.v', funct7=0b0011110, funct3=0b010)
def floor_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_ROUND_V, Round.FLOOR)


@inst_register.custom0(name='rne.v', funct7=0b0011110, funct3=0b011)
def rne_v(npu, proc, inst, cxt) -> int:
    return _unary_v(npu, proc, inst, F7_ROUND_V, Round.RNE)


@inst_register.custom0(name='clamp.min', funct7=0b0011111, funct3=0b000)
def clamp_min(npu, proc, inst, cxt) -> int:
    return _clamp_v(npu, proc, inst, Clamp.MIN)


@inst_register.custom0(name='clamp.max', funct7=0b0011111, funct3=0b001)
def clamp_max(npu, proc, inst, cxt) -> int:
    return _clamp_v(npu, proc, inst, Clamp.MAX)


@inst_register.custom0(name='accum', funct7=0b0011111, funct3=0b010)
def accum_v(npu, proc, inst, cxt) -> int:
    return _clamp_v(npu, proc, inst, Clamp.ACCUM)


@inst_register.custom0(name='arange', funct7=0b0011111, funct3=0b011)
def arange_v(npu, proc, inst, cxt) -> int:
    return _clamp_v(npu, proc, inst, Clamp.ARANGE)


# ----- L0 SVR (II / I) variants ----------------------------------------------
@inst_register.custom0(name='add.ii', funct7=0b0011000, funct3=0b100)
def add_ii(npu, proc, inst, cxt) -> int:
    return _arith_ii(npu, proc, inst, Arith.ADD)


@inst_register.custom0(name='sub.ii', funct7=0b0011000, funct3=0b101)
def sub_ii(npu, proc, inst, cxt) -> int:
    return _arith_ii(npu, proc, inst, Arith.SUB)


@inst_register.custom0(name='mul.ii', funct7=0b0011000, funct3=0b110)
def mul_ii(npu, proc, inst, cxt) -> int:
    return _arith_ii(npu, proc, inst, Arith.MUL)


@inst_register.custom0(name='div.ii', funct7=0b0011000, funct3=0b111)
def div_ii(npu, proc, inst, cxt) -> int:
    return _arith_ii(npu, proc, inst, Arith.DIV)


@inst_register.custom0(name='fmadd.iii', funct7=0b0011001, funct3=0b100)
def fmadd_iii(npu, proc, inst, cxt) -> int:
    return _fmadd_iii(npu, proc, inst)


@inst_register.custom0(name='sqrt.i', funct7=0b0011100, funct3=0b100)
def sqrt_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_MATH_V, Math.SQRT)


@inst_register.custom0(name='exp.i', funct7=0b0011100, funct3=0b101)
def exp_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_MATH_V, Math.EXP)


@inst_register.custom0(name='ln.i', funct7=0b0011100, funct3=0b110)
def ln_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_MATH_V, Math.LN)


@inst_register.custom0(name='abs.i', funct7=0b0011101, funct3=0b100)
def abs_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_SIGN_V, Sign.ABS)


@inst_register.custom0(name='neg.i', funct7=0b0011101, funct3=0b101)
def neg_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_SIGN_V, Sign.NEG)


@inst_register.custom0(name='sign.i', funct7=0b0011101, funct3=0b110)
def sign_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_SIGN_V, Sign.SIGN)


@inst_register.custom0(name='step.i', funct7=0b0011101, funct3=0b111)
def step_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_SIGN_V, Sign.STEP)


@inst_register.custom0(name='ceil.i', funct7=0b0011110, funct3=0b100)
def ceil_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_ROUND_V, Round.CEIL)


@inst_register.custom0(name='trunc.i', funct7=0b0011110, funct3=0b101)
def trunc_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_ROUND_V, Round.TRUNC)


@inst_register.custom0(name='floor.i', funct7=0b0011110, funct3=0b110)
def floor_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_ROUND_V, Round.FLOOR)


@inst_register.custom0(name='rne.i', funct7=0b0011110, funct3=0b111)
def rne_i(npu, proc, inst, cxt) -> int:
    return _unary_i(npu, proc, inst, F7_ROUND_V, Round.RNE)


@inst_register.custom0(name='and.ii', funct7=0b0011111, funct3=0b100)
def and_ii(npu, proc, inst, cxt) -> int:
    return _logic_i(npu, proc, inst, Logic.AND)


@inst_register.custom0(name='or.ii', funct7=0b0011111, funct3=0b101)
def or_ii(npu, proc, inst, cxt) -> int:
    return _logic_i(npu, proc, inst, Logic.OR)


@inst_register.custom0(name='not.i', funct7=0b0011111, funct3=0b110)
def not_i(npu, proc, inst, cxt) -> int:
    return _logic_i(npu, proc, inst, Logic.NOT)


@inst_register.custom0(name='shift.i', funct7=0b0011111, funct3=0b111)
def shift_i(npu, proc, inst, cxt) -> int:
    return _logic_i(npu, proc, inst, Logic.SHIFT)
