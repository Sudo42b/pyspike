"""
add.vv	4'b0011	3'b000	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 add (A+B)	-
sub.vv	4'b0011	3'b000	rsvd	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 sub (A-B)	-
mul.vv	4'b0011	3'b000	rsvd	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 mult (A*B)	-
div.vv	4'b0011	3'b000	rsvd	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 div (A/B)	-
fmadd.vvv	4'b0011	3'b001	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 fmadd (A*B+C)	-
sqrt.v	4'b0011	3'b100	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 sqrt(A)	-
exp.v	4'b0011	3'b100	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	mode[1:0]	N/A	N/A	N/A	N/A	fp16 exp(A)	00: exp(A) / 01 :2^(A) / 10: 3^(A) / 11: 5^(A)
ln.v	4'b0011	3'b100	rsvd	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	mode[1:0]	N/A	N/A	N/A	N/A	fp16 ln(A)	00: ln(A) / 01: log2(A) / 10: log10(A)
abs.v	4'b0011	3'b101	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 abs(A)	-
neg.v	4'b0011	3'b101	rsvd	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 neg(A)	-
sign.v	4'b0011	3'b101	rsvd	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 sign(A)	-
step.v	4'b0011	3'b101	rsvd	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 step(A)	-
ceil.v	4'b0011	3'b110	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 ceil(A)	-
trunc.v	4'b0011	3'b110	rsvd	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 trunc(A)	-
floor.v	4'b0011	3'b110	rsvd	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 floor(A)	-
rne.v	4'b0011	3'b110	rsvd	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 rne(A)	-
clamp.min	4'b0011	3'b111	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 minimum_clamp(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
clamp.max	4'b0011	3'b111	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 maximum_clamp(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
accum	4'b0011	3'b111	rsvd	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 accumulate_sum(A)	-
arange	4'b0011	3'b111	gpr	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	start_value[15:0], step[31:16]	N/A	N/A	N/A	N/A	fp16 arange(A)	-
add.ii	4'b0011	3'b000	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_SVR_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 add (A+B) imm	-
sub.ii	4'b0011	3'b000	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_SVR_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 sub (A-B) imm	-
mul.ii	4'b0011	3'b000	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_SVR_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 mult (A*B) imm	-
div.ii	4'b0011	3'b000	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_SVR_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 div (A/B) imm	-
fmadd.iii	4'b0011	3'b001	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_SVR_addr_B[4:0], src_SVR_addr_C[9:5]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 fmadd (A*B+C) imm	-
sqrt.i	4'b0011	3'b100	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 sqrt(A) imm	-
exp.i	4'b0011	3'b100	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	mode[1:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 exp(A) imm	00: exp(A) / 01 :2^(A) / 10: 3^(A) / 11: 5^(A)
ln.i	4'b0011	3'b100	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	mode[1:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 ln(A) imm	00: ln(A) / 01: log2(A) / 10: log10(A)
abs.i	4'b0011	3'b101	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 abs(A) imm	-
neg.i	4'b0011	3'b101	rsvd	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 neg(A) imm	-
sign.i	4'b0011	3'b101	rsvd	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 sign(A) imm	-
step.i	4'b0011	3'b101	rsvd	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 step(A) imm	-
ceil.i	4'b0011	3'b110	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 ceil(A) imm	-
trunc.i	4'b0011	3'b110	rsvd	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 trunc(A) imm	-
floor.i	4'b0011	3'b110	rsvd	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 floor(A) imm	-
rne.i	4'b0011	3'b110	rsvd	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 rne(A) imm	-
and.ii	4'b0011	3'b111	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_svr_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	bit-wise and(A, B) imm	-
or.ii	4'b0011	3'b111	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	src_svr_addr_B[4:0]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	bit-wise or(A, B) imm	-
not.i	4'b0011	3'b111	rsvd	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	bit-wise not(A) imm	-
shift.i	4'b0011	3'b111	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	shift_num[3:0], shift_mode[4]	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	bit-wise shift(A)	shift_mode[4] 0: >> / 1:<<
"""
"""VEC op @handler entries + vector kernels + firmware dispatcher.

Single-file consolidation of the former three-layer split
(``vec_core`` element-wise kernels + ``vec_engine`` decode/dispatch +
``ops/vec`` @handler decorators). Layout:

  1. SASMD / DOT / SUM / clamp / accum / arange kernels.
  2. Helpers: FP16 bit-pattern decoders, L1 / L0 views.
  3. Unary helper (``_apply_unary``) shared by MATH / SIGN / ROUND.
  4. Sub-dispatchers for SASMD / arith L0II / unary L0.
  5. ``vec_op`` / ``firmware_vec_op`` main dispatcher.
  6. @handler entries -- one per (funct7, funct3) tuple matching
     ``gtx_npu_disasm.inc:67-142``.

Per RESEARCH Pitfall 7: ``vec_size = (rs1 & 0xFFFF) or 0x10000``
(HW convention: 0 -> 65536).
"""
from __future__ import annotations

import torch

from ...inst_handler import inst_register
from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM

from ....csr import GSPR, LSPR


# =========================================================================
# Phase 5: VEC funct7 constants
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:67-142
# =========================================================================
F7_VEC_SASMD: int = 0x10        # SASMD scalar arith (add/sub/mul/div x VS/IS)
F7_VEC_FMADD: int = 0x11        # fmadd_vss / fmadd_iss (P5 stub; lower priority)
F7_VEC_MINMAX: int = 0x13       # max_vs / min_vs / max_is / min_is (disasm.inc:80-84)
F7_VEC_ARITH: int = 0x18        # SASMD vector arith (add/sub/mul/div x VV/II)
# AUTHORITATIVE CORRECTION (Plan 05-02 deviation Rule 1):
# Plan 01 seeded F7_VEC_DOT_SUM=0x13 from a draft note; vendor
# `gtx_npu.h:308` defines F7_DOT_SUM = 0b0011010 = 0x1A. funct7=0x13
# is actually MIN/MAX scalar arith (disasm.inc:80-84). DOT/SUM lives at
# 0x1A with `dot_vvs` at funct3=0 and `sum_vs` (vsum) at funct3=1
# (disasm.inc:101-104; firmware_vec_op.cc:632-637).
F7_VEC_DOT_SUM: int = 0x1A      # dot funct3=0, vsum funct3=1 (gtx_npu_vec.cc:632-637)
F7_VEC_FMADD_VV: int = 0x19     # vector fmadd (P5 stub; lower priority)
F7_VEC_MATH: int = 0x1C         # sqrt/exp/log
F7_VEC_SIGN: int = 0x1D         # abs/neg/sgn/step
F7_VEC_ROUND: int = 0x1E        # ceil/trunc/floor/rne
F7_VEC_CLAMP: int = 0x1F        # clamp_min_v/clamp_max_v/accum_v/arange_v + bitwise

# ============================================================================
# Vector sub-opcodes (passed via GSPR_OPCODE)
# ============================================================================
VEC_ADD:int   = 0
VEC_SUB:int   = 1
VEC_MUL:int   = 2
VEC_DIV:int   = 3
VEC_FMADD:int = 4
VEC_VSUM:int  = 5
VEC_VEXP:int  = 6
VEC_VSQRT:int = 7
VEC_VLN:int   = 8
VEC_VABS:int  = 9
VEC_VNEG:int  = 10
VEC_MAX:int   = 11
VEC_MIN:int   = 12
VEC_SIGN:int      = 13
VEC_STEP:int      = 14
VEC_CEIL:int      = 15
VEC_TRUNC:int     = 16
VEC_FLOOR:int     = 17
VEC_RNE:int       = 18
VEC_ACCUM:int     = 19
VEC_CLAMP_MAX:int = 20
VEC_CLAMP_MIN:int = 21
VEC_ARANGE:int    = 22
VEC_DOT:int       = 23

#  ── L0-based _IMM operation groups ─────────────────────────
F7_SCALAR_IMM:int = 0b1010100  # 0x54 — scalar _IS arith
F7_VECTOR_IMM:int = 0b1011000  # 0x58 — vector _II arith
F7_VFUNC_IMM:int = 0b1011010   # 0x5A — L0 math functions
F7_BITWISE_IMM:int = 0b1011011 # 0x5B — L0 bitwise



# Vector Calculations
F7_VEC_ARITH:int = 0b0011000   # ADD/SUB/MUL/DIV vector add.vv, sub.vv, mul.vv, div.vv, add.ii, sub.ii, mul.ii, div.ii
F7_FMADD_V:int = 0b0011001     # Vector FMADD fmadd.vvv, fmadd.iii
F7_MATH_V:int = 0b0011100      # SQRT/EXP/LN, sqrt.v, exp.v, ln.v
F7_SIGN_V:int = 0b0011101      # ABS/NEG/SIGN/STEP abs.v, neg.v, sign.v, step.v
F7_ROUND_V:int = 0b0011110     # CEIL/TRUNC/FLOOR/RNE, ceil.v, trunc.v, floor.v, rne.v, ceil.i, trunc.i, floor.i, rne.i
F7_CLAMP_V:int = 0b0011111     # CLAMP/ACCUM/ARANGE/LOGIC, clamp.min, clamp.max, accum, arange, and.ii, or.ii, not.i, shift.i



# =============================================================================
# 1. Vector kernels (FP32 internal, FP16 output)
# =============================================================================
def _as_fp32(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32)
    return torch.as_tensor(a, dtype=torch.float32)


def sasmd(a, b, op: int) -> torch.Tensor:
    """SASMD element-wise FP32 internal, FP16 output. ``b`` scalar or array."""
    a_f32 = _as_fp32(a)
    if isinstance(b, torch.Tensor) and b.dim() > 0:
        b_f32 = b.to(torch.float32)
    elif hasattr(b, 'shape') and getattr(b, 'shape', ()):
        b_f32 = torch.as_tensor(b, dtype=torch.float32)
    else:
        b_f32 = torch.full_like(a_f32, float(b))
    if op == VEC_ADD:
        out = a_f32 + b_f32
    elif op == VEC_SUB:
        out = a_f32 - b_f32
    elif op == VEC_MUL:
        out = a_f32 * b_f32
    elif op == VEC_DIV:
        # Vendor convention (gtx_npu_vec.cc:333): div-by-zero -> 0.0.
        safe_b = torch.where(b_f32 == 0.0, torch.ones_like(b_f32), b_f32)
        raw = a_f32 / safe_b
        out = torch.where(b_f32 == 0.0, torch.zeros_like(raw), raw)
    else:
        raise ValueError(f"unknown SASMD op {op}")
    return out.to(torch.float16)


def dot(a, b) -> torch.Tensor:
    """FP16 dot product — FP32 reduce on DEVICE, FP16 output."""
    a_f32 = _as_fp32(a).reshape(-1)
    b_f32 = _as_fp32(b).reshape(-1)
    if a_f32.shape != b_f32.shape:
        raise ValueError(f"shape mismatch: {a_f32.shape} vs {b_f32.shape}")
    return torch.dot(a_f32, b_f32).to(torch.float16)


def vsum(view) -> torch.Tensor:
    """FP16 vector sum — FP32 reduce on DEVICE, FP16 output."""
    return torch.sum(_as_fp32(view).reshape(-1)).to(torch.float16)


def clamp_min(a, scalar) -> torch.Tensor:
    """``out[i] = max(a[i], scalar)``."""
    return torch.clamp(_as_fp32(a), min=float(scalar)).to(torch.float16)


def clamp_max(a, scalar) -> torch.Tensor:
    """``out[i] = min(a[i], scalar)``."""
    return torch.clamp(_as_fp32(a), max=float(scalar)).to(torch.float16)


def accum(a) -> torch.Tensor:
    """Prefix sum: FP32 accumulator across whole vec, per-element FP16 cast.

    ``torch.cumsum`` is the left-to-right vectorised form of the Python
    accumulator loop — same numerical order, no per-element kernel launch.
    """
    return torch.cumsum(_as_fp32(a).reshape(-1), dim=0).to(torch.float16)


def arange(n: int, start, step) -> torch.Tensor:
    """``out[i] = start + i*step`` (FP32 internal)."""
    from ....config_params import DEVICE
    idx = torch.arange(int(n), dtype=torch.float32, device=DEVICE)
    return (float(start) + idx * float(step)).to(torch.float16)


# =============================================================================
# 2. Helpers
# =============================================================================
def _fp16_low16(packed: int) -> torch.Tensor:
    """Decode bits[15:0] of an int as FP16 (LE bit-pattern), 0-d tensor."""
    u16 = torch.tensor([packed & 0xFFFF], dtype=torch.uint16)
    return u16.view(torch.float16)[0]


def _fp16_high16(packed: int) -> torch.Tensor:
    """Decode bits[31:16] of an int as FP16 (LE bit-pattern), 0-d tensor."""
    u16 = torch.tensor([(packed >> 16) & 0xFFFF], dtype=torch.uint16)
    return u16.view(torch.float16)[0]


def _l1_view_addr(npu, nest: int, spu: int, addr_byte: int,
                   length: int) -> torch.Tensor:
    """Return an FP16 view of ``L1[addr:addr + length*2]`` (no copy)."""
    l1_f16 = npu.mem.l1_f16(nest, spu)
    off = addr_byte // 2
    return l1_f16[off:off + length]


def _l0_block_view(npu, nest: int, spu: int, reg: int) -> torch.Tensor:
    """Return an FP16 view of ``L0[(reg & 0x1F)*32 .. +32]``; 16 FP16."""
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    return l0.view(torch.float16)[off // 2:off // 2 + 16]


# =============================================================================
# 3. Unary apply (MATH / SIGN / ROUND family bodies)
# =============================================================================
def _apply_unary(funct7: int, sub_op: int, view: torch.Tensor) -> torch.Tensor:
    """Element-wise unary kernels for funct7 0x1C/0x1D/0x1E.

    SIGN / ROUND families operate on the FP16 view directly — sign bit
    or integer rounding doesn't benefit from FP32 promotion and the
    saved fp16↔fp32 conversion kernels matter when this is invoked
    once per (NEST, SPU) tile. MATH (sqrt / exp / log) keeps the FP32
    accumulator path for precision.
    """
    if funct7 == 0x1D:   # SIGN: abs / neg / sign / step
        if sub_op == 0:
            return torch.abs(view)
        if sub_op == 1:
            return -view
        if sub_op == 2:
            return torch.sign(view)
        if sub_op == 3:
            return (view > 0.0).to(torch.float16)
    if funct7 == 0x1E:   # ROUND
        if sub_op == 0:
            return torch.ceil(view)
        if sub_op == 1:
            return torch.trunc(view)
        if sub_op == 2:
            return torch.floor(view)
        if sub_op == 3:
            return torch.round(view)
    if funct7 == 0x1C:   # MATH: sqrt / exp / log (FP32 accumulator)
        f32 = view.to(torch.float32)
        if sub_op == 0:
            return torch.sqrt(f32).to(torch.float16)
        if sub_op == 1:
            return torch.exp(f32).to(torch.float16)
        if sub_op == 2:
            tiny = torch.finfo(torch.float32).tiny
            return torch.where(f32 > 0.0,
                                torch.log(f32.clamp(min=tiny)),
                                torch.zeros_like(f32)).to(torch.float16)
    return view.clone()


# =============================================================================
# 4. Sub-dispatchers
# =============================================================================
def sasmd(npu, nest: int, spu: int, funct3: int,
                     rs1: int, rs2: int, insn, vec_size: int) -> int:
    op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
    sub = funct3 & 3
    assert sub not in op_map or funct3 & 4, "SASMD sub-op must be 0-3 with bit 2 set"

    scalar = _fp16_low16(rs2)
    if not (funct3 & 4):
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = sasmd(view_a, scalar, op=op_map[sub])
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    result = sasmd(view_a, scalar, op=op_map[sub])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def arith_l0_ii(npu, nest: int, spu: int, sub_op: int,
                            rs1: int, rs2: int, insn) -> int:
    op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
    if sub_op not in op_map:
        return 0
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    view_b = _l0_block_view(npu, nest, spu, b_reg)
    result = sasmd(view_a, view_b, op=op_map[sub_op])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def unary_l0(npu, nest: int, spu: int, funct7: int, sub_op: int,
                        rs1: int, insn) -> int:
    input_reg = rs1 & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, 0xFFFFFFFF))
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub_op, view)
    _l0_block_view(npu, nest, spu, result_reg).copy_(result)
    return 0


# =============================================================================
# 5. vec_op / firmware_vec_op
# =============================================================================
def vec_op(npu, proc, insn) -> int:
    """Direct port of ``gtx_npu_vec.cc:572-754``."""
    rs1 = int(proc.state.XPR[insn.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000

    funct7 = insn.funct
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    assert nest < NEST_NUM, f"NEST id {nest} >= NEST_NUM={NEST_NUM}"
    assert spu < SPU_NUM, f"SPU id {spu} >= SPU_NUM={SPU_NUM}"

    rs2 = int(proc.state.XPR[insn.rs2])
    npu.gspr[GSPR['GSPR_OPERAND2'].address] = rs2

    if funct7 == F7_VEC_SASMD:
        return sasmd(npu, nest, spu, funct3, rs1, rs2, insn, vec_size)

    if funct7 == F7_VEC_ARITH and (funct3 & 4):
        return arith_l0_ii(npu, nest, spu, funct3 & 3, rs1, rs2, insn)
    if funct7 in (0x1C, 0x1D, 0x1E) and (funct3 & 4):
        return unary_l0(npu, nest, spu, funct7, funct3 & 3, rs1, insn)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR['SPM_ADDRB'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    if funct7 == F7_VEC_ARITH:
        op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
        if (funct3 & 3) in op_map:
            view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            result = sasmd(view_a, view_b, op=op_map[funct3 & 3])
            _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
            return 0

    if funct7 == F7_VEC_DOT_SUM:
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if (funct3 & 3) == 0:
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            scalar = dot(view_a, view_b)
        else:
            scalar = vsum(view_a)
        _l1_view_addr(npu, nest, spu, addr_r, 1)[0] = scalar
        # Reinterpret the 0-d FP16 scalar as 2 bytes (little-endian) and
        # blit straight into L0 — no bit-masking, no Python-side raw int.
        scalar_bytes = scalar.to(torch.float16).reshape(1).contiguous().view(torch.uint8)
        l0 = npu.mem.l0_byte(nest, spu)
        l0[0:2] = scalar_bytes
        return 0

    if funct7 == F7_VEC_CLAMP:
        sub = funct3 & 3
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        #! sub op 0, 1, 2 가 무엇인지 상수로 표현하는게 좋을듯.
        if sub == 0:
            scalar = _fp16_low16(rs2)
            result = clamp_min(view_a, scalar)
        elif sub == 1:
            scalar = _fp16_low16(rs2)
            result = clamp_max(view_a, scalar)
        elif sub == 2:
            result = accum(view_a)
        else:
            start = _fp16_low16(rs2)
            step = _fp16_high16(rs2)
            result = arange(vec_size, start, step)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    if funct7 in (0x1C, 0x1D, 0x1E):
        view = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = _apply_unary(funct7, funct3 & 3, view)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    return 0



# =============================================================================
# 6. @handler entries
# =============================================================================
# ----- SASMD scalar arith (funct7=0x10): VS funct3=0..3, IS funct3=4..7 ------

@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=0,
         mnemonic='add.vs', mask_funct3=True)
def add_vs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=1,
         mnemonic='sub.vs', mask_funct3=True)
def sub_vs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=2,
         mnemonic='mul.vs', mask_funct3=True)
def mul_vs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=3,
         mnemonic='div.vs', mask_funct3=True)
def div_vs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=4,
         mnemonic='add.is', mask_funct3=True)
def add_is(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=5,
         mnemonic='sub.is', mask_funct3=True)
def sub_is(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=6,
         mnemonic='mul.is', mask_funct3=True)
def mul_is(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_SASMD, funct3=7,
         mnemonic='div.is', mask_funct3=True)
def div_is(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


# ----- VSUM / DOT (funct7=0x1A): DOT funct3=0, SUM funct3=1 ------------------

@inst_register.custom0(kind='custom0', funct7=F7_VEC_DOT_SUM, funct3=0,
         mnemonic='dot.vvs', mask_funct3=True)
def dot_vvs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_DOT_SUM, funct3=1,
         mnemonic='sum.vs', mask_funct3=True)
def sum_vs(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


# ----- SASMD vector arith (funct7=0x18): VV funct3=0..3, II funct3=4..7 ------

@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=0,
         mnemonic='add.vv', mask_funct3=True)
def add_vv(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=1,
         mnemonic='sub.vv', mask_funct3=True)
def sub_vv(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=2,
         mnemonic='mul.vv', mask_funct3=True)
def mul_vv(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=3,
         mnemonic='div.vv', mask_funct3=True)
def div_vv(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=4,
         mnemonic='add.ii', mask_funct3=True)
def add_ii(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=5,
         mnemonic='sub.ii', mask_funct3=True)
def sub_ii(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=6,
         mnemonic='mul.ii', mask_funct3=True)
def mul_ii(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_ARITH, funct3=7,
         mnemonic='div.ii', mask_funct3=True)
def div_ii(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


# ----- CLAMP family (funct7=0x1F): funct3=0..3 -------------------------------

@inst_register.custom0(kind='custom0', funct7=F7_VEC_CLAMP, funct3=0,
         mnemonic='clamp_min.v', mask_funct3=True)
def clamp_min_v(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_CLAMP, funct3=1,
         mnemonic='clamp_max.v', mask_funct3=True)
def clamp_max_v(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_CLAMP, funct3=2,
         mnemonic='accum.v', mask_funct3=True)
def accum_v(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


@inst_register.custom0(kind='custom0', funct7=F7_VEC_CLAMP, funct3=3,
         mnemonic='arange.v', mask_funct3=True)
def arange_v(npu, proc, insn, xs1, xs2):
    return vec_op(npu, proc, insn)


# ----- MATH / SIGN / ROUND (funct7 = 0x1C / 0x1D / 0x1E) ---------------------
# Sub-op selected by funct3 (& 3); P8 NEG fix preserved (one mnemonic per
# funct3, all routed through vec_op → _apply_unary).
#
# The stacked @handler decorators below register THIS function under 11
# different (funct7, funct3, mnemonic) tuples — abs_v, sqrt_v, exp_v,
# log_v, neg_v, sign_v, step_v, ceil_v, trunc_v, floor_v, rne_v. The
# dispatch table calls it whenever any of those mnemonics decodes; ABS
# firmware → ``abs_v`` is one of those entry points (ABS regression
# proves the wiring). Renamed from ``unary_family`` to
# ``vec_unary`` for clarity.
@inst_register.custom0(kind='custom0', funct7=F7_VEC_MATH, funct3=0,
         mnemonic='sqrt.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_MATH, funct3=1,
         mnemonic='exp.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_MATH, funct3=2,
         mnemonic='log.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_SIGN, funct3=0,
         mnemonic='abs.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_SIGN, funct3=1,
         mnemonic='neg.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_SIGN, funct3=2,
         mnemonic='sign.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_SIGN, funct3=3,
         mnemonic='step.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_ROUND, funct3=0,
         mnemonic='ceil.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_ROUND, funct3=1,
         mnemonic='trunc.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_ROUND, funct3=2,
         mnemonic='floor.v', mask_funct3=True)
@inst_register.custom0(kind='custom0', funct7=F7_VEC_ROUND, funct3=3,
         mnemonic='rne.v', mask_funct3=True)
def vec_unary(npu, proc, insn, xs1, xs2):
    """Element-wise unary entry (MATH/SIGN/ROUND).

    Sub-op decoded from (funct7, funct3) inside vec_op → _apply_unary.
    """
    return vec_op(npu, proc, insn)
