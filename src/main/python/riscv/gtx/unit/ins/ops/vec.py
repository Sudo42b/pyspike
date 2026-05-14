"""VEC op @handler entries + vector kernels + firmware dispatcher.

Single-file consolidation of the former three-layer split
(``vec_core`` element-wise kernels + ``vec_engine`` decode/dispatch +
``ops/vec`` @handler decorators). Layout:

  1. SASMD / DOT / SUM / clamp / accum / arange kernels.
  2. Helpers: FP16 bit-pattern decoders, L1 / L0 views.
  3. Unary helper (``_apply_unary``) shared by MATH / SIGN / ROUND.
  4. Sub-dispatchers for SASMD / arith L0II / unary L0.
  5. ``exec_vec_op`` / ``firmware_vec_op`` main dispatcher.
  6. @handler entries -- one per (funct7, funct3) tuple matching
     ``gtx_npu_disasm.inc:67-142``.

Per RESEARCH Pitfall 7: ``vec_size = (rs1 & 0xFFFF) or 0x10000``
(HW convention: 0 -> 65536).
"""
from __future__ import annotations

import torch

from ...._registry import handler
from ....config_params import GTX_L0_SIZE_BYTES, GTX_NEST_NUM, GTX_SPU_NUM
from ..encoding import (
    GTX_F7_VEC_ARITH, GTX_F7_VEC_CLAMP, GTX_F7_VEC_DOT_SUM,
    GTX_F7_VEC_MATH, GTX_F7_VEC_ROUND, GTX_F7_VEC_SASMD, GTX_F7_VEC_SIGN,
    GTX_VEC_ADD, GTX_VEC_DIV, GTX_VEC_MUL, GTX_VEC_SUB,
)
from ...csr import GSPR, LSPR


# =============================================================================
# 1. Vector kernels (FP32 internal, FP16 output)
# =============================================================================
def _as_fp32(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32)
    return torch.as_tensor(a, dtype=torch.float32)


def sasmd_kernel(a, b, op: int) -> torch.Tensor:
    """SASMD element-wise FP32 internal, FP16 output. ``b`` scalar or array."""
    a_f32 = _as_fp32(a)
    if isinstance(b, torch.Tensor) and b.dim() > 0:
        b_f32 = b.to(torch.float32)
    elif hasattr(b, 'shape') and getattr(b, 'shape', ()):
        b_f32 = torch.as_tensor(b, dtype=torch.float32)
    else:
        b_f32 = torch.full_like(a_f32, float(b))
    if op == GTX_VEC_ADD:
        out = a_f32 + b_f32
    elif op == GTX_VEC_SUB:
        out = a_f32 - b_f32
    elif op == GTX_VEC_MUL:
        out = a_f32 * b_f32
    elif op == GTX_VEC_DIV:
        # Vendor convention (gtx_npu_vec.cc:333): div-by-zero -> 0.0.
        safe_b = torch.where(b_f32 == 0.0, torch.ones_like(b_f32), b_f32)
        raw = a_f32 / safe_b
        out = torch.where(b_f32 == 0.0, torch.zeros_like(raw), raw)
    else:
        raise ValueError(f"unknown SASMD op {op}")
    return out.to(torch.float16)


def dot_kernel(a, b) -> torch.Tensor:
    """FP16 dot product — FP32 reduce on DEVICE, FP16 output."""
    a_f32 = _as_fp32(a).reshape(-1)
    b_f32 = _as_fp32(b).reshape(-1)
    if a_f32.shape != b_f32.shape:
        raise ValueError(f"shape mismatch: {a_f32.shape} vs {b_f32.shape}")
    return torch.dot(a_f32, b_f32).to(torch.float16)


def vsum_kernel(view) -> torch.Tensor:
    """FP16 vector sum — FP32 reduce on DEVICE, FP16 output."""
    return torch.sum(_as_fp32(view).reshape(-1)).to(torch.float16)


def clamp_min_kernel(a, scalar) -> torch.Tensor:
    """``out[i] = max(a[i], scalar)``."""
    return torch.clamp(_as_fp32(a), min=float(scalar)).to(torch.float16)


def clamp_max_kernel(a, scalar) -> torch.Tensor:
    """``out[i] = min(a[i], scalar)``."""
    return torch.clamp(_as_fp32(a), max=float(scalar)).to(torch.float16)


def accum_kernel(a) -> torch.Tensor:
    """Prefix sum: FP32 accumulator across whole vec, per-element FP16 cast.

    ``torch.cumsum`` is the left-to-right vectorised form of the Python
    accumulator loop — same numerical order, no per-element kernel launch.
    """
    return torch.cumsum(_as_fp32(a).reshape(-1), dim=0).to(torch.float16)


def arange_kernel(n: int, start, step) -> torch.Tensor:
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
    off = ((reg & 0x1F) * 32) % GTX_L0_SIZE_BYTES
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
def _dispatch_sasmd(npu, nest: int, spu: int, funct3: int,
                     rs1: int, rs2: int, insn, vec_size: int) -> int:
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    sub = funct3 & 3
    assert sub not in op_map or funct3 & 4, "SASMD sub-op must be 0-3 with bit 2 set"

    scalar = _fp16_low16(rs2)
    if not (funct3 & 4):
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = sasmd_kernel(view_a, scalar, op=op_map[sub])
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get("GSPR['GSPR_GTX_OPERAND3'].address", insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    result = sasmd_kernel(view_a, scalar, op=op_map[sub])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def _dispatch_arith_l0_ii(npu, nest: int, spu: int, sub_op: int,
                            rs1: int, rs2: int, insn) -> int:
    op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
    if sub_op not in op_map:
        return 0
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    r_reg = int(npu.gspr.get("GSPR['GSPR_GTX_OPERAND3'].address", insn.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    view_b = _l0_block_view(npu, nest, spu, b_reg)
    result = sasmd_kernel(view_a, view_b, op=op_map[sub_op])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def _dispatch_unary_l0(npu, nest: int, spu: int, funct7: int, sub_op: int,
                        rs1: int, insn) -> int:
    input_reg = rs1 & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0xFFFFFFFF))
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub_op, view)
    _l0_block_view(npu, nest, spu, result_reg).copy_(result)
    return 0


# =============================================================================
# 5. exec_vec_op / firmware_vec_op
# =============================================================================
def exec_vec_op(npu, proc, insn) -> int:
    """Direct port of ``gtx_npu_vec.cc:572-754``."""
    rs1 = int(proc.state.XPR[insn.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000

    funct7 = insn.funct
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    assert nest < GTX_NEST_NUM, f"NEST id {nest} >= GTX_NEST_NUM={GTX_NEST_NUM}"
    assert spu < GTX_SPU_NUM, f"SPU id {spu} >= GTX_SPU_NUM={GTX_SPU_NUM}"

    rs2 = int(proc.state.XPR[insn.rs2])
    npu.gspr[GSPR['GSPR_GTX_OPERAND2'].address] = rs2

    if funct7 == GTX_F7_VEC_SASMD:
        return _dispatch_sasmd(npu, nest, spu, funct3, rs1, rs2, insn, vec_size)

    if funct7 == GTX_F7_VEC_ARITH and (funct3 & 4):
        return _dispatch_arith_l0_ii(npu, nest, spu, funct3 & 3, rs1, rs2, insn)
    if funct7 in (0x1C, 0x1D, 0x1E) and (funct3 & 4):
        return _dispatch_unary_l0(npu, nest, spu, funct7, funct3 & 3, rs1, insn)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR['SPM_ADDRB'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    if funct7 == GTX_F7_VEC_ARITH:
        op_map = {0: GTX_VEC_ADD, 1: GTX_VEC_SUB, 2: GTX_VEC_MUL, 3: GTX_VEC_DIV}
        if (funct3 & 3) in op_map:
            view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            result = sasmd_kernel(view_a, view_b, op=op_map[funct3 & 3])
            _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
            return 0

    if funct7 == GTX_F7_VEC_DOT_SUM:
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        if (funct3 & 3) == 0:
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            scalar = dot_kernel(view_a, view_b)
        else:
            scalar = vsum_kernel(view_a)
        _l1_view_addr(npu, nest, spu, addr_r, 1)[0] = scalar
        # Reinterpret the 0-d FP16 scalar as 2 bytes (little-endian) and
        # blit straight into L0 — no bit-masking, no Python-side raw int.
        scalar_bytes = scalar.to(torch.float16).reshape(1).contiguous().view(torch.uint8)
        l0 = npu.mem.l0_byte(nest, spu)
        l0[0:2] = scalar_bytes
        return 0

    if funct7 == GTX_F7_VEC_CLAMP:
        sub = funct3 & 3
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        #! sub op 0, 1, 2 가 무엇인지 상수로 표현하는게 좋을듯.
        if sub == 0:
            scalar = _fp16_low16(rs2)
            result = clamp_min_kernel(view_a, scalar)
        elif sub == 1:
            scalar = _fp16_low16(rs2)
            result = clamp_max_kernel(view_a, scalar)
        elif sub == 2:
            result = accum_kernel(view_a)
        else:
            start = _fp16_low16(rs2)
            step = _fp16_high16(rs2)
            result = arange_kernel(vec_size, start, step)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    if funct7 in (0x1C, 0x1D, 0x1E):
        view = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = _apply_unary(funct7, funct3 & 3, view)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    return 0


# Backwards-compat alias for sites that still grep ``firmware_vec_op``.
firmware_vec_op = exec_vec_op


# =============================================================================
# 6. @handler entries
# =============================================================================
# ----- SASMD scalar arith (funct7=0x10): VS funct3=0..3, IS funct3=4..7 ------

@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=0,
         mnemonic='add_vs', mask_funct3=True)
def _exec_add_vs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=1,
         mnemonic='sub_vs', mask_funct3=True)
def _exec_sub_vs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=2,
         mnemonic='mul_vs', mask_funct3=True)
def _exec_mul_vs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=3,
         mnemonic='div_vs', mask_funct3=True)
def _exec_div_vs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=4,
         mnemonic='add_is', mask_funct3=True)
def _exec_add_is(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=5,
         mnemonic='sub_is', mask_funct3=True)
def _exec_sub_is(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=6,
         mnemonic='mul_is', mask_funct3=True)
def _exec_mul_is(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_SASMD, funct3=7,
         mnemonic='div_is', mask_funct3=True)
def _exec_div_is(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


# ----- VSUM / DOT (funct7=0x1A): DOT funct3=0, SUM funct3=1 ------------------

@handler(kind='custom0', funct7=GTX_F7_VEC_DOT_SUM, funct3=0,
         mnemonic='dot_vvs', mask_funct3=True)
def _exec_dot_vvs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_DOT_SUM, funct3=1,
         mnemonic='sum_vs', mask_funct3=True)
def _exec_sum_vs(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


# ----- SASMD vector arith (funct7=0x18): VV funct3=0..3, II funct3=4..7 ------

@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=0,
         mnemonic='add_vv', mask_funct3=True)
def _exec_add_vv(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=1,
         mnemonic='sub_vv', mask_funct3=True)
def _exec_sub_vv(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=2,
         mnemonic='mul_vv', mask_funct3=True)
def _exec_mul_vv(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=3,
         mnemonic='div_vv', mask_funct3=True)
def _exec_div_vv(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=4,
         mnemonic='add_ii', mask_funct3=True)
def _exec_add_ii(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=5,
         mnemonic='sub_ii', mask_funct3=True)
def _exec_sub_ii(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=6,
         mnemonic='mul_ii', mask_funct3=True)
def _exec_mul_ii(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_ARITH, funct3=7,
         mnemonic='div_ii', mask_funct3=True)
def _exec_div_ii(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


# ----- CLAMP family (funct7=0x1F): funct3=0..3 -------------------------------

@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=0,
         mnemonic='clamp_min_v', mask_funct3=True)
def _exec_clamp_min_v(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=1,
         mnemonic='clamp_max_v', mask_funct3=True)
def _exec_clamp_max_v(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=2,
         mnemonic='accum_v', mask_funct3=True)
def _exec_accum_v(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


@handler(kind='custom0', funct7=GTX_F7_VEC_CLAMP, funct3=3,
         mnemonic='arange_v', mask_funct3=True)
def _exec_arange_v(npu, proc, insn, xs1, xs2):
    return exec_vec_op(npu, proc, insn)


# ----- MATH / SIGN / ROUND (funct7 = 0x1C / 0x1D / 0x1E) ---------------------
# Sub-op selected by funct3 (& 3); P8 NEG fix preserved (one mnemonic per
# funct3, all routed through exec_vec_op → _apply_unary).
#
# The stacked @handler decorators below register THIS function under 11
# different (funct7, funct3, mnemonic) tuples — abs_v, sqrt_v, exp_v,
# log_v, neg_v, sign_v, step_v, ceil_v, trunc_v, floor_v, rne_v. The
# dispatch table calls it whenever any of those mnemonics decodes; ABS
# firmware → ``abs_v`` is one of those entry points (ABS regression
# proves the wiring). Renamed from ``_exec_unary_family`` to
# ``_exec_vec_unary`` for clarity.
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=0,
         mnemonic='sqrt_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=1,
         mnemonic='exp_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_MATH, funct3=2,
         mnemonic='log_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=0,
         mnemonic='abs_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=1,
         mnemonic='neg_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=2,
         mnemonic='sign_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_SIGN, funct3=3,
         mnemonic='step_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=0,
         mnemonic='ceil_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=1,
         mnemonic='trunc_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=2,
         mnemonic='floor_v', mask_funct3=True)
@handler(kind='custom0', funct7=GTX_F7_VEC_ROUND, funct3=3,
         mnemonic='rne_v', mask_funct3=True)
def _exec_vec_unary(npu, proc, insn, xs1, xs2):
    """Element-wise unary entry (MATH/SIGN/ROUND).

    Sub-op decoded from (funct7, funct3) inside exec_vec_op → _apply_unary.
    """
    return exec_vec_op(npu, proc, insn)
