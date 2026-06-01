"""Scalar (_VS/_IS) arithmetic handlers — sole owner of the scalar funct7 family.

Splits the scalar operand ops out of :mod:`vector` so each module owns a
disjoint funct7 set (the registry keys on ``(funct7, funct3)``; a duplicate
would silently overwrite). This module owns:

  0x10  SASMD     add/sub/mul/div  _VS (L1) / _IS (L0)
  0x11  FMADD_S   fmadd.vss / fmadd.iss      (a*b + c, b/c scalar)
  0x13  MINMAX_S  max.vs/min.vs/max.is/min.is (reduce vec against scalar)

Vector (VV/II) ops — 0x18/0x19/0x1A/0x1C-0x1F — stay in :mod:`vector`.

Dispatch key reminder: the runtime computes ``funct3 = xd<<2 | xs1<<1 | xs2``.
For a shared funct7 the ``funct3`` selects the sub-op; ``funct3 & 4`` switches
the L1 (VS) path to the L0 SVR (IS) path. funct3 assignments verified against
gtx_npu_vec.cc:572-754 / gtx_npu_custom0.cc:304-399.

Shared FP32-internal kernels and L1/L0 view helpers are imported from
:mod:`vector` (the single definitions live there); only the scalar-exclusive
kernels (fmadd / minmax reduce) are defined here.
"""
from __future__ import annotations

import numpy as np

from ...inst_handler import inst_register
from ....config_params import NEST_NUM, SPU_NUM, MX_IO_DTYPE
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu, rs_select, operand3
from .vector import Arith, _as_fp32, sasmd_kernel
# Shared MX I/O-width helpers (FP32 default / FP16 toggle) — single defs in
# the package __init__.
from . import (
    _io_low, _io_high, _fp32_low32, _fp32_high32,
    _l1_view_addr_io as _l1_view_addr,
    _l0_block_view_io as _l0_block_view,
)

# =============================================================================
# Scalar-exclusive kernels (FP32 internal, MX_IO_DTYPE output)
# =============================================================================
def fmadd_kernel(a, scalar_b, scalar_c) -> np.ndarray:
    """``out[i] = a[i]*b + c`` — FP32 internal, MX_IO output (vendor:334/394)."""
    a_f32 = _as_fp32(a)
    b = float(scalar_b)
    c = float(scalar_c)
    return (a_f32 * b + c).astype(MX_IO_DTYPE)


def minmax_reduce_kernel(a, scalar, is_min: bool) -> np.ndarray:
    """Reduce ``a`` against ``scalar`` to a single FP16 (vendor:307-323/375-383).

    ``result`` seeds at ``scalar`` then folds max/min over every element —
    matching the vendor seed-with-scalar reduction, FP32 internal.
    """
    a_f32 = _as_fp32(a).reshape(-1)
    seed = np.array(float(scalar), dtype=np.float32)
    if is_min:
        result = np.minimum(a_f32.min(), seed) if a_f32.size else seed
    else:
        result = np.maximum(a_f32.max(), seed) if a_f32.size else seed
    return result.astype(MX_IO_DTYPE)


# =============================================================================
# Preamble + leaf dispatchers — each handler calls its leaf directly with the
# op baked in; the registry already keyed on (funct7, funct3).
# =============================================================================
import os as _os, sys as _sys
_DBG_NORM = _os.environ.get("GTX_DEBUG_NORM")
_dbg_n = [0]


def _prep(npu, proc, inst) -> tuple:
    """Scalar-handler preamble: resolve (nest, spu), decode rs1/rs2 (vec_size is
    rs1[31:0]), stage OPERAND2. Returns ``(nest, spu, rs1, rs2, vec_size)``."""
    rs1 = int(proc.state.XPR[inst.rs1])
    rs2 = int(proc.state.XPR[inst.rs2])
    vec_size = rs1 & 0xFFFFFFFF
    nest, spu = _resolve_nest_spu(npu)
    assert nest < NEST_NUM, f"NEST id {nest} >= NEST_NUM={NEST_NUM}"
    assert spu < SPU_NUM, f"SPU id {spu} >= SPU_NUM={SPU_NUM}"
    npu.gspr[GSPR['GSPR_GTX_OPERAND2'].address] = rs2
    return nest, spu, rs1, rs2, vec_size


def _dbg(npu, tag, nest, spu, rs1, rs2, extra=""):
    if not _DBG_NORM or _dbg_n[0] > 40:
        return
    _dbg_n[0] += 1
    op3 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0xABCD))
    op5 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND5'].address, 0xABCD))
    svrs = []
    for r in range(4):
        v = _l0_block_view(npu, nest, spu, r).reshape(-1)
        svrs.append(float(v[0]))
    print(f"[NORM] n{nest}s{spu} {tag} rs1={rs1:#x} rs2={rs2:#x} op3={op3:#x} op5={op5:#x} "
          f"SVR0={svrs[0]:.4f} SVR1={svrs[1]:.4f} SVR2={svrs[2]:.4f} SVR3={svrs[3]:.4f} {extra}",
          file=_sys.stderr, flush=True)


# ----- SASMD (0x10): scalar = rs_select(rs2) → gpr / zero / L0 SVR -----------
def _scalar_operand(npu, nest, spu, rs2):
    """Decode the scalar (rs2) operand: SVR reads at the MX I/O width, GPR/zero
    immediates as FP32 (firmware packs immediates as FP32 bit patterns)."""
    raw, is_svr = rs_select(npu, nest, spu, rs2)
    return _io_low(raw) if is_svr else _fp32_low32(raw)

# =============================================================================
# Handlers — each names its op directly; the registry routes (funct7, funct3).
# =============================================================================
# ----- SASMD (funct7=0x10): VS funct3=0..3, IS funct3=4..7 -------------------
@inst_register.custom0(name='add.vs', funct7=0b0010000, funct3=0b000)
def add_vs(npu, proc, inst, cxt) -> int:
    """L1 vector-scalar ADD/SUB/MUL/DIV (A from ADDRA, R to ADDRR)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    _dbg(npu, f"{Arith.ADD.name}.vs", nest, spu, _rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} aA=0x{addr_a:x} aR=0x{addr_r:x}")
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = sasmd_kernel(view_a, scalar, op=Arith.ADD)
    return 0

@inst_register.custom0(name='sub.vs', funct7=0b0010000, funct3=0b001)
def sub_vs(npu, proc, inst, cxt) -> int:
    """L1 vector-scalar ADD/SUB/MUL/DIV (A from ADDRA, R to ADDRR)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    _dbg(npu, f"{Arith.SUB.name}.vs", nest, spu, _rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} aA=0x{addr_a:x} aR=0x{addr_r:x}")
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = sasmd_kernel(view_a, scalar, op=Arith.SUB)
    return 0

@inst_register.custom0(name='mul.vs', funct7=0b0010000, funct3=0b010)
def mul_vs(npu, proc, inst, cxt) -> int:
    """L1 vector-scalar ADD/SUB/MUL/DIV (A from ADDRA, R to ADDRR)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    _dbg(npu, f"mul.vs", nest, spu, _rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} aA=0x{addr_a:x} aR=0x{addr_r:x}")
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = sasmd_kernel(view_a, scalar, op=Arith.MUL)
    return 0

@inst_register.custom0(name='div.vs', funct7=0b0010000, funct3=0b011)
def div_vs(npu, proc, inst, cxt) -> int:
    """L1 vector-scalar ADD/SUB/MUL/DIV (A from ADDRA, R to ADDRR)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    _dbg(npu, f"{Arith.DIV.name}.vs", nest, spu, _rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} aA=0x{addr_a:x} aR=0x{addr_r:x}")
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = sasmd_kernel(view_a, scalar, op=Arith.DIV)
    return 0

@inst_register.custom0(name='add.is', funct7=0b0010000, funct3=0b100)
def add_is(npu, proc, inst, cxt) -> int:
    """L0 SVR scalar ADD/SUB/MUL/DIV. a=rs1[4:0], r=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _dbg(npu, f"{Arith.ADD.name}.is", nest, spu, rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} -> SVR{r_reg}")
    _l0_block_view(npu, nest, spu, r_reg)[...] = sasmd_kernel(view_a, scalar, op=Arith.ADD)
    return 0


@inst_register.custom0(name='sub.is', funct7=0b0010000, funct3=0b101)
def sub_is(npu, proc, inst, cxt) -> int:
    """L0 SVR scalar ADD/SUB/MUL/DIV. a=rs1[4:0], r=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _dbg(npu, f"{Arith.SUB.name}.is", nest, spu, rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} -> SVR{r_reg}")
    _l0_block_view(npu, nest, spu, r_reg)[...] = sasmd_kernel(view_a, scalar, op=Arith.SUB)
    return 0


@inst_register.custom0(name='mul.is', funct7=0b0010000, funct3=0b110)
def mul_is(npu, proc, inst, cxt) -> int:
    """L0 SVR scalar ADD/SUB/MUL/DIV. a=rs1[4:0], r=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _dbg(npu, f"mul.is", nest, spu, rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} -> SVR{r_reg}")
    _l0_block_view(npu, nest, spu, r_reg)[...] = sasmd_kernel(view_a, scalar, op=Arith.MUL)
    return 0


@inst_register.custom0(name='div.is', funct7=0b0010000, funct3=0b111)
def div_is(npu, proc, inst, cxt) -> int:
    """L0 SVR scalar ADD/SUB/MUL/DIV. a=rs1[4:0], r=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    scalar = _scalar_operand(npu, nest, spu, rs2)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    r_reg = operand3(npu, inst.rd) & 0x1F
    _dbg(npu, f"div.is", nest, spu, rs1, rs2,
         f"scalar={float(scalar):.4f} a0={float(view_a.reshape(-1)[0]):.4f} -> SVR{r_reg}")
    _l0_block_view(npu, nest, spu, r_reg)[...] = sasmd_kernel(view_a, scalar, op=Arith.DIV)
    return 0


# ----- FMADD_S (funct7=0x11): VSS funct3=0, ISS funct3=4 --------------------
@inst_register.custom0(name='fmadd.vss', funct7=0b0010001, funct3=0b000)
def fmadd_vss(npu, proc, inst, cxt) -> int:
    """L1 A*b + c (b,c scalars packed in rs2)."""
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    # b = rs2[31:0], c = rs2[63:32] — both FP16 immediates packed by firmware.
    result = fmadd_kernel(view_a, _fp32_low32(rs2), _fp32_high32(rs2))
    _l1_view_addr(npu, nest, spu, addr_r, vsz)[...] = result
    return 0


@inst_register.custom0(name='fmadd.iss', funct7=0b0010001, funct3=0b100)
def fmadd_iss(npu, proc, inst, cxt) -> int:
    """L0 a*b + c on an SVR reg. src=rs1[4:0], dst=OPERAND3[4:0]."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    dst_reg = operand3(npu, inst.rd) & 0x1F
    result = fmadd_kernel(view_a, _fp32_low32(rs2), _fp32_high32(rs2))
    _l0_block_view(npu, nest, spu, dst_reg)[...] = result
    return 0

# ----- MINMAX_S (funct7=0x13): VS funct3=0/1, IS funct3=4/5 -----------------
@inst_register.custom0(name='max.vs', funct7=0b0010011, funct3=0b000)
def max_vs(npu, proc, inst, cxt) -> int:
    """L1 reduce A against scalar → one FP16 at the L0 SVR slot in OPERAND3.

    The result SVR addr is staged via ``opset(0, result_svr_addr)`` before the
    instruction (firmware ``__max_vs``/``__min_vs``), so it comes from GSPR
    OPERAND3 — not ADDRR. (ADDRR is the L1 result anchor, unrelated here.)
    """
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    result = minmax_reduce_kernel(view_a, _fp32_low32(rs2), False)
    dst = _l0_block_view(npu, nest, spu, operand3(npu, addr_r) & 0x1F)
    dst.fill(0)
    dst[0] = result
    return 0

@inst_register.custom0(name='min.vs', funct7=0b0010011, funct3=0b001)
def min_vs(npu, proc, inst, cxt) -> int:
    """L1 reduce A against scalar → one FP16 at the L0 SVR slot in OPERAND3.

    The result SVR addr is staged via ``opset(0, result_svr_addr)`` before the
    instruction (firmware ``__max_vs``/``__min_vs``), so it comes from GSPR
    OPERAND3 — not ADDRR. (ADDRR is the L1 result anchor, unrelated here.)
    """
    nest, spu, _rs1, rs2, vsz = _prep(npu, proc, inst)
    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    view_a = _l1_view_addr(npu, nest, spu, addr_a, vsz)
    result = minmax_reduce_kernel(view_a, _fp32_low32(rs2), True)
    dst = _l0_block_view(npu, nest, spu, operand3(npu, addr_r) & 0x1F)
    dst.fill(0)
    dst[0] = result
    return 0

@inst_register.custom0(name='max.is', funct7=0b0010011, funct3=0b100)
def max_is(npu, proc, inst, cxt) -> int:
    """L0 reduce the src SVR reg against scalar → dst SVR reg."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    dst_reg = operand3(npu, inst.rd) & 0x1F
    result = minmax_reduce_kernel(view_a, _fp32_low32(rs2), False)
    dst = _l0_block_view(npu, nest, spu, dst_reg)
    dst.fill(0)
    dst[0] = result
    return 0

@inst_register.custom0(name='min.is', funct7=0b0010011, funct3=0b101)
def min_is(npu, proc, inst, cxt) -> int:
    """L0 reduce the src SVR reg against scalar → dst SVR reg."""
    nest, spu, rs1, rs2, _vsz = _prep(npu, proc, inst)
    view_a = _l0_block_view(npu, nest, spu, rs1 & 0x1F)
    dst_reg = operand3(npu, inst.rd) & 0x1F
    result = minmax_reduce_kernel(view_a, _fp32_low32(rs2), True)
    dst = _l0_block_view(npu, nest, spu, dst_reg)
    dst.fill(0)
    dst[0] = result
    return 0