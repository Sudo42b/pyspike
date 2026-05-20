"""Vector + scalar arithmetic handlers — port of gtx_npu_vec.cc / gtx_npu_custom0.cc.

Single comprehensive owner of every VEC/SCALAR funct7 (0x10-0x1F plus the
0x11 FMADD_S / 0x13 MINMAX_S / 0x19 FMADD_V families). The scalar (_VS/_IS)
ops that previously lived in scalar.py are merged here so each
``(funct7, funct3)`` is registered exactly once — the registry keys solely on
``(funct7, funct3)``, so a duplicate would silently overwrite.

Dispatch key reminder: the runtime computes
``funct3 = inst.xd<<2 | inst.xs1<<1 | inst.xs2``. For a shared funct7 the
``funct3`` selects the sub-operation; ``funct3 & 4`` switches the L1 (VS/VV)
path to the L0 SVR (IS/II) path. funct3 assignments verified against the
vendor authoritative decode in gtx_npu_vec.cc:572-754 and
gtx_npu_custom0.cc:304-399 (ucode_to_funct7 in gtx_npu.h is the MEXEC path —
the RoCC encoding uses these {xd,xs1,xs2} bits directly).
"""
from __future__ import annotations

import torch

from ...inst_handler import inst_register

from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu

# funct7 values referenced by the vec_op dispatcher body.
F7_SASMD = 0b0010000        # 0x10  SASMD add/sub/mul/div _VS/_IS
F7_FMADD_S = 0b0010001      # 0x11  scalar FMADD fmadd.vss / fmadd.iss
F7_MINMAX_S = 0b0010011     # 0x13  scalar MIN/MAX max.vs/min.vs/max.is/min.is
F7_FMADD_V = 0b0011001      # 0x19  vector FMADD fmadd.vvv / fmadd.iii
F7_VEC_ARITH = 0b0011000    # 0x18
F7_VEC_DOT_SUM = 0b0011010  # 0x1A
F7_VEC_CLAMP = 0b0011111    # 0x1F

# vec sub-op ids used by the SASMD/CLAMP arithmetic kernels.
VEC_ADD, VEC_SUB, VEC_MUL, VEC_DIV = 0, 1, 2, 3

# CLAMP_V funct3>=4 logic ops (vendor gtx_npu.h:444-447 GTX_IMM_AND/OR/NOT/SHIFT,
# selected via funct3 & 3 once the >=4 bit routes to the L0 bitwise path).
LOGIC_AND, LOGIC_OR, LOGIC_NOT, LOGIC_SHIFT = 0, 1, 2, 3

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


def fmadd_kernel(a, scalar_b, scalar_c) -> torch.Tensor:
    """``out[i] = a[i]*b + c`` — FP32 internal, FP16 output (vendor:334/394)."""
    a_f32 = _as_fp32(a)
    b = float(scalar_b)
    c = float(scalar_c)
    return (a_f32 * b + c).to(torch.float16)


def minmax_reduce_kernel(a, scalar, is_min: bool) -> torch.Tensor:
    """Reduce ``a`` against ``scalar`` to a single FP16 (vendor:307-323/375-383).

    ``result`` starts at ``scalar`` then folds max/min over every element —
    matching the vendor seed-with-scalar reduction, FP32 internal.
    """
    a_f32 = _as_fp32(a).reshape(-1)
    seed = torch.tensor(float(scalar), dtype=torch.float32, device=a_f32.device)
    if is_min:
        result = torch.minimum(a_f32.min(), seed) if a_f32.numel() else seed
    else:
        result = torch.maximum(a_f32.max(), seed) if a_f32.numel() else seed
    return result.to(torch.float16)


def logic_kernel(a_u16: torch.Tensor, b_u16, sub_op: int) -> torch.Tensor:
    """Bitwise op on FP16 raw uint16 bits (vendor exec_bitwise_imm:532-560).

    ``b_u16`` is the second register's bits for AND/OR; for SHIFT it is the
    raw rs2 integer (bits[3:0]=amount, bit[4]=direction). NOT ignores ``b``.
    """
    if sub_op == LOGIC_AND:
        return a_u16 & b_u16
    if sub_op == LOGIC_OR:
        return a_u16 | b_u16
    if sub_op == LOGIC_NOT:
        # torch lacks bitwise_not for uint16; XOR-0xFFFF is the FP16-width
        # one's-complement, matching vendor ``~a_val`` on uint16_t.
        return a_u16 ^ torch.tensor(0xFFFF, dtype=torch.uint16, device=a_u16.device)
    # SHIFT: amount in b[3:0], direction in b[4] (1=left, 0=right).
    # torch has no uint16 shift; promote to int32 (values are non-negative,
    # so >> is logical) and mask back to 16 bits on the left-shift overflow.
    amt = int(b_u16) & 0xF
    a32 = a_u16.to(torch.int32)
    if int(b_u16) & 0x10:
        return ((a32 << amt) & 0xFFFF).to(torch.uint16)
    return (a32 >> amt).to(torch.uint16)


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


def _l0_block_view_u16(npu, nest: int, spu: int, reg: int) -> torch.Tensor:
    """Return a raw uint16 view of ``L0[(reg & 0x1F)*32 .. +32]``; 16 words.

    Used by the bitwise logic ops (AND/OR/NOT/SHIFT) which operate on the
    FP16 raw bit pattern, not numeric values (vendor exec_bitwise_imm).
    """
    l0 = npu.mem.l0_byte(nest, spu)
    off = ((reg & 0x1F) * 32) % L0_SIZE_BYTES
    return l0.view(torch.uint16)[off // 2:off // 2 + 16]


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
                     rs1: int, rs2: int, inst, vec_size: int) -> int:
    op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
    sub = funct3 & 3

    scalar = _fp16_low16(rs2)
    if not (funct3 & 4):
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = sasmd_kernel(view_a, scalar, op=op_map[sub])
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    result = sasmd_kernel(view_a, scalar, op=op_map[sub])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def arith_l0_ii(npu, nest: int, spu: int, sub_op: int,
                            rs1: int, rs2: int, inst) -> int:
    op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
    if sub_op not in op_map:
        return 0
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, a_reg)
    view_b = _l0_block_view(npu, nest, spu, b_reg)
    result = sasmd_kernel(view_a, view_b, op=op_map[sub_op])
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def unary_l0(npu, nest: int, spu: int, funct7: int, sub_op: int,
                        rs1: int, inst) -> int:
    input_reg = rs1 & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0xFFFFFFFF))
    result_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else input_reg
    view = _l0_block_view(npu, nest, spu, input_reg)
    result = _apply_unary(funct7, sub_op, view)
    _l0_block_view(npu, nest, spu, result_reg).copy_(result)
    return 0


def fmadd_s(npu, nest: int, spu: int, funct3: int,
            rs1: int, rs2: int, inst, vec_size: int) -> int:
    """Scalar FMADD (funct7=0x11). VS=L1 (funct3&4==0), IS=L0 (funct3&4).

    ``a*b + c`` where b=rs2[15:0], c=GSPR_GTX_OPERAND2[15:0]
    (vendor gtx_npu_custom0.cc:352-374; math in exec_vec_scalar:334).
    """
    scalar_b = _fp16_low16(rs2)
    scalar_c = _fp16_low16(int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0)))
    if not (funct3 & 4):
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = fmadd_kernel(view_a, scalar_b, scalar_c)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0
    src_reg = rs1 & 0x1F
    dst_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, src_reg)
    result = fmadd_kernel(view_a, scalar_b, scalar_c)
    _l0_block_view(npu, nest, spu, dst_reg).copy_(result)
    return 0


def minmax_s(npu, nest: int, spu: int, funct3: int,
             rs1: int, rs2: int, inst, vec_size: int) -> int:
    """Scalar MIN/MAX reduction (funct7=0x13). funct3&1 selects min vs max.

    Reduces the vector against scalar=rs2[15:0] to a single FP16, written to
    L0 (vendor gtx_npu_custom0.cc:377-398; reduction in exec_vec_scalar:307).
    """
    scalar = _fp16_low16(rs2)
    is_min = bool(funct3 & 1)
    if not (funct3 & 4):
        # VS: read L1[ADDRA], single scalar result -> L0[ADDRR & 0x1F].
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
        result = minmax_reduce_kernel(view_a, scalar, is_min)
        dst = _l0_block_view(npu, nest, spu, addr_r & 0x1F)
        dst.zero_()
        dst[0] = result
        return 0
    # IS: reduce the 16-element L0 src register -> L0 dst register.
    src_reg = rs1 & 0x1F
    dst_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _l0_block_view(npu, nest, spu, src_reg)
    result = minmax_reduce_kernel(view_a, scalar, is_min)
    dst = _l0_block_view(npu, nest, spu, dst_reg)
    dst.zero_()
    dst[0] = result
    return 0


def fmadd_v(npu, nest: int, spu: int, funct3: int,
            rs1: int, rs2: int, inst, vec_size: int) -> int:
    """Vector FMADD (funct7=0x19). VVV=L1 (funct3&4==0), III=L0 (funct3&4).

    L1 path: out = A*B + C across L1[ADDRA]/[ADDRB]/[ADDRR-as-C? vendor uses
    GTX_VEC_FMADD via exec_vector_op]. L0 III: a*b+c on SVR regs, c from
    rs2[9:5] (vendor gtx_npu_vec.cc:619-631 + exec_vector_imm:441-447).
    """
    if not (funct3 & 4):
        # VVV: element-wise A*B + C on L1 (vendor GTX_VEC_FMADD, exec_vector_op).
        addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
        addr_b = npu.lspr[nest][spu].get(LSPR['SPM_ADDRB'].address, 0)
        addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
        view_a = _as_fp32(_l1_view_addr(npu, nest, spu, addr_a, vec_size))
        view_b = _as_fp32(_l1_view_addr(npu, nest, spu, addr_b, vec_size))
        view_r = _l1_view_addr(npu, nest, spu, addr_r, vec_size)
        result = (view_a * view_b + _as_fp32(view_r)).to(torch.float16)
        view_r.copy_(result)
        return 0
    # III: a*b + c on L0 SVR regs. a=rs1[4:0], b=rs2[4:0], c=rs2[9:5].
    a_reg = rs1 & 0x1F
    b_reg = rs2 & 0x1F
    c_reg = (rs2 >> 5) & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _as_fp32(_l0_block_view(npu, nest, spu, a_reg))
    view_b = _as_fp32(_l0_block_view(npu, nest, spu, b_reg))
    view_c = _as_fp32(_l0_block_view(npu, nest, spu, c_reg))
    result = (view_a * view_b + view_c).to(torch.float16)
    _l0_block_view(npu, nest, spu, r_reg).copy_(result)
    return 0


def logic_l0(npu, nest: int, spu: int, sub_op: int,
             rs1: int, rs2: int, inst) -> int:
    """L0 bitwise logic AND/OR/NOT/SHIFT (vendor exec_bitwise_imm:510-564).

    a=rs1[4:0]; b=rs2[4:0] for AND/OR; SHIFT uses raw rs2; NOT ignores b.
    Result reg from GSPR_GTX_OPERAND3[4:0].
    """
    a_reg = rs1 & 0x1F
    r_reg = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, inst.rd)) & 0x1F
    view_a = _l0_block_view_u16(npu, nest, spu, a_reg)
    if sub_op == LOGIC_SHIFT:
        result = logic_kernel(view_a, rs2, sub_op)
    elif sub_op == LOGIC_NOT:
        result = logic_kernel(view_a, 0, sub_op)
    else:
        view_b = _l0_block_view_u16(npu, nest, spu, rs2 & 0x1F)
        result = logic_kernel(view_a, view_b, sub_op)
    _l0_block_view_u16(npu, nest, spu, r_reg).copy_(result)
    return 0


# =============================================================================
# 5. vec_op / firmware_vec_op
# =============================================================================
def vec_op(npu, proc, inst) -> int:
    """Direct port of ``gtx_npu_vec.cc:572-754``."""
    rs1 = int(proc.state.XPR[inst.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000

    funct7 = inst.fn7
    funct3 = inst.fn3

    nest, spu = _resolve_nest_spu(npu)
    assert nest < NEST_NUM, f"NEST id {nest} >= NEST_NUM={NEST_NUM}"
    assert spu < SPU_NUM, f"SPU id {spu} >= SPU_NUM={SPU_NUM}"

    rs2 = int(proc.state.XPR[inst.rs2])
    npu.gspr[GSPR['GSPR_GTX_OPERAND2'].address] = rs2

    if funct7 == F7_SASMD:
        return sasmd(npu, nest, spu, funct3, rs1, rs2, inst, vec_size)

    if funct7 == F7_FMADD_S:
        return fmadd_s(npu, nest, spu, funct3, rs1, rs2, inst, vec_size)

    if funct7 == F7_MINMAX_S:
        return minmax_s(npu, nest, spu, funct3, rs1, rs2, inst, vec_size)

    if funct7 == F7_FMADD_V:
        return fmadd_v(npu, nest, spu, funct3, rs1, rs2, inst, vec_size)

    # CLAMP_V funct3>=4 → L0 bitwise logic (AND/OR/NOT/SHIFT), not clamp.
    if funct7 == F7_VEC_CLAMP and (funct3 & 4):
        return logic_l0(npu, nest, spu, funct3 & 3, rs1, rs2, inst)

    if funct7 == F7_VEC_ARITH and (funct3 & 4):
        return arith_l0_ii(npu, nest, spu, funct3 & 3, rs1, rs2, inst)
    if funct7 in (0x1C, 0x1D, 0x1E) and (funct3 & 4):
        return unary_l0(npu, nest, spu, funct7, funct3 & 3, rs1, inst)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_b = npu.lspr[nest][spu].get(LSPR['SPM_ADDRB'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)

    if funct7 == F7_VEC_ARITH:
        op_map = {0: VEC_ADD, 1: VEC_SUB, 2: VEC_MUL, 3: VEC_DIV}
        if (funct3 & 3) in op_map:
            view_a = _l1_view_addr(npu, nest, spu, addr_a, vec_size)
            view_b = _l1_view_addr(npu, nest, spu, addr_b, vec_size)
            result = sasmd_kernel(view_a, view_b, op=op_map[funct3 & 3])
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
        import os as _os, sys as _sys
        if _os.environ.get("GTX_DEBUG_UNARY"):
            print(f"[UNARY] n{nest}s{spu} f7=0x{funct7:02x} f3={funct3} "
                  f"vs={vec_size} addr_a=0x{addr_a:x} addr_r=0x{addr_r:x} "
                  f"in0={float(view.reshape(-1)[0]):.3f} out0={float(result.reshape(-1)[0]):.3f}",
                  file=_sys.stderr, flush=True)
        _l1_view_addr(npu, nest, spu, addr_r, vec_size).copy_(result)
        return 0

    return 0

# ----- Scalar SASMD (funct7=0x10): VS funct3=0..3, IS funct3=4..7 -----------
@inst_register.custom0(name='add.vs', funct7=0b0010000, funct3=0b000)
def add_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='sub.vs', funct7=0b0010000, funct3=0b001)
def sub_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='mul.vs', funct7=0b0010000, funct3=0b010)
def mul_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='div.vs', funct7=0b0010000, funct3=0b011)
def div_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='add.is', funct7=0b0010000, funct3=0b100)
def add_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='sub.is', funct7=0b0010000, funct3=0b101)
def sub_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='mul.is', funct7=0b0010000, funct3=0b110)
def mul_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='div.is', funct7=0b0010000, funct3=0b111)
def div_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


# ----- Vector arith (funct7=0x18): VV funct3=0..3, II funct3=4..7 ------------
@inst_register.custom0(name='add.vv', funct7=0b0011000, funct3=0b000)
def add_vv(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='sub.vv', funct7=0b0011000, funct3=0b001)
def sub_vv(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='mul.vv', funct7=0b0011000, funct3=0b010)
def mul_vv(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='div.vv', funct7=0b0011000, funct3=0b011)
def div_vv(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='fmadd.vvv', funct7=0b0011001, funct3=0b000)
def fmadd_vvv(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='sqrt.v', funct7=0b0011100, funct3=0b000)
def sqrt_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='exp.v', funct7=0b0011100, funct3=0b001)
def exp_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)






# ----- Scalar FMADD (funct7=0x11): fmadd.vss=0, fmadd.iss=4 ------------------
@inst_register.custom0(name='fmadd.vss', funct7=0b0010001, funct3=0b000)
def fmadd_vss(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='fmadd.iss', funct7=0b0010001, funct3=0b100)
def fmadd_iss(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

# ----- Scalar MIN/MAX (funct7=0x13): max.vs=0,min.vs=1,max.is=4,min.is=5 -----
@inst_register.custom0(name='max.vs', funct7=0b0010011, funct3=0b000)
def max_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='min.vs', funct7=0b0010011, funct3=0b001)
def min_vs(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='max.is', funct7=0b0010011, funct3=0b100)
def max_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='min.is', funct7=0b0010011, funct3=0b101)
def min_is(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='ln.v', funct7=0b0011100, funct3=0b010)
def ln_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='abs.v', funct7=0b0011101, funct3=0b000)
def abs_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='neg.v', funct7=0b0011101, funct3=0b001)
def neg_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='sign.v', funct7=0b0011101, funct3=0b010)
def sign_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='step.v', funct7=0b0011101, funct3=0b011)
def step_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='ceil.v', funct7=0b0011110, funct3=0b000)
def ceil_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='trunc.v', funct7=0b0011110, funct3=0b001)
def trunc_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='floor.v', funct7=0b0011110, funct3=0b010)
def floor_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='rne.v', funct7=0b0011110, funct3=0b011)
def rne_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='clamp.min', funct7=0b0011111, funct3=0b000)
def clamp_min(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='clamp.max', funct7=0b0011111, funct3=0b001)
def clamp_max(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='accum', funct7=0b0011111, funct3=0b010)
def accum_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='arange', funct7=0b0011111, funct3=0b011)
def arange_v(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='add.ii', funct7=0b0011000, funct3=0b100)
def add_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='sub.ii', funct7=0b0011000, funct3=0b101)
def sub_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='mul.ii', funct7=0b0011000, funct3=0b110)
def mul_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='div.ii', funct7=0b0011000, funct3=0b111)
def div_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='fmadd.iii', funct7=0b0011001, funct3=0b100)
def fmadd_iii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='sqrt.i', funct7=0b0011100, funct3=0b100)
def sqrt_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='exp.i', funct7=0b0011100, funct3=0b101)
def exp_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='ln.i', funct7=0b0011100, funct3=0b110)
def ln_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='abs.i', funct7=0b0011101, funct3=0b100)
def abs_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='neg.i', funct7=0b0011101, funct3=0b101)
def neg_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='sign.i', funct7=0b0011101, funct3=0b110)
def sign_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='step.i', funct7=0b0011101, funct3=0b111)
def step_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='ceil.i', funct7=0b0011110, funct3=0b100)
def ceil_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='trunc.i', funct7=0b0011110, funct3=0b101)
def trunc_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='floor.i', funct7=0b0011110, funct3=0b110)
def floor_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='rne.i', funct7=0b0011110, funct3=0b111)
def rne_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)

@inst_register.custom0(name='and.ii', funct7=0b0011111, funct3=0b100)
def and_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='or.ii', funct7=0b0011111, funct3=0b101)
def or_ii(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='not.i', funct7=0b0011111, funct3=0b110)
def not_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)


@inst_register.custom0(name='shift.i', funct7=0b0011111, funct3=0b111)
def shift_i(npu, proc, inst, cxt) -> int:
    return vec_op(npu, proc, inst)