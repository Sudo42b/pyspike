"""
ACT op @handler entries + activation/format-cvt/pool kernels
    prelu	4'b0101	3'b000	gpr	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	slop_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 prelu(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    gelu	4'b0101	3'b010	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 gelu(A)	-
    tanh	4'b0101	3'b100	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 tanh(A)	-
    sigm	4'b0101	3'b101	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 sigmoid(A)	-
    prelu.i	4'b0101	3'b000	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	slop_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 prelu(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    gelu.i	4'b0101	3'b010	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 gelu(A) imm	-
    tanh.i	4'b0101	3'b100	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 tanh(A) imm	-
    sigm.i	4'b0101	3'b101	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 sigmoid(A) imm	-

"""
from __future__ import annotations

import torch
from torch import Tensor
from .softmax import softmax, esum

from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM
from ....csr import GSPR, LSPR
# ============================================================================
# Activation sub-opcodes (passed via GSPR_OPCODE or encoded in operands)
# ============================================================================
ACT_RELU:int    = 0
ACT_TANH:int    = 1
ACT_SOFTMAX:int = 2
ACT_GELU:int    = 3
ACT_SIGMOID:int = 4
ACT_PRELU:int   = 5
ACT_ESUM:int    = 6
# =========================================================================
# Phase 5: ACT funct7 constants (8 ISS activations split across 5 funct7 values)
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:152-157
# =========================================================================
F7_ACT_PRELU: int = 0x28        # prelu funct3=3 / prelu_i funct3=7
F7_ACT_GELU: int = 0x2A         # gelu funct3=0 / gelu_i funct3=4
F7_ACT_TANH: int = 0x2C         # tanh funct3=0 / tanh_i funct3=4
F7_ACT_SIGM: int = 0x2D         # sigm funct3=0 / sigm_i funct3=4

# Activations that swap LSPR direction: rd=ADDRR, wr=ADDRA (vendor
# gtx_npu_act.cc:37-42). RELU/SOFTMAX/ESUM use forward ADDRA→ADDRR.
ACT_OPS_REVERSED: frozenset = frozenset({
    ACT_PRELU, ACT_GELU, ACT_TANH, ACT_SIGMOID,
})

# L0 activations (funct7=0x5C)
IMM_ACT_PRELU   = 0
IMM_ACT_GELU    = 1
IMM_ACT_TANH    = 2
IMM_ACT_SIGM    = 3
F7_ACT_IMM:int = 0b1011100     # 0x5C — L0 activations

# Activation functions
F7_PRELU:int = 0b0101000       # PReLU, prelu, prelu.i
F7_GELU:int = 0b0101010        # GeLU gelu, gelu.i
F7_TANH:int = 0b0101100        # Tanh tanh, tanh.i
F7_SIGM:int = 0b0101101        # Sigmoid sigmoid, sigm.i
# =============================================================================
# 3. Activation kernels
# =============================================================================
def relu(arr_f16: Tensor) -> torch.Tensor:
    return torch.relu(arr_f16.to(torch.float32)).to(torch.float16)


def prelu(arr_f16: Tensor, slope: Tensor) -> Tensor:
    return torch.nn.functional.prelu(arr_f16.to(torch.float32), slope).to(torch.float16)


def gelu(arr_f16: Tensor) -> Tensor:
    return torch.nn.functional.gelu(arr_f16.to(torch.float32)).to(torch.float16)


def tanh(arr_f16: Tensor) -> Tensor:
    return torch.tanh(arr_f16.to(torch.float32)).to(torch.float16)


def sigmoid(arr_f16: Tensor) -> Tensor:
    return torch.sigmoid(arr_f16.to(torch.float32)).to(torch.float16)

# =============================================================================
# dispatch surface
# =============================================================================
def act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:23-164`` (``exec_activation``).

    Direction asymmetry
        forward  (RELU/SOFTMAX/ESUM):       rd=ADDRA, wr=ADDRR  (ESUM → L0) 정방향
        reversed (TANH/GELU/SIGMOID/PRELU): rd=ADDRR, wr=ADDRA 역방향
    """
    # 적용되는 NEST / SPU 결정
    from ... import _resolve_nest_spu
    nest, spu = _resolve_nest_spu(npu)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    rd_addr, wr_addr = (addr_r, addr_a) if is_reversed else (addr_a, addr_r)

    length = int(proc.state.XPR[insn.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    l1_f16 = npu.mem.l1_f16(nest, spu)
    rd_off = (rd_addr // 2) % (l1_f16.shape[0])
    view_in = l1_f16[rd_off:rd_off + length]

    if op_id == ACT_RELU:
        result = relu(view_in)
    elif op_id == ACT_TANH:
        result = tanh(view_in)
    elif op_id == ACT_SOFTMAX:
        result = softmax(view_in)
    elif op_id == ACT_GELU:
        result = gelu(view_in)
    elif op_id == ACT_SIGMOID:
        result = sigmoid(view_in)
    elif op_id == ACT_PRELU:
        slope = _fp16_low16(int(npu.gspr.get(GSPR['GSPR_OPERAND2'].address, 0)))
        result = prelu(view_in, slope)
    elif op_id == ACT_ESUM:
        # Pitfall 8: ESUM is forward (rd=ADDRA) but writes a scalar to L0
        # at offset (GSPR_OPERAND3 & 0x1F)*32 — not to L1[ADDRR].
        op2 = int(npu.gspr.get(GSPR['GSPR_OPERAND2'].address, 0))
        max_val = _fp16_low16(op2)
        init_accum = _fp16_high16(op2)
        scalar = esum(view_in, max_val=max_val, init_accum=init_accum)
        l0_offset = (
            (int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, 0)) & 0x1F) * 32
        ) % L0_SIZE_BYTES
        _write_l0_fp16_scalar(npu, nest, spu, l0_offset, scalar)
        return 0
    else:
        return 0

    wr_off = (wr_addr // 2) % (l1_f16.shape[0])
    l1_f16[wr_off:wr_off + length] = result
    return 0


def act_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path. ``gtx_npu_act.cc:374-431``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[insn.rs1]) & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, 0xFFFFFFFF))
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (insn.rd & 0x1F)

    view_in = _l0_block_view(npu, nest, spu, in_reg)
    view_out = _l0_block_view(npu, nest, spu, out_reg)

    if op_id == ACT_PRELU:
        slope = _fp16_low16(int(npu.gspr.get(GSPR['GSPR_OPERAND2'].address, 0)))
        result = prelu(view_in, slope)
    elif op_id == ACT_GELU:
        result = gelu(view_in)
    elif op_id == ACT_TANH:
        result = tanh(view_in)
    elif op_id == ACT_SIGMOID:
        result = sigmoid(view_in)
    else:
        return 0

    view_out[:] = result
    return 0


# =============================================================================
# 7. @handler entries (funct7/funct3 binding policy)
# =============================================================================
# ----- ISS L1 path (8 activations: 4 reversed + 2 forward at 0x2F) -----------

@handler(kind='custom0', funct7=F7_ACT_PRELU, funct3=3,
         mnemonic='prelu', mask_funct3=True)
def prelu(npu, proc, insn, xs1, xs2):
    """PRELU: funct7=0x28 funct3=3, REVERSED (vendor cc:37-42)."""
    return act(npu, proc, insn,
                         op_id=ACT_PRELU, is_reversed=True)


@handler(kind='custom0', funct7=F7_ACT_GELU, funct3=0,
         mnemonic='gelu', mask_funct3=True)
def gelu(npu, proc, insn, xs1, xs2):
    """GELU: funct7=0x2A funct3=0, REVERSED."""
    return act(npu, proc, insn,
                         op_id=ACT_GELU, is_reversed=True)


@handler(kind='custom0', funct7=F7_ACT_TANH, funct3=0,
         mnemonic='tanh', mask_funct3=True)
def tanh(npu, proc, insn, xs1, xs2):
    """TANH: funct7=0x2C funct3=0, REVERSED."""
    return act(npu, proc, insn,
                         op_id=ACT_TANH, is_reversed=True)


@handler(kind='custom0', funct7=F7_ACT_SIGM, funct3=0,
         mnemonic='sigmoid', mask_funct3=True)
def sigmoid(npu, proc, insn, xs1, xs2):
    """SIGMOID: funct7=0x2D funct3=0, REVERSED."""
    return act(npu, proc, insn,
                         op_id=ACT_SIGMOID, is_reversed=True)



# ----- L0 immediate path (6 _imm activations: funct3 & 4 selects L0) --------

@handler(kind='custom0', funct7=F7_ACT_PRELU, funct3=7,
         mnemonic='prelu_i', mask_funct3=True)
def prelu_i(npu, proc, insn, xs1, xs2):
    return act_imm(npu, proc, insn, op_id=ACT_PRELU)


@handler(kind='custom0', funct7=F7_ACT_GELU, funct3=4,
         mnemonic='gelu_i', mask_funct3=True)
def gelu_i(npu, proc, insn, xs1, xs2):
    return act_imm(npu, proc, insn, op_id=ACT_GELU)


@handler(kind='custom0', funct7=F7_ACT_TANH, funct3=4,
         mnemonic='tanh_i', mask_funct3=True)
def tanh_i(npu, proc, insn, xs1, xs2):
    return act_imm(npu, proc, insn, op_id=ACT_TANH)


@handler(kind='custom0', funct7=F7_ACT_SIGM, funct3=4,
         mnemonic='sigm_i', mask_funct3=True)
def sigm_i(npu, proc, insn, xs1, xs2):
    return act_imm(npu, proc, insn, op_id=ACT_SIGMOID)

