from __future__ import annotations

import math as _math

import numpy as np
from numpy import ndarray as Tensor
from .softmax import softmax_step3, esum

from ...inst_handler import inst_register
from ....config_params import L0_SIZE_BYTES, NEST_NUM, SPU_NUM, MX_IO_DTYPE, MX_IO_BYTES
from ....csr import GSPR, LSPR
from ... import _resolve_nest_spu, operand3, rs_select
from . import _io_low, _io_high, _l0_block_view_io, _write_l0_io_pair


def _resolve_softmax_op2(npu, nest, spu) -> int:
    """ESUM/SOFTMAX max/esum operand. Firmware passes max_value (and esum_value
    for softmax) as the OPERAND2 immediate but sets r2_sel (source_sel/OPERAND5)
    to read them from an SVR instead: SVR_0 (max from max.vs) for esum, SVR_1
    (the max,esum pair esum wrote) for softmax. ``rs_select`` returns the SVR
    word in SVR mode, else the OPERAND2 immediate. Decode max=_io_low,
    esum/accum=_io_high of the result."""
    op2_imm = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    raw, _is_svr = rs_select(npu, nest, spu, op2_imm)
    return raw

# activation op-id enum (internal dispatch ids for act/act_imm kernels).
ACT_RELU, ACT_TANH, ACT_SOFTMAX, ACT_GELU, ACT_SIGMOID, ACT_PRELU, ACT_ESUM = range(7)

# =============================================================================
# 3. Activation kernels
# =============================================================================
# FP32 internal compute, MX_IO_DTYPE output (FP32 by default — see config_params).
_erf_vec = np.vectorize(_math.erf, otypes=[np.float32])   # exact erf (no scipy dep)


def relu(arr: Tensor) -> np.ndarray:
    return np.maximum(arr.astype(np.float32), np.float32(0)).astype(MX_IO_DTYPE)


def prelu(arr: Tensor, slope: Tensor) -> Tensor:
    x = arr.astype(np.float32)
    s = slope.astype(np.float32)
    return np.where(x >= 0, x, s * x).astype(MX_IO_DTYPE)


def gelu(arr: Tensor) -> Tensor:
    x = arr.astype(np.float32)
    return (x * 0.5 * (1.0 + _erf_vec(x * np.float32(0.7071067811865476)))).astype(MX_IO_DTYPE)


def tanh(arr: Tensor) -> Tensor:
    return np.tanh(arr.astype(np.float32)).astype(MX_IO_DTYPE)


def sigmoid(arr: Tensor) -> Tensor:
    return (1.0 / (1.0 + np.exp(-arr.astype(np.float32)))).astype(MX_IO_DTYPE)

# =============================================================================
# dispatch surface
# =============================================================================
def act(npu, proc, inst, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of ``gtx_npu_act.cc:23-164`` (``exec_activation``).

    Direction asymmetry
        forward  (RELU/SOFTMAX/ESUM):       rd=ADDRA, wr=ADDRR  (ESUM → L0) 정방향
        reversed (TANH/GELU/SIGMOID/PRELU): rd=ADDRR, wr=ADDRA 역방향
    """
    # 적용되는 NEST / SPU 결정
    nest, spu = _resolve_nest_spu(npu)

    addr_a = npu.lspr[nest][spu].get(LSPR['SPM_ADDRA'].address, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR['SPM_ADDRR'].address, 0)
    rd_addr, wr_addr = (addr_r, addr_a) if is_reversed else (addr_a, addr_r)

    length = int(proc.state.XPR[inst.rs1]) & 0xFFFF
    if length == 0:
        length = 0x10000

    l1_io = npu.mem.l1_io(nest, spu)
    rd_off = (rd_addr // MX_IO_BYTES) % (l1_io.shape[0])
    view_in = l1_io[rd_off:rd_off + length]

    if op_id == ACT_RELU:
        result = relu(view_in)
    elif op_id == ACT_TANH:
        result = tanh(view_in)
    elif op_id == ACT_SOFTMAX:
        # max/esum come from the SVR named by r2_sel (SVR_1 = max,esum pair),
        # not the OPERAND2 immediate — see _resolve_softmax_op2.
        op2 = _resolve_softmax_op2(npu, nest, spu)
        result = softmax_step3(view_in, _io_low(op2), _io_high(op2))
    elif op_id == ACT_GELU:
        result = gelu(view_in)
    elif op_id == ACT_SIGMOID:
        result = sigmoid(view_in)
    elif op_id == ACT_PRELU:
        slope = _io_low(int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0)))
        result = prelu(view_in, slope)
    elif op_id == ACT_ESUM:
        # Pitfall 8: ESUM is forward (rd=ADDRA) but writes a scalar to L0
        # at offset (GSPR_OPERAND3 & 0x1F)*32 — not to L1[ADDRR].
        # max comes from the SVR named by r2_sel (SVR_0 = max.vs result), not
        # the OPERAND2 immediate — see _resolve_softmax_op2.
        op2 = _resolve_softmax_op2(npu, nest, spu)
        max_val = _io_low(op2)
        init_accum = _io_high(op2)
        scalar = esum(view_in, max_val=max_val, init_accum=init_accum)
        l0_offset = ((operand3(npu) & 0x1F) * 32) % L0_SIZE_BYTES
        _write_l0_io_pair(npu, nest, spu, l0_offset, max_val, scalar)
        return 0
    else:
        return 0

    wr_off = (wr_addr // MX_IO_BYTES) % (l1_io.shape[0])
    l1_io[wr_off:wr_off + length] = result
    return 0


def act_imm(npu, proc, inst, *, op_id: int) -> int:
    """L0 immediate path. ``gtx_npu_act.cc:374-431``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[inst.rs1]) & 0x1F
    op3_raw = operand3(npu, 0xFFFFFFFF)
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (inst.rd & 0x1F)

    view_in = _l0_block_view_io(npu, nest, spu, in_reg)
    view_out = _l0_block_view_io(npu, nest, spu, out_reg)

    if op_id == ACT_PRELU:
        slope = _io_low(int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0)))
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


@inst_register.custom0(name='prelu', funct7=0b0101000, funct3=3)
def _prelu(npu, proc, inst, cxt) -> int:
    # prelu	4'b0101	3'b000	gpr	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	slop_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 prelu(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return act(npu, proc, inst,
                         op_id=ACT_PRELU, is_reversed=True)


@inst_register.custom0(name='gelu', funct7=0b0101010, funct3=0)
def _gelu(npu, proc, inst, cxt) -> int:
    # gelu	4'b0101	3'b010	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 gelu(A)	-
    return act(npu, proc, inst,
                         op_id=ACT_GELU, is_reversed=True)


@inst_register.custom0(name='tanh', funct7=0b0101100, funct3=0)
def _tanh(npu, proc, inst, cxt) -> int:
    #     tanh	4'b0101	3'b100	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 tanh(A)	-
    return act(npu, proc, inst,
                         op_id=ACT_TANH, is_reversed=True)


@inst_register.custom0(name='sigmoid', funct7=0b0101101, funct3=0)
def _sigmoid(npu, proc, inst, cxt) -> int:
    #     sigm	4'b0101	3'b101	rsvd	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	N/A	N/A	N/A	N/A	N/A	fp16 sigmoid(A)	-
    return act(npu, proc, inst,
                         op_id=ACT_SIGMOID, is_reversed=True)



# ----- L0 immediate path (6 _imm activations: funct3 & 4 selects L0) --------

@inst_register.custom0(name='prelu.i', funct7=0b0101000, funct3=7)
def _prelu_i(npu, proc, inst, cxt) -> int:
    # prelu.i	4'b0101	3'b000	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	slop_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 prelu(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return act_imm(npu, proc, inst, op_id=ACT_PRELU)


@inst_register.custom0(name='gelu.i', funct7=0b0101010, funct3=4)
def _gelu_i(npu, proc, inst, cxt) -> int:
    # gelu.i	4'b0101	3'b010	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 gelu(A) imm	-
    return act_imm(npu, proc, inst, op_id=ACT_GELU)


@inst_register.custom0(name='tanh.i', funct7=0b0101100, funct3=4)
def _tanh_i(npu, proc, inst, cxt) -> int:
    #     tanh.i	4'b0101	3'b100	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 tanh(A) imm	-
    return act_imm(npu, proc, inst, op_id=ACT_TANH)


@inst_register.custom0(name='sigm.i', funct7=0b0101101, funct3=4)
def _sigm_i(npu, proc, inst, cxt) -> int:
    #     sigm.i	4'b0101	3'b101	rsvd	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	N/A	result_SVR_addr[4:0]	N/A	N/A	result[255:0]	fp16 sigmoid(A) imm	-
    return act_imm(npu, proc, inst, op_id=ACT_SIGMOID)
