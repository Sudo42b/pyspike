import torch
from torch import Tensor

from ...inst_handler import inst_register
from ....config_params import L0_SIZE_BYTES
from ....csr import GSPR
from ... import _resolve_nest_spu
from . import _fp16_low16, _fp16_high16, _fp16_raw_bits, _l0_block_view

# activation op-id enum (must match act.py's ACT_* values).
ACT_SOFTMAX, ACT_ESUM = 2, 6


def softmax_imm(npu, proc, inst, *, op_id: int) -> int:
    """L0 immediate path for ESUM / SOFTMAX. ``gtx_npu_act.cc:436-487``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[inst.rs1]) & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0xFFFFFFFF))
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (inst.rd & 0x1F)

    view_in = _l0_block_view(npu, nest, spu, in_reg)

    op2 = int(npu.gspr.get(GSPR['GSPR_GTX_OPERAND2'].address, 0))
    max_val = _fp16_low16(op2)
    accum_val = _fp16_high16(op2)

    if op_id == ACT_ESUM:
        scalar = esum(view_in, max_val=max_val, init_accum=accum_val)
        l0 = npu.mem.l0_byte(nest, spu)
        r_off = ((out_reg & 0x1F) * 32) % L0_SIZE_BYTES
        r16 = _fp16_raw_bits(scalar)
        m16 = _fp16_raw_bits(max_val)
        l0[r_off] = r16 & 0xFF
        l0[r_off + 1] = (r16 >> 8) & 0xFF
        l0[r_off + 2] = m16 & 0xFF
        l0[r_off + 3] = (m16 >> 8) & 0xFF
        for x in range(2, 16):
            l0[r_off + x * 2] = 0
            l0[r_off + x * 2 + 1] = 0
    elif op_id == ACT_SOFTMAX:
        # r[i] = exp(x[i] - max - ln(esum)); vendor uses pre-computed esum.
        f32 = view_in.to(torch.float32)
        max_f = max_val.to(torch.float32)
        esum_f = accum_val.to(torch.float32)
        ln_esum = (torch.log(esum_f) if float(esum_f) > 0.0
                   else torch.tensor(0.0, device=esum_f.device))
        result = torch.exp(f32 - max_f - ln_esum).to(torch.float16)
        view_out = _l0_block_view(npu, nest, spu, out_reg)
        view_out[:] = result
    return 0


def softmax(arr_f16: Tensor) -> Tensor:
    return torch.nn.functional.softmax(arr_f16.to(torch.float32), dim=0).to(torch.float16)


def esum(arr_f16: Tensor, max_val: float, init_accum: float) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    max_val_f32 = torch.as_tensor(max_val, dtype=torch.float32)
    init_accum_f32 = torch.as_tensor(init_accum, dtype=torch.float32)
    return (init_accum_f32 + torch.sum(torch.exp(arr_f32 - max_val_f32))).to(torch.float16)

@inst_register.custom0(name='esum', funct7=0b0101111, funct3=0b001)
def _esum(npu, proc, inst, cxt) -> int:
    # esum	4'b0101	3'b111	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	max_value[15:0], accumulated_data[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0], esum_value[31:16]	fp16 softmax(step2) - exponential sum	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    """ESUM: funct7=0x2F funct3=1, FORWARD — writes scalar to L0 (Pitfall 8)."""
    from .act import act
    return act(npu, proc, inst, op_id=ACT_ESUM, is_reversed=False)

@inst_register.custom0(name='softmax', funct7=0b0101111, funct3=0b010)
def _softmax(npu, proc, inst, cxt) -> int:
    # softmax	4'b0101	3'b111	gpr	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	max_value[15:0], esum_value[31:16]	N/A	r2_sel[8:0]	N/A	N/A	fp16 softmax(step3) - softmax	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    from .act import act
    return act(npu, proc, inst,
                         op_id=ACT_SOFTMAX, is_reversed=False)

@inst_register.custom0(name='esum.i', funct7=0b0101111, funct3=0b101)
def _esum_i(npu, proc, inst, cxt) -> int:
    # esum.i	4'b0101	3'b111	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	max_value[15:0], accumulated_data[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0], esum_value[31:16]	fp16 softmax(step2) - exponential sum imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return softmax_imm(npu, proc, inst, op_id=ACT_ESUM)


@inst_register.custom0(name='softmax.i', funct7=0b0101111, funct3=0b110)
def _softmax_i(npu, proc, inst, cxt) -> int:
    # softmax.i	4'b0101	3'b111	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	max_value[15:0], esum_value[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 softmax(step3) - softmax imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return softmax_imm(npu, proc, inst, op_id=ACT_SOFTMAX)




