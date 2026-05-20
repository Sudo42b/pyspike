"""
esum	4'b0101	3'b111	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	max_value[15:0], accumulated_data[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0], esum_value[31:16]	fp16 softmax(step2) - exponential sum	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
softmax	4'b0101	3'b111	gpr	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	max_value[15:0], esum_value[31:16]	N/A	r2_sel[8:0]	N/A	N/A	fp16 softmax(step3) - softmax	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
esum.i	4'b0101	3'b111	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	max_value[15:0], accumulated_data[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0], esum_value[31:16]	fp16 softmax(step2) - exponential sum imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
softmax.i	4'b0101	3'b111	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	max_value[15:0], esum_value[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 softmax(step3) - softmax imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
"""

SOFTMAX_IMM:int = 0b1011101 # 0x5D — L0 ESUM/SOFTMAX
import torch
F7_ACT_SOFTMAX: int = 0x2F      # esum funct3=1, softmax funct3=2; _imm at funct3=5/6

# Softmax
F7_SOFTMAX:int = 0b0101111     # Softmax/ESUM esum, softmax, esum.i, softmax.i
IMM_ACT_ESUM    = 4
IMM_ACT_SOFTMAX = 5

def softmax_imm(npu, proc, insn, *, op_id: int) -> int:
    """L0 immediate path for ESUM / SOFTMAX. ``gtx_npu_act.cc:436-487``."""
    nest, spu = _resolve_nest_spu(npu)

    in_reg = int(proc.state.XPR[insn.rs1]) & 0x1F
    op3_raw = int(npu.gspr.get(GSPR['GSPR_OPERAND3'].address, 0xFFFFFFFF))
    out_reg = (op3_raw & 0x1F) if op3_raw <= 0x1F else (insn.rd & 0x1F)

    view_in = _l0_block_view(npu, nest, spu, in_reg)

    op2 = int(npu.gspr.get(GSPR['GSPR_OPERAND2'].address, 0))
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





@handler(kind='custom0', funct7=F7_ACT_SOFTMAX, funct3=5,
         mnemonic='esum_i', mask_funct3=True)
def _esum_i(npu, proc, insn, xs1, xs2):
    """L0 immediate ESUM (gtx_npu_act.cc:436-487 ESUM branch)."""
    return softmax_imm(npu, proc, insn, op_id=ACT_ESUM)


@handler(kind='custom0', funct7=F7_ACT_SOFTMAX, funct3=6,
         mnemonic='softmax_i', mask_funct3=True)
def _softmax_i(npu, proc, insn, xs1, xs2):
    """L0 immediate SOFTMAX (pre-computed esum from GSPR_OPERAND2 high-16)."""
    return softmax_imm(npu, proc, insn, op_id=ACT_SOFTMAX)


@handler(kind='custom0', funct7=F7_ACT_SOFTMAX, funct3=1,
         mnemonic='esum', mask_funct3=True)
def _esum(npu, proc, insn, xs1, xs2):
    """ESUM: funct7=0x2F funct3=1, FORWARD — writes scalar to L0 (Pitfall 8)."""
    return act(npu, proc, insn,
                         op_id=ACT_ESUM, is_reversed=False)


@handler(kind='custom0', funct7=F7_ACT_SOFTMAX, funct3=2,
         mnemonic='softmax', mask_funct3=True)
def _softmax(npu, proc, insn, xs1, xs2):
    """SOFTMAX: funct7=0x2F funct3=2, FORWARD."""
    return act(npu, proc, insn,
                         op_id=ACT_SOFTMAX, is_reversed=False)

def softmax(arr_f16: Tensor) -> Tensor:
    return torch.nn.functional.softmax(arr_f16.to(torch.float32), dim=0).to(torch.float16)


def esum(arr_f16: Tensor, max_val: float, init_accum: float) -> Tensor:
    arr_f32 = arr_f16.to(torch.float32)
    max_val_f32 = torch.as_tensor(max_val, dtype=torch.float32)
    init_accum_f32 = torch.as_tensor(init_accum, dtype=torch.float32)
    return (init_accum_f32 + torch.sum(torch.exp(arr_f32 - max_val_f32))).to(torch.float16)

