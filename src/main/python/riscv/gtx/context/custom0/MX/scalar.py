"""
add.vs	4'b0010	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 add (A+s) 	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
sub.vs	4'b0010	3'b000	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 sub (A-s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
mul.vs	4'b0010	3'b000	gpr	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 mult (A*s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
div.vs	4'b0010	3'b000	gpr	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 div (A/s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
fmadd.vss	4'b0010	3'b001	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value0[15:0], scalar_value1[31:16]	N/A	r2_sel[8:0]	N/A	N/A	fp16 fmadd (A*s0+s1)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
max.vs	4'b0010	3'b011	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	previous_max_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0]	fp16 max(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
min.vs	4'b0010	3'b011	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	previous_min_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	min_value[15:0]	fp16 min(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
add.is	4'b0010	3'b000	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 add (A+s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
sub.is	4'b0010	3'b000	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 sub (A-s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
mul.is	4'b0010	3'b000	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 mult (A*s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
div.is	4'b0010	3'b000	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 div (A/s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
fmadd.iss	4'b0010	3'b001	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_svr_addr_A[4:0]	scalar_value0[15:0], scalar_value1[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 fmadd (A*s0+s1) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
max.is	4'b0010	3'b011	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	previous_max_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0]	fp16 max(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
min.is	4'b0010	3'b011	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	previous_min_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	min_value[15:0]	fp16 min(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
"""

# ============================================================================
# _IMM sub-opcodes (via GSPR_OPCODE, within funct7 groups)
# ============================================================================
# Scalar _IS arith (funct7=0x54)
IMM_ADD   = 0
IMM_SUB   = 1
IMM_MUL   = 2
IMM_DIV   = 3
IMM_FMADD = 4
# Vector _II arith (funct7=0x58)
IMM_MAX   = 5
IMM_MIN   = 6
# L0 bitwise (funct7=0x5B)
IMM_AND   = 0
IMM_OR    = 1
IMM_NOT   = 2
IMM_SHIFT = 3
# Scalar Calulations
ISS_F7_SCALAR_ARITH:int = 0b0010000 # ADD/SUB/MUL/DIV scalar add.vs, sub.vs, mul.vs, div.vs, add.is, sub.is, mul.is, div.is
ISS_F7_FMADD_S:int = 0b0010001     # Scalar FMADD fmadd.vss, fmadd.iss
ISS_F7_MINMAX_S:int = 0b0010011    # Scalar MIN/MAX min.vs, max.vs, max.is, min.is
ISS_F7_DOT_SUM:int = 0b0011010     # DOT/SUM
# L0 math functions (funct7=0x5A) — same numbering as VFUNC_IMM in ISS
IMM_SQRT  = 0
IMM_EXP   = 1
IMM_LN    = 2
IMM_ABS   = 3
IMM_NEG   = 4
IMM_SIGN  = 5
IMM_STEP  = 6
IMM_CEIL  = 7
IMM_TRUNC = 8
IMM_FLOOR = 9
IMM_RNE   = 10

from ...inst_handler import inst_register
from ....csr import GSPR, LSPR
import torch

@inst_register.custom0(kind='custom0', 
                       funct7=0b0010000, 
                       funct3=0b000,
                       mnemonic='mul.vs')
def _mul_vs(npu, proc, insn, xs1, xs2):
    # mul.vs	4'b0010	3'b000	gpr	gpr	3'b010	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 mult (A*s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass


@inst_register.custom0(kind='custom0',
                       funct7=0b0010000,
                       funct3=0b110,
                       mnemonic='mul.is')
def _div_vs(npu, proc, insn, xs1, xs2):
    # div.vs	4'b0010	3'b000	gpr	gpr	3'b011	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 div (A/s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010001,
                        funct3=0b100,
                        mnemonic='fmadd.vss')
def _fmadd_vss(npu, proc, insn, xs1, xs2):
    # fmadd.vss	4'b0010	3'b001	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value0[15:0], scalar_value1[31:16]	N/A	r2_sel[8:0]	N/A	N/A	fp16 fmadd (A*s0+s1)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass


@inst_register.custom0(kind='custom0',
                        funct7=0b0010011,
                        funct3=0b000,
                        mnemonic='max.vs')
def _max_vs(npu, proc, insn, xs1, xs2):
    # max.vs	4'b0010	3'b011	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	previous_max_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0]	fp16 max(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010011,
                        funct3=0b001,
                        mnemonic='min.vs')
def _min_vs(npu, proc, insn, xs1, xs2):
    # min.vs	4'b0010	3'b011	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	previous_min_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	min_value[15:0]	fp16 min(A)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010000,
                        funct3=0b100,
                        mnemonic='add.is')
def _add_is(npu, proc, insn, xs1, xs2):
    # add.is	4'b0010	3'b000	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 add (A+s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass                    

@inst_register.custom0(kind='custom0',
                        funct7=0b0010000,
                        funct3=0b101,
                        mnemonic='sub.is')
def _sub_is(npu, proc, insn, xs1, xs2):
    # sub.is	4'b0010	3'b000	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 sub (A-s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010000,
                        funct3=0b110,
                        mnemonic='mul.is')
def _mul_is(npu, proc, insn, xs1, xs2):
    # mul.is	4'b0010	3'b000	gpr	gpr	3'b110	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 mult (A*s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010000,
                        funct3=0b111,
                        mnemonic='div.is')
def _div_is(npu, proc, insn, xs1, xs2):
    # div.is	4'b0010	3'b000	gpr	gpr	3'b111	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	scalar_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 div (A/s) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    pass

@inst_register.custom0(kind='custom0',
                        funct7=0b0010001,
                        funct3=0b100,
                        mnemonic='fmadd.iss')
def _fmadd_iss(npu, proc, insn, xs1, xs2):
# fmadd.iss	4'b0010	3'b001	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_svr_addr_A[4:0]	scalar_value0[15:0], scalar_value1[31:16]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	result[255:0]	fp16 fmadd (A*s0+s1) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
# max.is	4'b0010	3'b011	gpr	gpr	3'b100	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	previous_max_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	max_value[15:0]	fp16 max(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
# min.is	4'b0010	3'b011	gpr	gpr	3'b101	rsvd	gtx op	yes	yes	spu	3	N/A	src_SVR_addr_A[4:0]	previous_min_value[15:0]	result_SVR_addr[4:0]	r2_sel[8:0]	N/A	min_value[15:0]	fp16 min(A) imm	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]


@inst_register.custom0(kind='custom0', 
                       funct7=0b0010000, 
                       funct3=0b000,
                       mnemonic='add.vs')
def _add_vs(npu, proc, insn, xs1, xs2):
    """Scalar-vector add (funct7=0x10): FP16 vector + scalar from GPR/zero/SVR."""
    # vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 add (A+s) 	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return 0

@inst_register.custom0(kind='custom0',
                       funct7=0b0010000,
                       funct3=0b001,
                       mnemonic='sub.vs')
def _sub_vs(npu, proc, insn, xs1, xs2):
    # sub.vs	4'b0010	3'b000	gpr	gpr	3'b001	rsvd	gtx op	yes	yes	spu	3	spm_addr	vector_size[23:0]	scalar_value[15:0]	N/A	r2_sel[8:0]	N/A	N/A	fp16 sub (A-s)	r2_sel = source_sel[8:7] 00: gpr / 10: zero / 11: svr, svr_addr[6:2], svr_sub_addr[1:0]
    return 0

