"""
im2col.n	4'b0001	3'b000	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	row_A_size[15:0], col_A_size[31:16]	kernal_size[4:0], dialate[9:8], stride[31:16], #_channel[47:32]	N/A	N/A	N/A	N/A	im2col for normal convolution
im2col.d	4'b0001	3'b001	gpr	gpr	3'b000	rsvd	gtx op	yes	yes	spu	3	spm_addr	row_A_size[15:0], col_A_size[31:16]	kernal_size[4:0], dialate[9:8], stride[31:16], #_channel[47:32]	N/A	N/A	N/A	N/A	im2col for depth-wise convolution
"""
# GTX_F7_WRSPR/GTX_F7_RDSPR per vendor convention.
F7_IM2COL_N: int = 0b0001000  # IM2COL normal
F7_IM2COL_D: int = 0b0001001  # IM2COL depthwise

from ...inst_handler import inst_register
from ....csr import GSPR, LSPR
import torch


@inst_register.custom0(kind='custom0', funct7=F7_IM2COL_N, mnemonic='im2col.n')
def _im2col_n(npu, proc, insn, xs1, xs2):
     return _im2col(npu, proc, insn, is_depthwise=False)
 
@inst_register.custom0(kind='custom0', funct7=F7_IM2COL_D, mnemonic='im2col.d')
def _im2col_d(npu, proc, insn, xs1, xs2):
    return _im2col(npu, proc, insn, is_depthwise=True)