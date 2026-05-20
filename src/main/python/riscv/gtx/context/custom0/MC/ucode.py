"""
mexec	4'b1110	3'b000	gpr	gpr	3'b000	gpr	gtx op	yes	no	nsu	1	N/A	start_addr[36:0]	target_nest[63:0]	N/A	N/A	status[0]	N/A	run microcode	(count==0: until halt)
mbar	4'b1110	3'b100	rsvd	rsvd	3'b000	rsvd	gtx op	no	no	nsu	1	N/A	N/A	N/A	N/A	N/A	N/A	N/A	microcode barrier, return ready after all microcode execution completed	-
msync	4'b1110	3'b101	rsvd	rsvd	3'b000	rsvd	gtx op	no	no	nsu	1	N/A	N/A	N/A	N/A	N/A	N/A	N/A	microcode sync, hold code issue until all pending execution completed	-
eom	4'b1110	3'b111	rsvd	rsvd	3'b000	rsvd	gtx op	no	no	nsu	1	N/A	N/A	N/A	N/A	N/A	N/A	N/A	end of microcode	-

"""

# MicroCode
F7_MEXEC:int = 0b1110000       # Macro execute
F7_MBAR:int = 0b1110100        # Memory barrier (NOP)
F7_MSYNC:int = 0b1110101       # Memory sync
F7_EOM:int = 0b1110111         # End of model
F7_BAR:int = 0b1111000         # Barrier
from ...inst_handler import inst_register
from ....csr import GSPR, LSPR

@inst_register.custom0(kind='custom0', funct7=F7_MEXEC, mnemonic='mexec')
def _mexec(npu, proc, insn, xs1, xs2):
    pass

@inst_register.custom0(kind='custom0', funct7=F7_MBAR, mnemonic='mbar')
def _mbar(npu, proc, insn, xs1, xs2):
    return 0

@inst_register.custom0(kind='custom0', funct7=F7_MSYNC, mnemonic='msync')
def _msync(npu, proc, insn, xs1, xs2):
    return 0

@inst_register.custom0(kind='custom0', funct7=F7_EOM, mnemonic='eom')
def _eom(npu, proc, insn, xs1, xs2):
    return 0

@inst_register.custom0(kind='custom0', funct7=F7_BAR, mnemonic='bar')
def _bar(npu, proc, insn, xs1, xs2):
    return 0
