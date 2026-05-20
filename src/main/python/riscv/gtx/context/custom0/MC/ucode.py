"""Microcode control ops (custom0) — NOP in the functional model.

Port of gtx_npu_dispatch.cc:908-926. In the vendor, MEXEC runs a DDR microcode
fetch/decode/execute loop; pyspike stubs it as NOP (no test firmware emits
MEXEC). MBAR/MSYNC/EOM are sync NOPs. ``bar`` lives in :mod:`..SN.sync`.
"""
from ...inst_handler import inst_register

@inst_register.custom0(name='mexec', funct7=0b1110000)
def mexec(npu, proc, inst, cxt) -> int:
    # start_addr[36:0]	target_nest
    return 0


@inst_register.custom0(name='mbar', funct7=0b1110100)
def mbar(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='msync', funct7=0b1110101)
def msync(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='eom', funct7=0b1110111)
def eom(npu, proc, inst, cxt) -> int:
    return 0
