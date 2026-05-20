"""Sync / barrier ops (custom0) — NOP in the functional model.

Port of gtx_npu_dispatch.cc:915-926: BAR/WAIT/INTR/FLUSH/HALT are sync/control
ops with no architectural effect in the ISS (DMA is instantaneous, single
hart, sequential execution). All emitted with funct3=0; all return 0.
"""
from ...inst_handler import inst_register


@inst_register.custom0(name='bar', funct7=0b1111000, funct3=0)
def bar(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='wait', funct7=0b1111001, funct3=0)
def wait(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='intr', funct7=0b1111011, funct3=0)
def intr(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='flush', funct7=0b1111100, funct3=0)
def flush(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='halt', funct7=0b1111111, funct3=0)
def halt(npu, proc, inst, cxt) -> int:
    return 0
