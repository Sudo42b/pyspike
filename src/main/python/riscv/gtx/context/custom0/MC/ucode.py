"""Microcode control ops (custom0).

Port of the SystemC golden: ``NSU::mexec`` (vendor/simulator/src/NSU.cpp:1406)
and the MEXEC/MSYNC/EOM dispatch in GTX_extension.h:1503-1525.

  MEXEC  — DDR microcode VM: fetch 32-byte words, decode each, and re-dispatch
           it through the custom0 registry, looping until EOM (opcode 0x3b8).
  MBAR   — nop in the ISS (GTX_extension.h:1614 "nop in simulation").
  MSYNC  — ISS-internal; the golden only warns and no-ops (GTX_extension.h:1515).
  EOM    — microcode terminator; consumed inside MEXEC. Emitted standalone it is
           ISS-internal, so the golden warns and no-ops (GTX_extension.h:1521).

NOTE: no committed test firmware emits MEXEC, so this VM is a faithful port of
the SystemC golden but is currently unexercised by the regression suite.
"""
import sys

from ...inst_handler import inst_register
from ...disasm import Custom0_Insn
from ...exec_st import CXT
from ....config_params import DDR_BASE
from ....csr import GSPR

_OPERAND3 = GSPR['GSPR_GTX_OPERAND3'].address
_OPERAND5 = GSPR['GSPR_GTX_OPERAND5'].address
_EOM_OPCODE = 0x3B8          # op_GTX_Codes EOM — terminates the microcode loop
_MEXEC_MAX_STEPS = 1 << 20   # safety bound against a missing EOM terminator


# ---------------------------------------------------------------------------
# Operand bridge — the microcode word carries operand *values* (rdata[0..2]),
# while the registry handlers read register *indices* via proc.state.XPR. These
# proxies hand a handler its rs1/rs2 values without mutating the real CPU state.
# ---------------------------------------------------------------------------
class _McodeState:
    __slots__ = ('_real', '_xpr')

    def __init__(self, real_state, rs1_val: int, rs2_val: int):
        self._real = real_state
        self._xpr = {1: rs1_val, 2: rs2_val}

    @property
    def XPR(self):
        return self

    def __getitem__(self, idx):
        if idx in self._xpr:
            return self._xpr[idx]
        return int(self._real.XPR[idx])


class _McodeProc:
    __slots__ = ('_real', 'state')

    def __init__(self, real_proc, rs1_val: int, rs2_val: int):
        self._real = real_proc
        self.state = _McodeState(real_proc.state, rs1_val, rs2_val)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _McodeInsn:
    """Duck-typed RoCC insn for one microcode word. rs1/rs2 are stub indices
    (1/2) that ``_McodeState`` resolves to the microcode operand values."""
    __slots__ = ('funct', 'rs1', 'rs2', 'rd', 'xd', 'xs1', 'xs2')

    def __init__(self, funct7: int, funct3: int, rd: int):
        self.funct = funct7
        self.rs1 = 1
        self.rs2 = 2
        self.rd = rd
        self.xd = (funct3 >> 2) & 1
        self.xs1 = (funct3 >> 1) & 1
        self.xs2 = funct3 & 1


def _context_for(gflag: bool, sflag: bool, lflag: bool) -> CXT:
    """Map microcode loop flags → npu.CONTEXT (mirror of WarpState derivation)."""
    if not gflag:
        return CXT.C1
    if lflag:
        return CXT.C3
    if sflag:
        return CXT.C2
    return CXT.C4


@inst_register.custom0(name='mexec', funct7=0b1110000)
def mexec(npu, proc, inst, cxt) -> int:
    """MEXEC — DDR microcode VM (port of NSU::mexec).

    rs1 = microcode start address (DDR). Each 32-byte word packs four 64-bit
    little-endian values ``rdata[0..3]``; ``rdata[3]`` holds the control fields:
        opcode  = bits[9:0]    (op_GTX_Codes = funct7<<3 | funct3)
        op_sel  = bits[18:10]
        lflag   = bit 19 (t-loop)  sflag = bit 20 (s-loop)  gflag = bit 21 (plan)
        spu_id  = bits[27:22]      nest_id = bits[33:28]
    ``rdata[0..2]`` are the rs1/rs2/rs3 operand values. Each decoded op runs
    through the custom0 registry until opcode == 0x3b8 (EOM).
    """
    raddr = int(proc.state.XPR[inst.rs1])
    ddr_off = (raddr - DDR_BASE) if raddr >= DDR_BASE else raddr

    saved_ctx = npu.CONTEXT
    saved_nest = npu.warp.current_nest
    saved_spu = npu.warp.current_spu
    status = 0
    try:
        for _ in range(_MEXEC_MAX_STEPS):
            npu.mem.ensure_ddr(ddr_off + 32)
            word = npu.mem.ddr.read(ddr_off, 32)
            rdata = [int.from_bytes(bytes(word[i * 8:i * 8 + 8].tolist()), 'little')
                     for i in range(4)]
            ddr_off += 32

            ctrl = rdata[3]
            opcode = ctrl & 0x3FF
            if opcode == _EOM_OPCODE:
                break

            funct7, funct3 = opcode >> 3, opcode & 0x7
            func = inst_register._c0_funcs.get((funct7, funct3))
            if func is None:
                continue

            # Stage rs3 + op_sel where handlers read them; set the warp scope.
            npu.gspr[_OPERAND3] = rdata[2]
            npu.gspr[_OPERAND5] = (ctrl >> 10) & 0x1FF
            npu.CONTEXT = _context_for(bool((ctrl >> 21) & 1),
                                       bool((ctrl >> 20) & 1),
                                       bool((ctrl >> 19) & 1))
            npu.warp.current_nest = (ctrl >> 28) & 0x3F
            npu.warp.current_spu = (ctrl >> 22) & 0x3F

            minsn = Custom0_Insn(getattr(func, 'mnemonic', 'mcode'),
                                 _McodeInsn(funct7, funct3, rdata[2] & 0x1F))
            result = func(npu, _McodeProc(proc, rdata[0], rdata[1]),
                          minsn, npu.CONTEXT) or 0
            status = max(status, result)
        else:
            print("[mexec] no EOM within step bound — aborting microcode loop",
                  file=sys.stderr, flush=True)
    finally:
        npu.CONTEXT = saved_ctx
        npu.warp.current_nest = saved_nest
        npu.warp.current_spu = saved_spu
    return status


@inst_register.custom0(name='mbar', funct7=0b1110100)
def mbar(npu, proc, inst, cxt) -> int:
    # nop in the ISS; the golden groups MBAR with the sync barriers and warns if
    # it is cast inside split/join (GTX_extension.h:1614-1620).
    if npu.CONTEXT is not CXT.C1:
        print("[WARNING] GTX SYNC COMMAND (mbar) can't cast in split join",
              file=sys.stderr, flush=True)
    return 0


@inst_register.custom0(name='msync', funct7=0b1110101)
def msync(npu, proc, inst, cxt) -> int:
    # ISS-internal; the golden warns and no-ops (GTX_extension.h:1515).
    print("[WARNING] MSYNC should be generated by ISS", file=sys.stderr, flush=True)
    return 0


@inst_register.custom0(name='eom', funct7=0b1110111)
def eom(npu, proc, inst, cxt) -> int:
    # Consumed inside MEXEC; emitted standalone it is ISS-internal (GTX_ext:1521).
    print("[WARNING] EOM should be generated by ISS", file=sys.stderr, flush=True)
    return 0
