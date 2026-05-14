from typing import Any, NamedTuple, Tuple

from .encoding import CUSTOM0, CUSTOM1
from typing import List
# pylint: disable=import-error,no-name-in-module
from riscv.csrs import csr_t
from riscv.decode import insn_t
from riscv.disasm import disasm_insn_t
from riscv.extension import extension_t
from riscv.processor import insn_desc_t, processor_t, illegal_instruction


class GTX_ISA(extension_t):
    """
    GTX_ISA () Instruction
        ins rd, rs1, rs2, imm2
    """

    # pylint: disable=unused-argument
    def get_instructions(self, proc: processor_t) -> List[insn_desc_t]:
        return [
            insn_desc_t(0x100b, 0xf800707f, *(self._do_th_addsl, ) * 2, *(illegal_instruction, ) * 6),
        ]

    # pylint: disable=unused-argument
    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]:
        return [
            disasm_insn_t("th.addsl", 0x100b, 0xf800707f, arg.rd, arg.rs1, arg.rs2, arg.imm2)
        ]

    # pylint: disable=unused-argument
    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []

    # pylint: disable=unused-argument
    def reset(self, proc: processor_t) -> None:
        super().reset(proc)

    def _do_th_addsl(self, p: processor_t, i: insn_t, pc: int) -> int:
        """
        reg[rd] := reg[rs1] + (reg[rs2] << imm2)
        """
        bits = int.from_bytes(i.bits, 'little')
        imm2 = (bits >> 25) & 0b11
        wdata = p.state.XPR[i.rs1] + (p.state.XPR[i.rs2] << imm2)
        p.state.XPR.write(i.rd, wdata)
        p.state.log_reg_write[i.rd << 4] = (wdata, 0)
        return pc + len(i)

_RISCV_DISASM_AVAILABLE = False

class _PyDisasmInsn(NamedTuple):
    """Offline fallback for disasm_insn_t -- holds the same surface
    (name, match, mask, args) for unit-test inspection without _riscv.so."""
    name: str
    match: int
    mask: int
    args: Tuple[Any, ...]


try:
    # pylint: disable=import-error,no-name-in-module
    from riscv.disasm import disasm_insn_t as _real_disasm_insn_t  # type: ignore
    from riscv.disasm import xpr_name as _xpr_name  # type: ignore
    from riscv import isa as _isa

    @_isa.arg
    def gtx_xrd(insn):  # pylint: disable=missing-function-docstring
        return _xpr_name[insn.rd]

    @_isa.arg
    def gtx_xrs1(insn):  # pylint: disable=missing-function-docstring
        return _xpr_name[insn.rs1]

    @_isa.arg
    def gtx_xrs2(insn):  # pylint: disable=missing-function-docstring
        return _xpr_name[insn.rs2]

    _RISCV_DISASM_AVAILABLE = True
except ImportError:
    # Sentinel arg objects (just unique markers; their .to_string is unused offline).
    class _SentinelArg:  # pylint: disable=too-few-public-methods
        """Offline arg_t stand-in -- repr only, no to_string formatter."""
        def __init__(self, name: str) -> None:
            self._name = name

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            return f"<arg:{self._name}>"

    gtx_xrd = _SentinelArg("xrd")
    gtx_xrs1 = _SentinelArg("xrs1")
    gtx_xrs2 = _SentinelArg("xrs2")


def _build_insn(name: str, match: int, mask: int) -> Any:
    """Construct either a real disasm_insn_t or the offline sentinel."""
    if _RISCV_DISASM_AVAILABLE:
        # Real binding accepts py::args (positional varargs of arg_t).
        return _real_disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)
    return _PyDisasmInsn(name, match, mask, (gtx_xrd, gtx_xrs1, gtx_xrs2))


# --------------------------------------------------------------------------
# Mask/match helpers -- gtx_npu_disasm.inc:23-36 verbatim
# --------------------------------------------------------------------------
def add_r_custom0(name: str, funct7: int) -> Any:
    """R-type custom0: match on funct7 + opcode only (mask ignores funct3).

    match = (funct7 << 25) | 0x0b
    mask  = (0x7f << 25)   | 0x7f
    """
    match = (funct7 << 25) | CUSTOM0
    mask = (0x7f << 25) | 0x7f
    return _build_insn(name, match, mask)


def add_rf3_custom0(name: str, funct7: int, funct3: int) -> Any:
    """R-type custom0 with funct3 sub-variant.

    match = (funct7 << 25) | (funct3 << 12) | 0x0b
    mask  = (0x7f << 25)   | (0x7 << 12)    | 0x7f
    """
    match = (funct7 << 25) | (funct3 << 12) | CUSTOM0
    mask = (0x7f << 25) | (0x7 << 12) | 0x7f
    return _build_insn(name, match, mask)


def add_warp(name: str, funct3: int) -> Any:
    """custom1 warp control: match on funct3 + opcode (funct7 ignored).

    match = (funct3 << 12) | 0x2b
    mask  = (0x7 << 12)    | 0x7f
    """
    match = (funct3 << 12) | CUSTOM1
    mask = (0x7 << 12) | 0x7f
    return _build_insn(name, match, mask)

from typing import Any, NamedTuple, Tuple, Union

from .encoding import *


class Instruction:
    def __init__(self, name, instruction: int) -> None:
        self._nemonic = name
        self.instruction = instruction

    @property
    def nemonic(self):
        return self._nemonic
    
    @property
    def fn7(self) -> Tuple[int, int, int]:
        # 25:31, 28:31, 25:27
        fn7 = (self.instruction & 0xFE000000) >> 25
        fn7_4 = (fn7) & 0xF
        fn7_3 = (fn7) & 0x1
        return fn7, fn7_4, fn7_3
    
    @property
    def rs2(self) -> int:
        # 20:24
        return (self.instruction & 0x1F00000) >> 20
    
    @property
    def rs1(self) -> int:
        # 15:19
        return (self.instruction & 0xF8000) >> 15
    
    @property
    def fn3(self) -> int:
        # 12:14
        return (self.instruction & 0x7000) >> 12
    
    @property
    def rd(self) -> int:
        # 7:11
        return (self.instruction & 0xF80) >> 7

    @property
    def opcode(self) -> int:
        # 0:6
        return self.instruction & 0x7F
