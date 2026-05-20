from typing import Any, Callable, Dict, List, Tuple

from .custom0 import CUSTOM0
from .custom1 import CUSTOM1
from .exec_st import CXT

try:
    from riscv.disasm import disasm_insn_t  # type: ignore
    from riscv.disasm import xpr_name as _xpr_name  # type: ignore
    from riscv import isa as _isa
    _RISCV_DISASM_AVAILABLE = True
except ImportError:        # offline (no _riscv C extension) — disasm is best-effort
    disasm_insn_t = None   # type: ignore
    _xpr_name = None        # type: ignore
    _isa = None             # type: ignore
    _RISCV_DISASM_AVAILABLE = False

class _PyDisasmInsn:
    __slots__ = ['name', 'match', 'mask', 'args']
    def __init__(self, name: str, match: int, mask: int, args: Tuple[Any, ...]):
        self.name, self.match, self.mask, self.args = name, match, mask, args
        

@_isa.arg
def gtx_xrd(insn): 
    return _xpr_name[insn.rd]
@_isa.arg
def gtx_xrs1(insn): 
    return _xpr_name[insn.rs1]
@_isa.arg
def gtx_xrs2(insn): 
    return _xpr_name[insn.rs2]

# RoCC rocc_insn_t bitfield: [funct7|rs2|rs1|xd|xs1|xs2|rd|opcode]
#   opcode[6:0]  = 0x0b (custom-0)
#   rd[11:7]     = destination register
#   xs2[12]      = 1 if rs2 used  ─┐
#   xs1[13]      = 1 if rs1 used  ─┤ these control register value passing!
#   xd[14]       = 1 if rd written ┘
#   rs1[19:15]   = source register 1
#   rs2[24:20]   = source register 2
#   funct[31:25] = function code (funct7) — used for sub-command dispatch
class Custom0_Insn:
    __slots__ = ['_nemonic', 'instruction']  # 파이썬 객체 생성 오버헤드 최소화
    def __init__(self, name:str, instruction: Any) -> None:
        self._nemonic = name
        self.instruction = instruction

    @property
    def nemonic(self):
        return self._nemonic
    
    @property
    def fn7(self) -> int:
        if hasattr(self.instruction, 'funct'):
            return self.instruction.funct
        return (self.instruction >> 25) & 0x7F

    @property
    def rs2(self) -> int:
        if hasattr(self.instruction, 'rs2'):
            return self.instruction.rs2
        return (self.instruction & 0x1F00000) >> 20   # bits 24:20

    @property
    def rs1(self) -> int:
        if hasattr(self.instruction, 'rs1'):
            return self.instruction.rs1
        return (self.instruction & 0xF8000) >> 15      # bits 19:15

    @property
    def fn3(self) -> int:
        """[xd | xs1 | xs2] 비트 플래그 조합 연산"""
        if hasattr(self.instruction, 'xd'):
            return (self.instruction.xd << 2) | (self.instruction.xs1 << 1) | self.instruction.xs2
        return (self.instruction >> 12) & 0x7          # bits 14:12

    @property
    def rd(self) -> int:
        if hasattr(self.instruction, 'rd'):
            return self.instruction.rd
        return (self.instruction & 0xF80) >> 7         # bits 11:7

    @property
    def opcode(self) -> int:
        if hasattr(self.instruction, 'opcode'):
            return self.instruction.opcode
        return self.instruction & 0x7F                 # bits 6:0

# ============================================================================
# custom1() — Warp control dispatch (custom-1 opcode 0x2b)
# Encoding: funct3 (bits[14:12]) selects the warp control variant:
#   funct3=000: START_T   funct3=001: END_T
#   funct3=010: START_S   funct3=011: END_S
#   funct3=100: SPLIT     funct3=101: JOIN
#   funct3=110: START_P   funct3=111: END_P
# In RoCC, bits[14:12] are {xd,xs1,xs2} flags, not funct3.
# We reconstruct funct3 from these bits and read registers directly.
# ============================================================================
class Custom1_Insn:
    __slots__ = ['_nemonic', 'instruction']  # 파이썬 객체 생성 오버헤드 최소화
    def __init__(self, name:str, instruction: Any) -> None:
        self._nemonic = name
        self.instruction = instruction

    @property
    def nemonic(self):
        return self._nemonic
    
    @property
    def imm12(self):
        # bits 31:20 → imm_valid (bit 30), imm_id (bits 25:20).
        # Object form: bits 31:25 = funct, 24:20 = rs2 ⇒ imm12 = funct<<5 | rs2.
        raw = self.instruction
        if hasattr(raw, 'funct'):
            imm12 = (raw.funct << 5) | raw.rs2
        else:
            imm12 = (raw & 0xFFF00000) >> 20
        return (imm12 >> 10) & 0x1, imm12 & 0x3F

    @property
    def rs1(self) -> int:
        if hasattr(self.instruction, 'rs1'):
            return self.instruction.rs1
        return (self.instruction & 0xF8000) >> 15      # bits 19:15

    @property
    def rs2(self) -> int:
        if hasattr(self.instruction, 'rs2'):
            return self.instruction.rs2
        return (self.instruction & 0x1F00000) >> 20    # bits 24:20

    @property
    def fn3(self) -> int:
        if hasattr(self.instruction, 'xd'):
            return (self.instruction.xd << 2) | (self.instruction.xs1 << 1) | self.instruction.xs2
        return (self.instruction & 0x7000) >> 12       # bits 14:12

    @property
    def rd(self) -> int:
        if hasattr(self.instruction, 'rd'):
            return self.instruction.rd
        return (self.instruction & 0xF80) >> 7         # bits 11:7

    @property
    def opcode(self) -> int:
        if hasattr(self.instruction, 'opcode'):
            return self.instruction.opcode
        return self.instruction & 0x7F                 # bits 6:0

"""@handler decorator + dispatch-table builders.

Extended with `context=` parameter (NpuContext | tuple | None) for Style C
per-context dispatch — see ORDER.md and npu_context.py.
"""

from .exec_st import CXT

class I_Handler:
    def __init__(self):
        # Custom0 레지스트리: (context, funct7, funct3) -> handler
        self._c0_funcs: Dict[Tuple[int, int], Callable[[Custom0_Insn], None]] = {}
        # Custom1 레지스트리: (context, funct3) -> handler
        self._c1_funcs: Dict[Tuple[int], Callable[[Custom1_Insn], None]] = {}
        # (name, match, mask, type_flag_0_or_1)
        self._disasm_meta: List[Tuple[str, int, int, int]] = []  # (name, match, mask, args)

    def custom0(self, name:str, funct7: int, funct3: int = 0):
        def decorator(func: Callable):
            func.mnemonic = name
            self._c0_funcs[(funct7, funct3)] = func
            """R-type custom0 with funct3 sub-variant.

            match = (funct7 << 25) | (funct3 << 12) | 0x0b
            mask  = (0x7f << 25)   | (0x7 << 12)    | 0x7f
            """
            match = (funct7 << 25) | (funct3 << 12) | CUSTOM0
            mask = (0x7f << 25) | (0x7 << 12) | 0x7f
            meta = (name, match, mask, 0)
            if meta not in self._disasm_meta:
                self._disasm_meta.append(meta)
            return func

        return decorator

    def custom1(self, name:str, funct3: int):
        def decorator(func: Callable):
            func.mnemonic = name
            self._c1_funcs[(funct3)] = func
            match = (funct3 << 12) | CUSTOM1
            mask = (0x7 << 12) | 0x7f
            meta = (name, match, mask, 1)
            if meta not in self._disasm_meta:
                self._disasm_meta.append(meta)
            return func
        return decorator

    def get_custom0_handler(self, cxt: CXT, funct7: int, funct3: int):
        return self._c0_funcs.get((cxt, funct7, funct3))

    def get_custom1_handler(self, cxt: CXT, funct3: int):
        return self._c1_funcs.get((cxt, funct3))
    
    def collect_disasms(self) -> List[disasm_insn_t]:
        """Collect disasm_insn_t entries for all registered instructions."""
        disasm_entrys = []
        for name, match, mask, type_flag in self._disasm_meta:
            if _RISCV_DISASM_AVAILABLE:
                entry = disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)
            else:
                entry = _PyDisasmInsn(name, match, mask, (gtx_xrd, gtx_xrs1, gtx_xrs2))
            disasm_entrys.append(entry)
        return disasm_entrys

inst_register = I_Handler()


