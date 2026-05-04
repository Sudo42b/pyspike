#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""GTX disasm support -- D-10 / D-13.

Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:23-36.

Helpers:
    add_r_custom0(name, funct7)             -- R-type custom0 (funct7 only)
    add_rf3_custom0(name, funct7, funct3)   -- R-type custom0 with funct3
    add_warp(name, funct3)                  -- custom1 warp control (funct3)

Each helper builds a disasm_insn_t with mask/match per the formula and the
three GPR arg formatters (gtx_xrd, gtx_xrs1, gtx_xrs2).

Import-safety: when `_riscv.so` is not available we fall back to a pure-Python
_PyDisasmInsn namedtuple-style sentinel so unit tests run offline. The real
runtime always has the binding (wheel ships it).
"""
from typing import Any, NamedTuple, Tuple

from .encoding import CUSTOM0_OPCODE, CUSTOM1_OPCODE


# --------------------------------------------------------------------------
# Bind the real disasm_insn_t / arg_t / xpr_name when _riscv.so is available;
# otherwise fall back to pure-Python sentinels for test-time inspection.
# --------------------------------------------------------------------------
_RISCV_DISASM_AVAILABLE = False

# Pure-Python fallback type (always defined so type-checkers see it).
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
    match = (funct7 << 25) | CUSTOM0_OPCODE
    mask = (0x7f << 25) | 0x7f
    return _build_insn(name, match, mask)


def add_rf3_custom0(name: str, funct7: int, funct3: int) -> Any:
    """R-type custom0 with funct3 sub-variant.

    match = (funct7 << 25) | (funct3 << 12) | 0x0b
    mask  = (0x7f << 25)   | (0x7 << 12)    | 0x7f
    """
    match = (funct7 << 25) | (funct3 << 12) | CUSTOM0_OPCODE
    mask = (0x7f << 25) | (0x7 << 12) | 0x7f
    return _build_insn(name, match, mask)


def add_warp(name: str, funct3: int) -> Any:
    """custom1 warp control: match on funct3 + opcode (funct7 ignored).

    match = (funct3 << 12) | 0x2b
    mask  = (0x7 << 12)    | 0x7f
    """
    match = (funct3 << 12) | CUSTOM1_OPCODE
    mask = (0x7 << 12) | 0x7f
    return _build_insn(name, match, mask)
