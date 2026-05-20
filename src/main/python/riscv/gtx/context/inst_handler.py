"""Handler-registration re-exports.

The registry singleton and instruction wrappers live in :mod:`.disasm`; this
module re-exports them so handler packages can ``from ...inst_handler import
inst_register`` without reaching into the disasm internals.
"""
from .disasm import Custom0_Insn, Custom1_Insn, inst_register

__all__ = ["inst_register", "Custom0_Insn", "Custom1_Insn"]
