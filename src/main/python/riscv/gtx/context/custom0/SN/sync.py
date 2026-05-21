"""Sync / barrier ops (custom0).

Port of the SystemC golden dispatch (vendor/simulator GTX_extension.h:1608-1623).
In the ISS these are nop / debug only — DMA is instantaneous, single hart,
sequential — so they carry no architectural effect:

  BAR / HALT / FLUSH — nop in simulation (GTX_extension.h:1615-1617).
  INTR               — nop (GTX_extension.h:1622).
  WAIT               — ISS memory-dump debug command (GTX_extension.h:1608);
                       pyspike dumps DDR via the GTX_DDR_DUMP atexit hook instead,
                       so it stays a nop here.

The golden warns when a sync command is cast inside a split/join region
(GTX_extension.h:1618-1620); we mirror that warning. All return 0.
"""
import sys

from ...inst_handler import inst_register
from ...exec_st import CXT


def _warn_if_in_loop(npu, name: str) -> None:
    """The golden rejects sync commands inside split/join (a plan/loop nest,
    i.e. CONTEXT != C1). Mirror its warning — no functional effect."""
    if npu.CONTEXT is not CXT.C1:
        print(f"[WARNING] GTX SYNC COMMAND ({name}) can't cast in split join",
              file=sys.stderr, flush=True)


@inst_register.custom0(name='bar', funct7=0b1111000, funct3=0)
def bar(npu, proc, inst, cxt) -> int:
    _warn_if_in_loop(npu, 'bar')
    return 0


@inst_register.custom0(name='wait', funct7=0b1111001, funct3=0)
def wait(npu, proc, inst, cxt) -> int:
    # ISS memory-dump debug command; pyspike dumps DDR via GTX_DDR_DUMP atexit.
    return 0


@inst_register.custom0(name='intr', funct7=0b1111011, funct3=0)
def intr(npu, proc, inst, cxt) -> int:
    return 0


@inst_register.custom0(name='flush', funct7=0b1111100, funct3=0)
def flush(npu, proc, inst, cxt) -> int:
    _warn_if_in_loop(npu, 'flush')
    return 0


@inst_register.custom0(name='halt', funct7=0b1111111, funct3=0)
def halt(npu, proc, inst, cxt) -> int:
    _warn_if_in_loop(npu, 'halt')
    return 0
