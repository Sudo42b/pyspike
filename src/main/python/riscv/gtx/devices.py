"""MMIO devices bundled with the GTX extension.

Pure-Python port of vendor/gtx_cpp_reference/spike-devices/exit/sifive_exit.cc.
Firmware writes a 4-byte exit code to base+0; this terminates the simulator
with that code so the GtxNpu ``atexit`` DDR-dump hook (npu.py) fires — the same
contract the vendor gets from ``std::exit`` running C++ atexit handlers.

Wiring (firmware must write to the device base; default 0x10100000):

    pyspike --extlib=riscv.gtx --extension=gtx \
            --device=sifive_exit,0x10100000 <elf>

NOTE: ``sys.exit``/``raise SystemExit`` cannot be used here — the device
trampoline (src/main/cpp/riscv_devices.cc:40-49) catches ``py::error_already_set``
and swallows it, so SystemExit never reaches the interpreter top level. We run
the Python atexit handlers explicitly and then ``os._exit``.
"""
import atexit
import os
from typing import Optional

from riscv import dev

_PGSIZE = 0x1000


@dev.register("sifive_exit", size=_PGSIZE)
class SifiveExit(dev.MMIO):
    """MMIO exit device — write a 4-byte code to base+0 to stop the simulator."""

    def load(self, addr: int, size: int) -> bytes:
        if addr != 0 or size != 4:
            return b""   # length mismatch → trampoline reports unhandled
        return (0).to_bytes(4, "little")

    def store(self, addr: int, data: bytes) -> None:
        if addr != 0 or len(data) != 4:
            return
        code = int.from_bytes(data, "little") & 0xFF
        atexit._run_exitfuncs()   # flush GtxNpu DDR dump (+ any other hooks)
        os._exit(code)
