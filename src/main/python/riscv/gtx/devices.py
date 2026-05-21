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

import numpy as np

from riscv import dev

from .config_params import DEFAULT_DDR_SIZE

_PGSIZE = 0x1000


# ============================================================================
# gtx_ddr — DDR as CPU-visible system memory on the Spike bus.
#
# The RISC-V CPU reaches DDR through the bus (load/store routed here); we proxy
# to the process-wide DDR buffer (memory.get_ddr()) — the same bytes the NPU
# uses — so CPU and NPU share one DDR. The buffer is Python-owned, so it
# outlives the C++ sim teardown and the atexit DDR dump stays valid (a
# C++-owned buffer use-after-frees there).
#
# CPU DDR access is sparse (firmware param reads / small result writes); bulk
# DDR I/O is the NPU's, straight on the torch buffer — so the per-access Python
# load/store cost here is negligible.
#
# Wiring:  pyspike ... --device=gtx_ddr,0x370000000
# ============================================================================
@dev.register("gtx_ddr", size=DEFAULT_DDR_SIZE)
class GtxDdr(dev.MMIO):
    """Bus window over the shared NPU DDR (offset-relative load/store)."""

    def load(self, addr: int, size: int) -> bytes:
        from .memory import get_ddr
        ddr = get_ddr()
        ddr.ensure(addr + size)
        return np.ascontiguousarray(ddr.read(addr, size)).tobytes()

    def store(self, addr: int, data: bytes) -> None:
        from .memory import get_ddr
        ddr = get_ddr()
        ddr.ensure(addr + len(data))
        ddr.write(addr, np.frombuffer(bytearray(data), dtype=np.uint8))


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
