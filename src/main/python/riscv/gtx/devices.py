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

from .config_params import DEFAULT_DDR_SIZE, L1_SIZE_BYTES

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


# ============================================================================
# gtx_spm — debug-only bus window over the per-(nest, spu) L1 scratchpad.
#
# Per D-05 the L1 SPM is NOT bus-registered in production HW (vendor
# cpp_reference, gtx_npu_core.cc:111-112 — "Create L1 shadow buffer for
# internal sync (NOT bus-registered). Per D-05: CPU cannot access L1 via
# bus."). The full SystemC ISS *does* expose L1 on the bus, and golden
# outputs for kernels that touch raw SPM addresses (notably LOG, which
# patches negative→QNAN through ``(volatile uint16_t*)BANK_C/BANK_R``)
# come from there.
#
# Without this device, pyspike faults on the host load → trap → mtvec=0
# → spin (the LOG hang). With it, host CPU loads/stores in the L1
# address range route to the active NPU's L1 bank at the current (nest,
# spu); outside P/T-loops that defaults to (0, 0).
#
# Address mapping
# ---------------
# Spike rejects ``device_base == 0`` (sim.cc:277 assert), so we map the
# device at PA = ``_SPM_BASE`` and shift the device-relative ``addr`` back
# into the absolute L1 offset (= the value the kernel literally writes as
# BANK_C/BANK_R). _SPM_BASE is page-aligned and small enough that the LOG
# kernel's banks (0x30000 BANK_C, 0x50000 BANK_R) fall inside the window;
# the bottom ``_SPM_BASE`` bytes of L1 are not host-visible (kernels do
# not host-access that range).
#
# Opt-in via:  pyspike ... --device=gtx_spm,0x10000
# (Spike defaults already occupy 0x1000–0x2000; 0x10000 is safely free.)
# ============================================================================
_SPM_BASE = 0x10000  # MUST match the --device=gtx_spm,<base> argument.


@dev.register("gtx_spm", size=L1_SIZE_BYTES - _SPM_BASE)
class GtxSpm(dev.MMIO):
    """Bus window over the active NPU's L1 SPM bank (debug-only)."""

    def _bank(self):
        # Imported lazily to avoid a devices↔npu import cycle at module load.
        from .npu import get_active_npu
        npu = get_active_npu()
        if npu is None:
            return None
        warp = npu.warp
        nest = warp.current_nest if warp.is_ploop else 0
        spu = warp.current_spu if warp.is_tloop else 0
        return npu.mem.l1_byte(nest, spu)

    def load(self, addr: int, size: int) -> bytes:
        bank = self._bank()
        l1_off = addr + _SPM_BASE
        if bank is None or l1_off < 0 or l1_off + size > bank.size:
            return b""
        return bytes(bank[l1_off:l1_off + size])

    def store(self, addr: int, data: bytes) -> None:
        bank = self._bank()
        l1_off = addr + _SPM_BASE
        if bank is None or l1_off < 0 or l1_off + len(data) > bank.size:
            return
        bank[l1_off:l1_off + len(data)] = np.frombuffer(
            bytearray(data), dtype=np.uint8
        )


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
