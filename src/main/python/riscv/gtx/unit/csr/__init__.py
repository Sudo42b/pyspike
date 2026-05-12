"""GTX CSR (Control & Status Register) subpackage.

Address scope encoding (PIPE bus, bits [11:10]):
    00  Global   (GSPR)  0x000-0x3FF  — single instance, shared
    01  Shared   (NSPR)  0x400-0x7FF  — one per nest
    10  Local    (LSPR)  0x800-0xBFF  — one per SPU (per nest × spu pair)
    11  SYSTEM           0xC00-0xFFF  — reserved

Bits [9:8]: 0x = architecture, 10 = performance, 11 = debug & H/W feature.
Bit  [7]  : 0 = Read/Write, 1 = Read-Only.

APB bus has its own flat 32-bit address space (no scope encoding); APB
registers are co-located with their semantic-scope PIPE counterparts
inside the same scope file.

Files:
    register.py — Register class, @csr decorator, bits() helper, make_csr factory
    gspr.py     — GSPR (Global)  scope
    nspr.py     — NSPR (Nest)    scope (PIPE + APB)
    lspr.py     — LSPR (Local)   scope (PIPE + APB)

Public API:
    GSPR / NSPR / LSPR              — raw per-scope dicts (PIPE + APB mixed)
    CSR_GSPR / CSR_NSPR / CSR_LSPR  — PIPE-only views (for RDSPR/WRSPR routing)
    ALL                              — union view across all scopes
    Register, BusType, bits, csr, make_csr  — re-exported from register.py
"""
from typing import Dict

from .register import Register, BusType, bits, csr, make_csr
from .gspr import GSPR
from .nspr import NSPR
from .lspr import LSPR


# ---------------------------------------------------------------------------
# PIPE-only views (RDSPR/WRSPR use the PIPE bus; APB is debug-side only)
# ---------------------------------------------------------------------------

CSR_GSPR: Dict[str, Register] = {
    name: reg for name, reg in GSPR.items() if reg.bus_type is BusType.PIPE
}
CSR_NSPR: Dict[str, Register] = {
    name: reg for name, reg in NSPR.items() if reg.bus_type is BusType.PIPE
}
CSR_LSPR: Dict[str, Register] = {
    name: reg for name, reg in LSPR.items() if reg.bus_type is BusType.PIPE
}


# ---------------------------------------------------------------------------
# Union view across scopes (for name-based lookup without scope guess)
# ---------------------------------------------------------------------------

ALL: Dict[str, Register] = {**GSPR, **NSPR, **LSPR}


# ---------------------------------------------------------------------------
# Address-based lookup helpers (used by RDSPR/WRSPR handlers)
# ---------------------------------------------------------------------------

def find_by_address(address: int, bus_type: BusType = BusType.PIPE
                    ) -> Register:
    """Return the Register at `address` on `bus_type`. Raises KeyError otherwise.

    For PIPE, scopes the lookup by address range to disambiguate quickly.
    For APB, scans the union (APB has no scope bits in its address).
    """
    if bus_type is BusType.PIPE:
        if address < 0x400:
            pool = CSR_GSPR
        elif address < 0x800:
            pool = CSR_NSPR
        elif address < 0xC00:
            pool = CSR_LSPR
        else:
            raise KeyError(f"PIPE address {address:#06x} in SYSTEM range — reserved")
        for reg in pool.values():
            if reg.address == address:
                return reg
        raise KeyError(f"No PIPE register at address {address:#06x}")
    # APB
    for reg in ALL.values():
        if reg.bus_type is BusType.APB and reg.address == address:
            return reg
    raise KeyError(f"No APB register at address {address:#06x}")


__all__ = [
    "Register",
    "BusType",
    "bits",
    "csr",
    "make_csr",
    "GSPR",
    "NSPR",
    "LSPR",
    "CSR_GSPR",
    "CSR_NSPR",
    "CSR_LSPR",
    "ALL",
    "find_by_address",
]
