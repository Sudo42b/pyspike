"""RegisterFile — live SPR state with typed name-based access.

GtxNpu stores SPR state as flat `dict[addr] -> int` (see `npu.py`:
`gspr`, `nspr[nest]`, `lspr[nest][spu]`). The `csr/` subpackage holds
the typed *definitions* — one `Register` instance per name, with bit
fields and bus type. Definitions are shared (one per name), values
are per-scope and per-instance (one set per NEST or per SPU).

`RegisterFile` glues the two: it owns the live dict and uses the
definitions to resolve name → address, decompose bit fields on demand,
and seed reset defaults. The underlying `dict[int, int]` is exposed
as `regs` so existing handlers that use `npu.{g,n,l}spr[addr]` keep
working.
"""
from __future__ import annotations

from typing import Dict, Iterator, Mapping, Optional, Union

from .csr import BusType
from .csr.register import Register


class RegisterFile:
    """Live SPR state for one scope (GSPR / one NSPR / one LSPR).

    Backed by `regs: dict[int, int]` (address → raw value). Lookup by
    register name resolves through the supplied typed registry; PIPE
    registers take precedence when an APB sibling shares a name.
    """

    def __init__(self, defs: Mapping[str, Register]) -> None:
        self._defs: Mapping[str, Register] = defs
        # PIPE-only address map for name-based access. APB registers
        # carry their own address space and aren't routed through PIPE
        # RDSPR/WRSPR — exclude them from name resolution.
        self._addr_by_name: Dict[str, int] = {
            name: reg.address
            for name, reg in defs.items()
            if reg.bus_type is BusType.PIPE
        }
        self.regs: Dict[int, int] = {}

    # ----- live state access (matches GtxNpu pattern) ----------------------

    def __getitem__(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            return self.regs.get(self._addr_by_name[key], 0)
        return self.regs.get(int(key), 0)

    def __setitem__(self, key: Union[int, str], value: int) -> None:
        if isinstance(key, str):
            self.regs[self._addr_by_name[key]] = int(value)
        else:
            self.regs[int(key)] = int(value)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            addr = self._addr_by_name.get(key)
            return addr is not None and addr in self.regs
        if isinstance(key, int):
            return key in self.regs
        return False

    def get(self, key: Union[int, str], default: int = 0) -> int:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[int]:
        return iter(self.regs)

    def __len__(self) -> int:
        return len(self.regs)

    # ----- typed access ----------------------------------------------------

    @property
    def defs(self) -> Mapping[str, Register]:
        """Typed register definitions (shared spec — do not mutate values)."""
        return self._defs

    def field(self, name: str, field: str) -> int:
        """Read one bit field from the live register value."""
        reg = self._defs[name]
        raw = self.regs.get(reg.address, 0)
        start, end = reg.fields[field]
        mask = (1 << (end - start + 1)) - 1
        return (raw >> start) & mask

    def set_field(self, name: str, field: str, value: int) -> None:
        """Update one bit field in the live register value."""
        reg = self._defs[name]
        addr = reg.address
        raw = self.regs.get(addr, 0)
        start, end = reg.fields[field]
        mask = (1 << (end - start + 1)) - 1
        raw &= ~(mask << start)
        raw |= (int(value) & mask) << start
        self.regs[addr] = raw

    # ----- reset -----------------------------------------------------------

    def reset(self, defaults: Optional[Mapping[int, int]] = None) -> None:
        """Clear all values and optionally seed vendor defaults."""
        self.regs.clear()
        if defaults:
            self.regs.update(defaults)
