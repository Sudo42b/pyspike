from __future__ import annotations

import enum
from typing import Any, Callable, Dict, Optional, Tuple, Union, overload


# ===========================================================================
# BusType
# ===========================================================================

class BusType(enum.Enum):
    PIPE = "pipe"   # 64-bit fabric (RDSPR/WRSPR path)
    APB = "apb"     # 32-bit APB (host-side debug / mmio)


# ===========================================================================
# Field — Bit-range metadata and logic (Stateless)
# ===========================================================================

class Field:
    """비트 필드의 메타데이터 및 연산 로직 보관. Stateless."""
    __slots__ = ("name", "start", "end", "mask", "shift")

    def __init__(self, name: str, start: int, end: int):
        self.name = name
        self.start = start
        self.end = end
        self.shift = start
        self.mask = (1 << (end - start + 1)) - 1

    def get_from(self, raw_val: int) -> int:
        """Raw 정수값에서 필드 값 추출"""
        return (raw_val >> self.shift) & self.mask

    def set_into(self, raw_val: int, field_val: int) -> int:
        """Raw 정수값에 필드 값 주입"""
        clean_val = int(field_val) & self.mask
        return (raw_val & ~(self.mask << self.shift)) | (clean_val << self.shift)

    def __repr__(self) -> str:
        return f"<Field {self.name} [{self.end}:{self.start}]>"


# ===========================================================================
# Bit-range marker (used inside @csr classes)
# ===========================================================================

class _Bits:
    """Declared bit range — sentinel produced by `bits()`.

    Stored as a class attribute on the @csr-decorated class; the
    decorator scans __dict__ for these and builds the field map.
    """
    __slots__ = ("start", "end", "value")

    def __init__(self, start: int, end: Optional[int] = None, value: Optional[int] = None):
        if start < 0:
            raise ValueError(f"bits(): start must be >= 0, got {start}")
        if end is not None:
            if end < start:
                raise ValueError(f"bits(): end ({end}) < start ({start})")
            self.start = start
            self.end = end
        else:
            self.start = start
            self.end = start
        self.value = value

    def __repr__(self) -> str:
        if self.start == self.end:
            return f"bits({self.start})"
        return f"bits({self.start}, {self.end})"


def bits(start: int, end: 
        Optional[int] = None, 
        value:Optional[int]=None) -> _Bits:
    """Declare a field's bit range. Inclusive endpoints.

      bits(0)        single bit at position 0       (1-bit field)
      bits(0, 9)     inclusive range [0..9]         (10-bit field)
      bits(48, 63)   inclusive range [48..63]       (16-bit field)
    """
    return _Bits(start, end, value=value)


# ===========================================================================
# Register — one CSR schema (Stateless)
# ===========================================================================

class Register:
    """A single CSR definition (Stateless Schema).

    This class no longer holds live values. It acts as metadata for
    RegisterFile to perform bit-field operations on a torch.Tensor.
    """

    def __init__(self, name: str, address: int, width: int, rw_type: str,
                 bus_type: BusType, fields: Dict[str, Field]):
        self._name = name
        self._address = address
        self._width = width
        self._rw_type = rw_type
        self._bus_type = bus_type
        self._fields = fields

    def __repr__(self) -> str:
        return f"<Register {self._bus_type.value}:{self._name} @ {self._address:#06x}>"

    def __index__(self) -> int:
        return self._address

    @property
    def name(self) -> str:        return self._name
    @property
    def address(self) -> int:     return self._address
    @property
    def width(self) -> int:       return self._width
    @property
    def rw_type(self) -> str:     return self._rw_type
    @property
    def bus_type(self) -> BusType: return self._bus_type
    @property
    def fields(self) -> Dict[str, Field]: return self._fields

    def _check_read(self) -> None:
        if self._rw_type not in ("RO", "RW"):
            raise PermissionError(f"Register {self._name} is not readable")

    def _check_write(self) -> None:
        if self._rw_type not in ("WO", "RW"):
            raise PermissionError(f"Register {self._name} is not writable")


# ===========================================================================
# @csr decorator (Handler-style)
# ===========================================================================

def csr(*, name: str, address: int, width: int, rw_type: str,
        registry: Dict[str, "Register"],
        bus_type: BusType = BusType.PIPE,
        ) -> Callable[[type], "Register"]:
    """Declare a CSR by decorating a class whose attributes use bits()."""
    if rw_type not in ("RO", "WO", "RW"):
        raise ValueError(f"@csr rw_type must be RO/WO/RW, got {rw_type!r}")
    target = registry

    def decorator(cls: type) -> "Register":
        # Iterate __dict__ to preserve declaration order (Python 3.7+).
        fields: Dict[str, Field] = {}
        for attr, val in cls.__dict__.items():
            if attr.startswith("_"):
                continue
            if isinstance(val, _Bits):
                fields[attr] = Field(attr, val.start, val.end)
        
        if not fields:
            raise ValueError(f"@csr {name}: no bits() fields declared")

        if name in target:
            raise ValueError(f"duplicate registry key {name!r}")

        reg = Register(
            name=name, address=address, width=width,
            rw_type=rw_type, bus_type=bus_type, fields=fields,
        )
        target[name] = reg
        return reg
    return decorator


def make_csr(
    registry: Dict[str, "Register"],
) -> Callable[..., Callable[[type], "Register"]]:
    """Bind the @csr decorator to a specific registry dict."""
    def _csr(*, name: str, address: int, width: int, rw_type: str,
             bus_type: BusType = BusType.PIPE
             ) -> Callable[[type], "Register"]:
        return csr(name=name, address=address, width=width, rw_type=rw_type,
                   bus_type=bus_type, registry=registry)
    return _csr


def _declare_generated_csr(
    *,
    name: str,
    address: int,
    width: int,
    rw_type: str,
    registry: Dict[str, "Register"],
    fields: Dict[str, Tuple[int, int]],
    bus_type: BusType = BusType.PIPE,
) -> Register:
    """Programmatic CSR declaration (for batches like LSPR_SGPR0..127)."""
    field_objs = {fn: Field(fn, s, e) for fn, (s, e) in fields.items()}
    reg = Register(
        name=name, address=address, width=width,
        rw_type=rw_type, bus_type=bus_type, fields=field_objs,
    )
    registry[name] = reg
    return reg
