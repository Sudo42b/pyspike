"""GTX CSR (Control & Status Register) declarations — handler-style.

Mirrors the @handler pattern from _registry.py: declare a register with
the @csr decorator on a class whose attributes use bits() to mark bit
ranges. The decorator builds a Register instance, registers it in GSPR,
and returns the instance (so the class name binds to the live register).

    @csr(name="STACK_INFO", address=0x010, width=64, rw_type="RW")
    class STACK_INFO:
        pointer = bits(0, 37)
        size    = bits(48, 63)

Access fields by attribute or item:

    STACK_INFO.pointer = 0x123        # set field
    GSPR["STACK_INFO"]["pointer"]     # equivalent (lookup-by-name)
    val = STACK_INFO.pointer          # read field
    raw = STACK_INFO.value            # composed register integer
    STACK_INFO.value = 0xABCD         # decompose into fields

Bit-range syntax:

    bits(0)        — single bit at position 0      (1-bit field)
    bits(0, 9)     — inclusive range [0..9]        (10-bit field)
    bits(48, 63)   — inclusive range [48..63]      (16-bit field)
"""
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
# Bit-range marker (used inside @csr classes)
# ===========================================================================

class _Bits:
    """Declared bit range — sentinel produced by `bits()`.

    Stored as a class attribute on the @csr-decorated class; the
    decorator scans __dict__ for these and builds the field map.
    """
    __slots__ = ("start", "end")

    def __init__(self, start: int, end: Optional[int] = None):
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

    def __repr__(self) -> str:
        if self.start == self.end:
            return f"bits({self.start})"
        return f"bits({self.start}, {self.end})"


def bits(start: int, end: Optional[int] = None) -> _Bits:
    """Declare a field's bit range. Inclusive endpoints.

      bits(0)        single bit at position 0       (1-bit field)
      bits(0, 9)     inclusive range [0..9]         (10-bit field)
      bits(48, 63)   inclusive range [48..63]       (16-bit field)
    """
    return _Bits(start, end)


# ===========================================================================
# Register — one CSR with named fields
# ===========================================================================

class Register:
    """A single CSR. Built by @csr; do not construct directly.

    Field access:
        reg.pointer = 0x123          # set (attribute)
        val = reg.pointer            # read (attribute)
        reg["pointer"] = 0x123       # set (item — equivalent)
        val = reg["pointer"]         # read (item — equivalent)

    Whole-register access:
        reg.value                    # composed integer from all fields
        reg.value = 0xABCD           # decompose into field values
        int(reg)                     # register address (for SPR routing)
    """
    # Internal-only attributes use a leading underscore. __setattr__ checks
    # the field map before delegating to object's default behavior, so
    # `_field_map` etc. must be set via super().__setattr__ during __init__.

    def __init__(self, name: str, address: int, width: int, rw_type: str,
                 bus_type: BusType, fields: Dict[str, Tuple[int, int]]):
        if rw_type not in ("RO", "WO", "RW"):
            raise ValueError(f"Register {name}: rw_type must be RO/WO/RW, got {rw_type!r}")
        if width <= 0:
            raise ValueError(f"Register {name}: width must be > 0, got {width}")
        for fname, (start, end) in fields.items():
            if end >= width:
                raise ValueError(
                    f"Register {name}: field {fname} bit {end} exceeds width {width}"
                )
        super().__setattr__("_name", name)
        super().__setattr__("_address", address)
        super().__setattr__("_width", width)
        super().__setattr__("_rw_type", rw_type)
        super().__setattr__("_bus_type", bus_type)
        super().__setattr__("_field_map", dict(fields))
        super().__setattr__("_values", {fn: 0 for fn in fields})

    # --- Identity / repr ----------------------------------------------------

    def __repr__(self) -> str:
        return f"<Register {self._bus_type.value}:{self._name} @ {self._address:#06x}>"

    def __index__(self) -> int:
        return self._address

    # --- Public properties --------------------------------------------------

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
    def fields(self) -> Dict[str, Tuple[int, int]]:
        """Field map snapshot (copy — does not allow in-place mutation)."""
        return dict(self._field_map)

    # --- Field access -------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is called only when normal lookup fails, so all real
        # instance attrs (_name etc.) are unaffected. Field access lives here.
        # Returns Any so callers using `reg.field` don't trip Pylance when
        # they subsequently subscript or call methods on the result (field
        # values are int at runtime but may be used as plain ints, hex
        # strings via format(), etc.).
        if name.startswith("_"):
            raise AttributeError(name)
        field_map = self.__dict__.get("_field_map", {})
        if name in field_map:
            self._check_read()
            return self.__dict__["_values"][name]
        raise AttributeError(f"Register {self.__dict__.get('_name', '?')}: no field {name!r}")

    def __setattr__(self, name: str, value: int) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        # Class-level descriptors (e.g., `value` property) take precedence —
        # custom __setattr__ would otherwise mask their setters.
        descriptor = type(self).__dict__.get(name)
        if isinstance(descriptor, property):
            if descriptor.fset is None:
                raise AttributeError(f"property {name!r} on Register is read-only")
            descriptor.__set__(self, value)
            return
        field_map = self.__dict__["_field_map"]
        if name in field_map:
            self._check_write()
            start, end = field_map[name]
            mask = (1 << (end - start + 1)) - 1
            self.__dict__["_values"][name] = int(value) & mask
            return
        raise AttributeError(f"Register {self._name}: no field {name!r}")

    @overload
    def __getitem__(self, key: str) -> int: ...
    @overload
    def __getitem__(self, key: int) -> int: ...
    def __getitem__(self, key: Union[str, int]) -> int:
        """Field access by name (str) or bit access by position (int).

            reg["pointer"]   → composed field value (int)
            reg[5]           → bit at position 5      (0 or 1)
        """
        if isinstance(key, int):
            self._check_read()
            return (self.value >> key) & 1
        return self.__getattr__(key)

    @overload
    def __setitem__(self, key: str, value: int) -> None: ...
    @overload
    def __setitem__(self, key: int, value: int) -> None: ...
    def __setitem__(self, key: Union[str, int], value: int) -> None:
        """Field set by name (str) or bit set by position (int).

            reg["pointer"] = 0x123  → set field
            reg[5] = 1              → set bit 5
        """
        if isinstance(key, int):
            self._check_write()
            cur = self.value
            if value:
                self.value = cur | (1 << key)
            else:
                self.value = cur & ~(1 << key)
            return
        self.__setattr__(key, value)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._field_map

    # --- Whole-register value ----------------------------------------------

    @property
    def value(self) -> int:
        """Composed register value (all fields merged into one integer)."""
        self._check_read()
        v = 0
        for fname, (start, end) in self._field_map.items():
            mask = (1 << (end - start + 1)) - 1
            v |= (self._values[fname] & mask) << start
        return v & ((1 << self._width) - 1)

    @value.setter
    def value(self, raw: int) -> None:
        """Set raw register value; decompose into field values."""
        self._check_write()
        raw = int(raw) & ((1 << self._width) - 1)
        for fname, (start, end) in self._field_map.items():
            mask = ((1 << (end - start + 1)) - 1) << start
            self._values[fname] = (raw & mask) >> start

    # --- Internal -----------------------------------------------------------

    def _check_read(self) -> None:
        if self._rw_type not in ("RO", "RW"):
            raise PermissionError(
                f"Register {self._name} is not readable (rw_type={self._rw_type})"
            )

    def _check_write(self) -> None:
        if self._rw_type not in ("WO", "RW"):
            raise PermissionError(
                f"Register {self._name} is not writable (rw_type={self._rw_type})"
            )

# ===========================================================================
# @csr decorator (Handler-style)
# ===========================================================================

def csr(*, name: str, address: int, width: int, rw_type: str,
        registry: Dict[str, "Register"],
        bus_type: BusType = BusType.PIPE,
        ) -> Callable[[type], "Register"]:
    """Declare a CSR by decorating a class whose attributes use bits().

    Args:
      name:     registry key (REQUIRED — also returned by Register.name).
      address:  register address (int).
      width:    register width in bits (typically 32 or 64).
      rw_type:  "RO" | "WO" | "RW".
      registry: target dict — REQUIRED so scope (GSPR/NSPR/LSPR) is explicit.
                Use `make_csr(SCOPE_DICT)` in each scope file to pre-bind
                this and avoid repeating `registry=...` per @csr call.
      bus_type: BusType.PIPE (default) or BusType.APB.

    Returns the Register instance — the decorated symbol in the enclosing
    module is rebound to the live Register.

    Mirrors @handler from _registry.py: declarative, typed, explicit
    registry.
    """
    if rw_type not in ("RO", "WO", "RW"):
        raise ValueError(f"@csr rw_type must be RO/WO/RW, got {rw_type!r}")
    target = registry

    def decorator(cls: type) -> "Register":
        # Iterate __dict__ to preserve declaration order (Python 3.7+).
        fields: Dict[str, Tuple[int, int]] = {}
        for attr, val in cls.__dict__.items():
            if attr.startswith("_"):
                continue
            if isinstance(val, _Bits):
                fields[attr] = (val.start, val.end)
        if not fields:
            raise ValueError(f"@csr {name}: no bits() fields declared")

        if name in target:
            raise ValueError(
                f"@csr duplicate registry key {name!r} "
                f"(existing: {target[name]!r})"
            )

        reg = Register(
            name=name,
            address=address,
            width=width,
            rw_type=rw_type,
            bus_type=bus_type,
            fields=fields,
        )
        target[name] = reg
        return reg
    return decorator


def make_csr(
    registry: Dict[str, "Register"],
) -> Callable[..., Callable[[type], "Register"]]:
    """Bind the @csr decorator to a specific registry dict.

    Used once at the top of each scope file (gspr/nspr/lspr.py) so the
    per-declaration call site stays terse:

        GSPR: Dict[str, Register] = {}
        csr = make_csr(GSPR)

        @csr(name="STACK_INFO", address=0x010, width=64, rw_type="RW")
        class STACK_INFO:
            pointer = bits(0, 37)
            size    = bits(48, 63)
    """
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
    """Programmatic CSR declaration (for batches like LSPR_SGPR0..127).

    Equivalent to writing a @csr-decorated class manually; emits the same
    Register and registers it.
    """
    namespace = {
        field_name: bits(start, end)
        for field_name, (start, end) in fields.items()
    }
    generated = type(name, (), namespace)
    return csr(
        name=name,
        address=address,
        width=width,
        rw_type=rw_type,
        bus_type=bus_type,
        registry=registry,
    )(generated)


