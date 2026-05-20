"""RegisterFile — Tensor-backed SPR state with broadcasting.

GtxNpu stores SPR state in `RegisterFile` instances. Each `RegisterFile`
owns a `torch.Tensor` storage.

Shapes:
    GSPR:  (1024,)
    NSPR:  (NEST, 1024)
    LSPR:  (NEST, SPU, 1024)

The last dimension is always the 10-bit address offset (0-1023).
Broadcasting is supported: setting a value on a multi-dimensional
RegisterFile propagates to all instances.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Union, Tuple

import torch
from . import BusType, Register, Field


class RegisterFile:
    """Live SPR state storage using torch.Tensor.

    Supports indexing to narrow down dimensions (e.g. lspr[nest][spu]).
    Attributes provide access to registers by name, returning a View
    that supports bit-field manipulation and broadcasting.
    """

    def __init__(self, 
                 defs: Mapping[str, Register], 
                 shape: Tuple[int, ...] = (1024,),
                 device: str = "cpu",
                 tensor: Optional[torch.Tensor] = None) -> None:
        self._defs = defs
        # last dim must be 1024 for address space
        if shape[-1] != 1024:
            raise ValueError(f"RegisterFile last dim must be 1024, got {shape[-1]}")

        if tensor is not None:
            self._tensor = tensor
        else:
            self._tensor = torch.zeros(shape, dtype=torch.int64, device=device)

        # Mapping for fast address lookup
        self._addr_by_name = {
            name: reg.address & 0x3FF
            for name, reg in defs.items()
            if reg.bus_type is BusType.PIPE
        }

    @property
    def tensor(self) -> torch.Tensor:
        """The underlying storage tensor."""
        return self._tensor

    @property
    def defs(self) -> Mapping[str, Register]:
        return self._defs

    # ----- Slicing / Indexing ----------------------------------------------

    def __getitem__(self, key: Union[int, str, slice]) -> Union[int, RegisterFile]:
        """Index into dimensions (if any) or access raw address (if key is int)."""
        if isinstance(key, int) and self._tensor.ndim == 1:
            # Raw address access for 1D (GSPR)
            return int(self._tensor[key & 0x3FF])
        
        if isinstance(key, str):
            # Name-based register access
            return getattr(self, key)

        # Dimension narrowing
        sub_tensor = self._tensor[key]
        if sub_tensor.ndim == 0:
            return int(sub_tensor)
        
        return RegisterFile(self._defs, sub_tensor.shape, tensor=sub_tensor)

    def __setitem__(self, key: Union[int, str], value: Any) -> None:
        if isinstance(key, str):
            setattr(self, key, value)
            return
        
        # Raw address write (modulo 1024 for scope)
        addr = int(key) & 0x3FF
        self._tensor[..., addr] = value

    # ----- Attribute access (Register names) -------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in self._addr_by_name:
            reg = self._defs[name]
            addr = self._addr_by_name[name]
            # Return a view of this specific register across all dimensions
            return RegisterView(reg, self._tensor[..., addr])

        # Support for adjacent ranges like SGPR (e.g. lspr.SGPR)
        # If user asks for 'SGPR', and we have SGPR0..127, we could return a batch view.
        # For now, we'll stick to exact name matches.
        raise AttributeError(f"RegisterFile has no register or attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if name in self._addr_by_name:
            reg = self._defs[name]
            addr = self._addr_by_name[name]
            # Decompose if value is int, or broadcast if value is tensor
            self._tensor[..., addr] = value
            return
        super().__setattr__(name, value)

    # ----- Utility ---------------------------------------------------------
    def reset(self, defaults: Optional[Mapping[int, int]] = None) -> None:
        """Clear all values and optionally seed vendor defaults."""
        self._tensor.zero_()
        if defaults:
            for addr, val in defaults.items():
                self._tensor[..., addr & 0x3FF] = val

    def __iter__(self) -> Iterator[Union[RegisterFile, int]]:
        """Iterate over the first dimension, yielding sub-views or values."""
        if self._tensor.ndim <= 1:
            # For 1D (GSPR or narrowed view), iterate over raw values
            for i in range(self._tensor.shape[0]):
                yield int(self._tensor[i])
        else:
            # For multi-dimensional, yield narrowed RegisterFile views
            for i in range(self._tensor.shape[0]):
                yield self[i]

    def get(self, key: Union[int, str], default: int = 0) -> int:
        """Compatibility method for dict-like access."""
        try:
            val = self[key]
            return int(val) if not isinstance(val, RegisterFile) else default
        except (KeyError, AttributeError):
            return default

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self) -> str:
        return f"RegisterFile(shape={tuple(self._tensor.shape)})"


class RegisterView:
    """Proxy for one or more instances of a specific Register.

    Attributes provide access to bit fields.
    Setting a field broadcasts the value across all instances in this view.
    """
    __slots__ = ("_reg", "_tensor")

    def __init__(self, reg: Register, tensor: torch.Tensor):
        self._reg = reg
        self._tensor = tensor  # Shape matches rf.dimensions (e.g. (), (N,), (N, S))

    def __getattr__(self, name: str) -> Union[int, torch.Tensor]:
        if name == "value":
            return self._tensor if self._tensor.ndim > 0 else int(self._tensor)
        if name in self._reg.fields:
            field = self._reg.fields[name]
            val = (self._tensor >> field.shift) & field.mask
            return val if self._tensor.ndim > 0 else int(val)
        
        raise AttributeError(f"Register {self._reg.name} has no field {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        
        if name == "value":
            self._tensor.copy_(torch.as_tensor(value, dtype=torch.int64))
            return

        if name in self._reg.fields:
            field = self._reg.fields[name]
            mask = field.mask
            shift = field.shift

            # Reinterpret the shifted mask as a signed int64 to avoid
            # Python's arbitrary-precision negative result from
            # `~(mask << shift)` — torch cannot cast that back into
            # int64 (OverflowError). See CONTEXT.md root_cause.
            u64 = (mask << shift) & ((1 << 64) - 1)
            shifted_mask = u64 - (1 << 64) if u64 >> 63 else u64

            # Same signed-int64 wrap for `value` when it is a Python int
            # with the top bit set (e.g. 0xCAFEBABEDEADBEEF) — torch
            # rejects unsigned ≥ 2^63 in int64 dtype.
            if isinstance(value, int):
                v64 = value & ((1 << 64) - 1)
                value = v64 - (1 << 64) if v64 >> 63 else v64

            new_val = torch.as_tensor(value, dtype=torch.int64) & mask
            self._tensor.copy_((self._tensor & ~shifted_mask) | (new_val << shift))
            return
        
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"<RegisterView {self._reg.name} shape={tuple(self._tensor.shape)}>"

    def __int__(self) -> int:
        if self._tensor.ndim > 0:
            raise TypeError("Cannot convert multi-dimensional RegisterView to int")
        return int(self._tensor)

    # @overload
    # def __getitem__(self, key: int) -> int: ...
    def __getitem__(self, key: Any) -> Any:
        # If it's a bit index
        if isinstance(key, int):
            return int((self._tensor >> key) & 1)
        return getattr(self, key)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            bit = 1 << key
            if value:
                self._tensor |= bit
            else:
                self._tensor &= ~bit
            return
        setattr(self, key, value)
