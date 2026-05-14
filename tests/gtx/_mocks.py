"""Minimal processor_t stand-in for tests/gtx/*.

Provides exactly the attribute surface GtxNpu.reset() and GtxNpu.custom0()
exercise -- nothing more. Per ORDER.md, real _riscv.so processor binding
is not loaded at test collection time; this mock lets us drive Python
dispatch paths in isolation.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Dict


class _MockXPR:
    """Mock of proc.state.XPR -- list-of-ints surface."""

    def __init__(self, n: int = 32) -> None:
        self._regs: list = [0] * n

    def __getitem__(self, idx: int) -> int:
        return self._regs[idx]

    def write(self, idx: int, val: int) -> None:
        self._regs[idx] = int(val) & 0xFFFFFFFFFFFFFFFF


class MockProcessor:
    """Minimal processor_t -- state.XPR + get_csr/put_csr only."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(XPR=_MockXPR())
        self._csrs: Dict[int, int] = {}

    def get_csr(self, addr: int) -> int:
        return self._csrs.get(addr, 0)

    def put_csr(self, addr: int, val: int) -> None:
        self._csrs[addr] = int(val) & 0xFFFFFFFFFFFFFFFF


class DummyInsn:
    """Minimal rocc_insn_t -- funct/rs1/rs2/rd/xd/xs1/xs2 only.

    Matches the attribute surface read by GtxNpu.custom0 fast-path and
    by run_pipeline -> state_decode. Defaults give a known-illegal funct7
    so run_pipeline falls through to illegal-instruction handling
    (still returns an int, which is what smoke checks).
    """

    def __init__(
        self,
        funct: int = 0,
        rs1: int = 0,
        rs2: int = 0,
        rd: int = 0,
        xd: int = 0,
        xs1: int = 0,
        xs2: int = 0,
    ) -> None:
        self.funct = funct
        self.rs1 = rs1
        self.rs2 = rs2
        self.rd = rd
        self.xd = xd
        self.xs1 = xs1
        self.xs2 = xs2
