#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Mocks for unit tests that must run without _riscv.so being built (D-19/D-20).

INTERNAL: never imported by riscv.gtx production code.
Plans 02-05 reuse these via `from ._mocks import MockProcessor, MockInsn`.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockXPR:
    _regs: List[int] = field(default_factory=lambda: [0] * 32)

    def __getitem__(self, i: int) -> int:
        return self._regs[i]

    def write(self, i: int, val: int) -> None:
        if i != 0:  # x0 is hardwired zero
            self._regs[i] = val & 0xFFFFFFFFFFFFFFFF


@dataclass
class MockState:
    XPR: MockXPR = field(default_factory=MockXPR)


@dataclass
class MockProcessor:
    _state: MockState = field(default_factory=MockState)

    # Plan 04-05 fix: real pybind11 processor_t exposes `state` as a
    # def_property_readonly (py_module.cc:711), NOT a `get_state()` method.
    # MockProcessor exposes BOTH so unit tests that already use get_state()
    # keep passing while production source code uses the real binding (proc.state).
    @property
    def state(self) -> MockState:
        return self._state

    def get_state(self) -> MockState:
        return self._state

    def get_csr(self, which: int) -> int:
        return 0

    def put_csr(self, which: int, val: int) -> None:
        pass


@dataclass
class MockInsn:
    """Mirrors rocc_insn_t fields exposed by py_module.cc:391-409."""
    opcode: int = 0x0b
    rd: int = 0
    xs2: int = 0
    xs1: int = 0
    xd: int = 0
    rs1: int = 0
    rs2: int = 0
    funct: int = 0  # this is funct7
