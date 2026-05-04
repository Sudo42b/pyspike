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
"""Phase 2 hybrid mock fallback (D-17). Same test code runs with or without
_riscv.so being built."""
import pytest

try:  # noqa: SIM105
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    from riscv.extension import rocc_insn_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


@pytest.fixture
def riscv_available() -> bool:
    return _RISCV_AVAILABLE


@pytest.fixture
def proc():
    from ._mocks import MockProcessor
    return MockProcessor()


@pytest.fixture
def insn_factory():
    from ._mocks import MockInsn
    return MockInsn
