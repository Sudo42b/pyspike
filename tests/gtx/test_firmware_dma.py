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
"""Wave 0 scaffold for DMA-02 (firmware_dma rs1/rs2/rs3 decode + branches).

Filled by Phase 3 plan 02. Module-level _RISCV_AVAILABLE detection so that
--noconftest acceptance command still selects correctly.
"""
import pytest

try:
    import riscv.processor   # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="firmware_dma tests require _riscv.so (built by setup.py); see Plan 02",
)


def test_placeholder():
    pytest.skip("Filled by Plan 02 -- placeholder")
