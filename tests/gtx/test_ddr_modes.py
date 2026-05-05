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
"""Wave 0 scaffold for DMA-04 (DDR hex I/O LTR + GTX_DDR_REVERSED round-trip).

Filled by Phase 3 plan 03. NO skipif -- DDR I/O is pure-python and
GtxMemory-only (CONTEXT D-07: ddr.py is spike-independent).
"""
import pytest


def test_placeholder():
    pytest.skip("Filled by Plan 03 -- placeholder")
