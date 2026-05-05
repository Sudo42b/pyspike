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
"""Tests for DMA-01 -- dma_engine helpers + constants + WarpState.wsplit_seen.

Phase 3 plan 01 Task 1 (constants/state) and Task 2 (dma_engine helpers).

Module has NO `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)` --
all helpers in dma_engine.py are spike-independent (CONTEXT D-01) and the
constants live in pure-Python modules. Tests run with --noconftest.
"""
import pytest


def test_gtx_ddr_base_constant():
    from riscv.gtx.params import GTX_DDR_BASE
    assert GTX_DDR_BASE == 0x370000000


def test_gspr_operand_addresses():
    # AUTHORITATIVE values from gtx_params.h:38-41 (orchestrator-verified).
    from riscv.gtx.encoding import (
        GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3, GSPR_GTX_OPCODE)
    assert GSPR_GTX_OPERAND1 == 0x001
    assert GSPR_GTX_OPERAND2 == 0x002
    assert GSPR_GTX_OPERAND3 == 0x003
    assert GSPR_GTX_OPCODE   == 0x004


def test_lspr_spm_addresses():
    # AUTHORITATIVE values from gtx_params.h:64-67.
    from riscv.gtx.encoding import (
        LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR)
    assert LSPR_SPM_ADDRA == 0x900
    assert LSPR_SPM_ADDRB == 0x901
    assert LSPR_SPM_ADDRC == 0x902
    assert LSPR_SPM_ADDRR == 0x903


def test_iss_dma_funct7_constants():
    from riscv.gtx.encoding import (
        GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
        GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
        GTX_ISS_F7_CREDIT_ST_CHK)
    assert (GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
            GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
            GTX_ISS_F7_CREDIT_ST_CHK) == (0x38, 0x39, 0x40, 0x41, 0x53)


def test_warp_wsplit_seen_persists_through_reset():
    from riscv.gtx.warp_state import WarpState
    w = WarpState()
    assert w.wsplit_seen is False
    w.wsplit_seen = True
    w.is_ploop = True
    w.reset()
    assert w.is_ploop is False        # normal field reset
    assert w.wsplit_seen is True      # process-lifetime sentinel persists


def test_wave0_scaffolds_exist():
    import pathlib
    td = pathlib.Path(__file__).parent
    for name in ("test_firmware_dma.py", "test_deferred_store.py",
                 "test_ddr_modes.py", "test_dma_roundtrip.py",
                 "test_dispatch_4mode.py"):
        assert (td / name).exists(), f"missing wave 0 scaffold: {name}"


def test_dma_engine_module_imports():
    pytest.skip("Filled by Task 2 -- placeholder")
