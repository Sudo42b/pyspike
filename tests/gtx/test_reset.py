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
"""Tests for CORE-02 -- GtxNpu.reset() sp init + zero-init + SPR defaults.

All tests require _riscv.so (since GtxNpu inherits from isa.ROCC which
needs the rocc_t binding). Skipif-gated via module-level detection so
the suite works under `pytest ... --noconftest -o "addopts="`.
"""
import numpy as np
import pytest

from riscv.gtx.params import GTX_NEST_NUM, GTX_SPU_NUM


# Module-level detection -- self-contained so --noconftest still works.
try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _RISCV_AVAILABLE,
    reason="_riscv.so not built -- GtxNpu requires rocc_t base",
)


def _make_npu():
    """Lazy import: only resolves when _riscv.so is built."""
    from riscv.gtx import GtxNpu
    return GtxNpu()


def _make_proc():
    from tests.gtx._mocks import MockProcessor
    return MockProcessor()


def test_reset_initializes_sp():
    """CORE-02: sp = 0x80100000 after reset (gtx_npu_core.cc:144-156)."""
    npu = _make_npu()
    proc = _make_proc()
    npu.reset(proc)
    assert proc.get_state().XPR[2] == 0x80100000


def test_reset_zeros_mxe_accum():
    """CORE-02 + D-06 corrected: shape (GTX_NEST_NUM, GTX_SPU_NUM) FP32, all zero."""
    npu = _make_npu()
    proc = _make_proc()
    # Pre-write garbage
    npu._mxe_accum.fill(123.0)
    npu.reset(proc)
    assert npu._mxe_accum.shape == (GTX_NEST_NUM, GTX_SPU_NUM)
    assert npu._mxe_accum.dtype == np.float32
    assert np.all(npu._mxe_accum == 0.0)


def test_reset_zeros_memory_arrays():
    """CORE-02: L0/L1/L2 ndarray all zero after reset."""
    npu = _make_npu()
    proc = _make_proc()
    npu.mem._l0_bytes.fill(0xFF)
    npu.mem._l1_bytes.fill(0xFF)
    npu.mem._l2_bytes.fill(0xFF)
    npu.reset(proc)
    assert np.all(npu.mem._l0_bytes == 0)
    assert np.all(npu.mem._l1_bytes == 0)
    assert np.all(npu.mem._l2_bytes == 0)


def test_reset_seeds_nspr_defaults():
    """gtx_npu_core.cc:80-109: NSPR_THREAD_MASK=0xFFFF, NSPR_TYPE=1 (FP16)."""
    npu = _make_npu()
    proc = _make_proc()
    npu.reset(proc)
    for n in range(GTX_NEST_NUM):
        assert npu.nspr[n][0x400] == 0xFFFF, f"NEST {n} THREAD_MASK"
        assert npu.nspr[n][0x402] == 1, f"NEST {n} TYPE (FP16)"


def test_reset_seeds_lspr_defaults():
    """gtx_npu_core.cc:80-109: LSPR_SPM_ADDRA..ADDRR=0 for every (NEST, SPU)."""
    npu = _make_npu()
    proc = _make_proc()
    npu.reset(proc)
    for n in range(GTX_NEST_NUM):
        for s in range(GTX_SPU_NUM):
            for addr in (0x900, 0x901, 0x902, 0x903):
                assert npu.lspr[n][s][addr] == 0, f"NEST {n} SPU {s} addr {hex(addr)}"


def test_reset_clears_warp_state():
    """CORE-02: WarpState reset to all-default values."""
    npu = _make_npu()
    proc = _make_proc()
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = 3
    npu.warp.curr_id = 7
    npu.reset(proc)
    assert npu.warp.is_ploop is False
    assert npu.warp.is_tloop is False
    assert npu.warp.is_sloop is False
    assert npu.warp.tmu_id == 0
    assert npu.warp.curr_id == 0


def test_reset_clears_gspr():
    """CORE-02: gspr cleared, then defaults seeded."""
    npu = _make_npu()
    proc = _make_proc()
    npu.gspr[0xABCD] = 0xFEED   # garbage
    npu.reset(proc)
    assert 0xABCD not in npu.gspr   # cleared
    # Defaults seeded (gtx_npu_core.cc:80-109)
    assert npu.gspr[0x000] == 0
    assert npu.gspr[0x010] == 0


def test_reset_fpu_enable_does_not_crash():
    """reset() must call put_csr(0x300, mstatus) without raising even when
    the binding's get_csr is not exposed. The try/except guards this."""
    npu = _make_npu()
    proc = _make_proc()
    # MockProcessor.get_csr returns 0; put_csr is no-op. No assertion needed --
    # just confirm reset() did not raise.
    npu.reset(proc)
