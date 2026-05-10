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


@pytest.fixture
def proc_with_addra_addrr_seeded():
    """Pre-seed distinct FP16 patterns at L1 ADDRA and ADDRR to prove activation
    direction asymmetry (RESEARCH Pitfall 3 / ROADMAP P5 success #2).

    Returns a callable: seed(npu, *, nest=0, spu=0, length=16,
                             addra_pattern, addrr_pattern)
    that writes addra_pattern (np.float16 array) at L1[ADDRA] and addrr_pattern
    at L1[ADDRR] and returns dict(addra=ndarray_view, addrr=ndarray_view) for
    after-call assertions.

    Implementations of activations (Plan 03) MUST overwrite ONE of these regions
    based on `is_reversed`; tests use this fixture to verify which buffer changed.

    Example usage (Plan 03 wave 1b GREEN tests will fill):
        def test_relu_forward_direction(proc_with_addra_addrr_seeded, ...):
            # forward = ADDRA -> ADDRR; only addrr region should change.
            seed = proc_with_addra_addrr_seeded
            seeded = seed(npu, addra_pattern=[1.0]*16, addrr_pattern=[-99.0]*16)
            ... run RELU ...
            # ADDRA region must be unchanged.
    """
    import numpy as np

    def seed(npu, *, nest: int = 0, spu: int = 0, length: int = 16,
             addra_pattern, addrr_pattern):
        from riscv.gtx.encoding import LSPR_SPM_ADDRA, LSPR_SPM_ADDRR
        addra = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
        addrr = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
        l1_f16 = npu.mem.l1_f16(nest, spu)
        addra_off = addra // 2
        addrr_off = addrr // 2
        l1_f16[addra_off:addra_off + length] = np.array(addra_pattern, dtype=np.float16)
        l1_f16[addrr_off:addrr_off + length] = np.array(addrr_pattern, dtype=np.float16)
        return {
            "addra": l1_f16[addra_off:addra_off + length].copy(),
            "addrr": l1_f16[addrr_off:addrr_off + length].copy(),
            "l1_f16": l1_f16,
            "addra_off": addra,
            "addrr_off": addrr,
        }

    return seed


@pytest.fixture
def _numba_available() -> bool:
    """P7 NJIT-01: True iff numba is installed (spike[fast] extra)."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def baseline_walltime() -> float:
    """P7 NJIT-06: P6 NumPy-only sweep walltime (one-shot baseline).

    Plan 05 GREEN-fills: reads from `tests/gtx/data/baseline_walltime.txt`
    (created by Plan 05 first-task baseline-recorder). Wave 0 placeholder
    returns 0.0 so test_njit_perf.py collection succeeds.

    P8 VTW-03 (D-12): file may carry leading ``#`` comment lines documenting
    the recording method (HAS_NUMBA=False venv, hardware, sweep set). The
    fixture parses the FIRST non-comment, non-empty line as the float baseline
    so callers can keep provenance metadata alongside the value.
    """
    import pathlib
    baseline_file = pathlib.Path(__file__).parent / "data" / "baseline_walltime.txt"
    if baseline_file.exists():
        for line in baseline_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return float(line)
    return 0.0  # placeholder; Plan 05 records real value
