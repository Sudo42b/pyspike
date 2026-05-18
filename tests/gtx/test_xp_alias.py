"""BM-01 unit tests: xp alias + helpers + fail-loud + DEVICE clean-cut.

Phase 9 Wave 6 (plan 09-03-finalize) closed the D-04 DEVICE clean-cut deferred
from Wave 0 (Option-A user decision 2026-05-18). All downstream consumers
(memory.py, register_file.py, npu.py, dma_engine.py, ops/*.py) ported off
`device=DEVICE` in Waves 1/2/5; the symbol is now removed from
config_params.py and the re-export from riscv.gtx.__init__.py is gone too.
Test 4 below asserts `ImportError` per the Wave 6 acceptance contract.
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
import pytest


def _spawn_python(env_overrides: dict, code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_xp_default_is_numpy():
    """D-01/D-02: xp resolves to numpy when GTX_USE_CUDA unset."""
    env = {k: v for k, v in os.environ.items() if k != "GTX_USE_CUDA"}
    proc = subprocess.run(
        [sys.executable, "-c",
         "from riscv.gtx.config_params import xp; print(xp.__name__)"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert proc.stdout.strip() == "numpy"


def test_to_host_to_device_identity_when_numpy():
    """D-12: helpers are no-ops under xp=numpy (literal identity)."""
    from riscv.gtx.config_params import to_host, to_device, xp
    assert xp.__name__ == "numpy", "test environment must be numpy"
    arr = np.array([1, 2, 3], dtype=np.int32)
    assert to_host(arr) is arr
    assert to_device(arr) is arr


def test_gtx_use_cuda_without_cupy_fails_loud():
    """D-03: silent fallback forbidden; RuntimeError with pip-install hint."""
    try:
        import cupy  # noqa: F401
        pytest.skip("cupy IS installed; cannot test fail-loud path")
    except ImportError:
        pass
    proc = _spawn_python(
        {"GTX_USE_CUDA": "1"},
        "from riscv.gtx.config_params import xp",
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "RuntimeError" in combined
    assert "pip install 'spike[cuda]'" in combined


def test_device_symbol_removed():
    """D-04 Wave 6 clean-cut: DEVICE symbol removed from config_params.py.

    The Option-A Wave 0/Wave 3 deferral is closed. Both
    `from riscv.gtx.config_params import DEVICE` and `from riscv.gtx import
    DEVICE` raise ImportError. All downstream consumers were ported off
    `device=DEVICE` kwarg in Waves 1/2/5."""
    with pytest.raises(ImportError):
        from riscv.gtx.config_params import DEVICE  # noqa: F401
    with pytest.raises(ImportError):
        from riscv.gtx import DEVICE  # noqa: F401
