"""BM-01 unit tests: xp alias + helpers + fail-loud + DEVICE deprecation.

Note (Option-A deferral, 2026-05-18 user decision): the original Wave 0 plan
required `from riscv.gtx.config_params import DEVICE` to raise ImportError. That
clean-cut is deferred to Wave 3 (plan 09-03-finalize) per CONTEXT.md line 232,
because downstream files still import DEVICE and removing it now would break
the wave-end smoke gate. Test 4 below therefore asserts the deprecated alias
exists (string 'cpu' under xp=numpy) instead of asserting ImportError. The
ImportError assertion becomes a Wave 3 acceptance criterion.
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


def test_device_symbol_deprecated_alias_present():
    """D-04 (Option-A Wave 0/Wave 3 deferral): DEVICE retained as deprecated
    string alias until Wave 3 clean-cut. See module docstring + CONTEXT line 232.

    Wave 3 will flip this to `pytest.raises(ImportError)`."""
    from riscv.gtx.config_params import DEVICE, xp
    assert isinstance(DEVICE, str), (
        f"DEVICE must be a string alias under Option-A deferral, "
        f"got {type(DEVICE).__name__}={DEVICE!r}"
    )
    expected = "cpu" if xp.__name__ == "numpy" else "cuda"
    assert DEVICE == expected, (
        f"DEVICE={DEVICE!r} does not match xp-backed expected={expected!r}"
    )
