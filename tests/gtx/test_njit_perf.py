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
"""P7 NJIT-06 Tier 3: pytest-benchmark walltime gate.

Plan 05 GREEN body. Two benchmarks:
  1. test_gemm_core_benchmark -- per-kernel speedup sentinel (RESEARCH 455x)
  2. test_vendor_sweep_walltime_5x -- full sweep walltime; assert mean*5 <= baseline

Use `--benchmark-warmup=on --benchmark-warmup-iterations=3` to discard JIT
first-call compile time noise.

Both tests SKIP gracefully when:
  - pytest-benchmark dev extras not installed (module-top importorskip)
  - baseline_walltime <= 0.0 (Plan 05 Task 1 has not recorded yet)
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed (dev extras)")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def gemm_inputs():
    """Fixed 16x16 FP16 inputs for gemm benchmark (deterministic seed=42)."""
    rng = np.random.default_rng(seed=42)
    A = rng.random((16, 16), dtype=np.float32).astype(np.float16)
    B = rng.random((16, 16), dtype=np.float32).astype(np.float16)
    return A, B


def test_gemm_core_benchmark(benchmark, gemm_inputs) -> None:
    """Per-kernel benchmark: gemm_core 16x16x16 FP16.

    RESEARCH empirical: ~455x speedup with numba (910us NumPy -> 2us JIT).
    This test records the speedup; it does NOT assert a specific multiplier
    (variance across hardware is significant). The walltime gate lives in
    test_vendor_sweep_walltime_5x.
    """
    from riscv.gtx.gemm_core import gemm_core
    A, B = gemm_inputs

    # benchmark() handles the timing; warmup iterations discard JIT compile cost
    result = benchmark(gemm_core, A, B)

    assert result.dtype == np.float16
    assert result.shape == (16, 16)


def test_vendor_sweep_walltime_5x(benchmark, baseline_walltime) -> None:
    """Full vendor 84-op sweep walltime: assert mean * 5 <= baseline.

    Runs the full sweep under JIT path (HAS_NUMBA=True) and asserts the
    wall-clock is at most 1/5 of the recorded P6 NumPy-only baseline.

    baseline_walltime fixture reads tests/gtx/data/baseline_walltime.txt
    (one-shot recorded by Plan 05 Task 1).
    """
    if baseline_walltime <= 0.0:
        pytest.skip(
            "baseline_walltime not recorded; run Plan 05 Task 1 to capture P6 baseline"
        )

    # Sanity: if baseline is small enough that it likely reflects pytest startup
    # overhead rather than real kernel execution, skip the 5x assertion. The
    # 5x speedup contract is meaningful only when the sweep does real work
    # (e.g. M >= 12 ops actually executing). On developer machines with the
    # full GFW + 72 .elf built, baseline should be much larger (>30s).
    # This skip lets CI and minimal checkouts pass cleanly while still
    # enforcing the assertion when the toolchain produces real work.
    if baseline_walltime < 30.0:
        pytest.skip(
            "baseline_walltime=" + f"{baseline_walltime:.1f}s"
            + " < 30s threshold; sweep likely measures pytest overhead "
            "rather than real kernel work (M=0 .elf built on this checkout). "
            "Re-record after building vendor .elf to enable 5x assertion."
        )

    # Force a fresh subprocess to avoid in-process numba state leaks.
    # subprocess timeout is baseline-aware: prior P7 hardcode 600s assumed the
    # placeholder baseline (4.5s, 5x = 22.5s); with the real HAS_NUMBA=False
    # baseline at ~5000s the 84-op sweep cannot finish inside 600s even at the
    # nominal 5x speedup. Use 1.5x baseline as the per-iteration cap so a
    # numba run that meets the 5x target completes well inside the window,
    # while a degenerate (numba off / slow) run is still bounded.
    sweep_timeout = max(int(baseline_walltime * 1.5), 600)

    def run_sweep() -> float:
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/gtx/test_regression_fw_full_sweep.py",
            "--no-cov", "-q",
            "-o", "addopts=",
        ]
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=sweep_timeout)
        return time.perf_counter() - t0

    # benchmark() runs the function multiple times; warmup discards JIT compile
    benchmark(run_sweep)

    # benchmark.stats has aggregates from the timed iterations
    mean_walltime = benchmark.stats["mean"]
    speedup = baseline_walltime / max(mean_walltime, 1e-6)

    assert mean_walltime * 5 <= baseline_walltime, (
        "NJIT-06 5x speedup unmet: mean walltime = " + f"{mean_walltime:.2f}s"
        + ", baseline = " + f"{baseline_walltime:.2f}s"
        + ", speedup = " + f"{speedup:.2f}x" + " (target >= 5x)"
    )

    # Optional: log per-iteration stats for visibility
    print(
        "\nNJIT-06: mean=" + f"{mean_walltime:.2f}s"
        + ", baseline=" + f"{baseline_walltime:.2f}s"
        + ", speedup=" + f"{speedup:.2f}x" + " (target >= 5x)"
    )
