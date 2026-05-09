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

Plan 05 GREEN-fills: per-kernel benchmarks + sweep walltime benchmark.
Acceptance: `assert benchmark.stats['mean'] * 5 <= baseline_walltime`.

Wave 0 leaves bodies as `pytest.skip(...)`. Per-task gating uses Tier 1
(parity) which runs in ~30s; this Tier 3 only runs in full-suite mode.
"""
from __future__ import annotations
import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed (dev extras)")


def test_gemm_core_benchmark(benchmark, baseline_walltime: float) -> None:
    """Per-kernel benchmark for gemm_core (~455x speedup expected per RESEARCH)."""
    pytest.skip("Plan 05 GREEN-fills per-kernel benchmark")


def test_vendor_sweep_walltime_5x(benchmark, baseline_walltime: float) -> None:
    """Full vendor 84-op sweep walltime: assert mean * 5 <= baseline."""
    pytest.skip("Plan 05 GREEN-fills sweep walltime benchmark")
