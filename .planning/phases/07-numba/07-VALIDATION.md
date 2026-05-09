---
phase: 7
slug: numba
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-09
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Sourced from `07-CONTEXT.md` decisions D-12/D-13/D-16 + `07-RESEARCH.md` Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-benchmark 4.x |
| **Config file** | `tests/gtx/conftest.py` (existing) + new `pyproject.toml [tool.pytest.ini_options]` benchmark group |
| **Quick run command** | `pytest tests/gtx/test_njit_parity.py -v --no-cov` |
| **Full suite command** | `pytest tests/gtx/test_njit_parity.py tests/gtx/test_regression_fw_full_sweep.py tests/gtx/test_njit_perf.py -v` |
| **Estimated runtime** | ~5–15 minutes (parity ~30s + 84-op sweep ~5–10min + perf ~1–2min) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/gtx/test_njit_parity.py -v --no-cov` (~30s — 28 kernel ULP-0 check)
- **After every plan wave:** Run full suite + `pytest tests/gtx/test_regression_fw_full_sweep.py --collect-only -q` (verify 84-op parametrize collection)
- **Before `/gsd:verify-work`:** Full suite must be green + benchmark stats show `mean * 5 <= baseline_walltime` for vendor 84-op sweep
- **Max feedback latency:** 30s (parity tier alone)

---

## Per-Task Verification Map

(Plan-stage will fill exact task IDs. Skeleton below maps requirements → test tier.)

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 7-01-XX | 01 | 0 | NJIT-01 | unit | `pytest tests/gtx/test_njit_parity.py::test_has_numba_detection -v` | ❌ W0 | ⬜ pending |
| 7-01-XX | 01 | 0 | NJIT-07 | unit | `pip install -e .[fast] && python -c "from numba import njit; print(njit.__module__)"` | ❌ W0 | ⬜ pending |
| 7-02-XX | 02 | 1a | NJIT-02 + NJIT-FP32-BOUNDARY | unit | `pytest tests/gtx/test_njit_parity.py -k gemm -v` | ❌ W0 | ⬜ pending |
| 7-03-XX | 03 | 1a | NJIT-02 (vec) | unit | `pytest tests/gtx/test_njit_parity.py -k vec -v` | ❌ W0 | ⬜ pending |
| 7-04-XX | 04 | 1a | NJIT-02 + NJIT-03 (act/transcendental) | unit | `pytest tests/gtx/test_njit_parity.py -k 'gelu or tanh or sigmoid or softmax or esum' -v` | ❌ W0 | ⬜ pending |
| 7-05-XX | 05 | 1b | NJIT-04 | integration | `pytest tests/gtx/test_regression_fw_full_sweep.py -v` (84-op sweep) | ❌ W0 | ⬜ pending |
| 7-05-XX | 05 | 1b | NJIT-04 (.elf build) | integration | `cd vendor/gtx_cpp_reference/test && bash run_tests_n1s16.sh && python scripts/import_vendor_golden.py --all` | ❌ W0 | ⬜ pending |
| 7-06-XX | 06 | 2 | NJIT-06 | benchmark | `pytest tests/gtx/test_njit_perf.py -v --benchmark-only` | ❌ W0 | ⬜ pending |
| 7-06-XX | 06 | 2 | NJIT-08 | doc | `grep -q 'optional spike\[fast\]' .planning/REQUIREMENTS.md && grep -q 'base wheel size' .planning/PROJECT.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/gtx/test_njit_parity.py` — RED scaffold for 28-kernel parity (ULP-0 vs NumPy oracle)
- [ ] `tests/gtx/test_regression_fw_full_sweep.py` — RED scaffold for 84-op vendor sweep (parametrize over `vendor/gtx_cpp_reference/test/<OP>/n1s16/data/*ref.txt`)
- [ ] `tests/gtx/test_njit_perf.py` — RED scaffold for pytest-benchmark + baseline_walltime fixture
- [ ] `tests/gtx/conftest.py` — extend with `_numba_available` fixture + `baseline_walltime` fixture (P6 NumPy-only sweep walltime, captured one-shot)
- [ ] `tests/gtx/_njit_helpers.py` (or extend `_oracles.py`) — kernel inventory registry (28 entries: name → numpy_fn + njit_fn)
- [ ] `riscv/gtx/_jit.py` (or module-top per core) — `HAS_NUMBA` detection + `njit` shim that no-ops when numba absent
- [ ] `pyproject.toml` — `[project.optional-dependencies] fast = ["numba>=0.61.2,<0.66"]` + dev pytest-benchmark line
- [ ] `scripts/import_vendor_golden.py` — extend P6 lineage from ~12 ops to all 84 vendor ops; emit per-op `tests/gtx/data/golden/<op>.hex`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pip install spike[fast]` smoke (extras install + import + first JIT compile) | NJIT-07 | cibuildwheel `test-extras` automation possible but cp310-cp312 × manylinux2014 매트릭스 검증은 CI run 후 수동 확인 (1회) | `pip install dist/spike-*.whl[fast] && python -c "from riscv.gtx.gemm_core import gemm_core; import numpy as np; A = np.random.rand(16,16).astype(np.float16); B = np.random.rand(16,16).astype(np.float16); print(gemm_core(A, B).dtype)"` |
| RISC-V toolchain availability for 72 .elf builds | NJIT-04 | `/opt/riscv/` 환경 의존; CI에서는 graceful skip 분기 동작 검증 필요 | `which riscv64-unknown-elf-gcc && cd vendor/gtx_cpp_reference/test && bash run_tests_n1s16.sh 2>&1 \| tail -50` |

---

## Validation Architecture (from RESEARCH)

### Tier 1: per-kernel parity (`test_njit_parity.py`)

```bash
pytest tests/gtx/test_njit_parity.py -v --no-cov
```

**Acceptance:** 28/28 kernels pass with `np.array_equal(numpy_out.view(np.uint16), njit_out.view(np.uint16))` (delta_ulp == 0). transcendental 5 kernels (gelu/tanh_act/sigmoid/softmax/esum) pass via `numba.objmode` escape (D-09 refined).

### Tier 2: vendor 84-op sweep (`test_regression_fw_full_sweep.py`)

```bash
pytest tests/gtx/test_regression_fw_full_sweep.py -v
```

**Acceptance:**
- 84/84 vendor op directory parametrize ID visible in collection.
- 84/84 op `compare_hex(strict=True)` PASS — assets present + .elf available.
- Graceful skip on missing assets or missing `/opt/riscv/` toolchain (CI env분기).
- `assert stats['within_tolerance'] == 0` for every passing op (strict mode = exact byte match).

### Tier 3: perf benchmark (`test_njit_perf.py`)

```bash
pytest tests/gtx/test_njit_perf.py -v --benchmark-only --benchmark-warmup=on --benchmark-warmup-iterations=3
```

**Acceptance:**
- `baseline_walltime` fixture loads P6 NumPy-only sweep walltime (one-shot recorded in conftest).
- 28 per-kernel benchmarks complete + walltime sweep benchmark complete.
- `assert benchmark.stats['mean'] * 5 <= baseline_walltime` for full vendor 84-op sweep.
- Per-kernel speedup recorded; gemm_core expected ≥ 100× (research empirical 455× on small matrix), vec/act expected ≥ 5×.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (test scaffolds + extras + helpers)
- [ ] No watch-mode flags (CI must run with `--benchmark-warmup=on` not watch mode)
- [ ] Feedback latency < 30s (Tier 1 alone for per-task gating)
- [ ] `nyquist_compliant: true` set in frontmatter (after planner completes scaffold)

**Approval:** pending
