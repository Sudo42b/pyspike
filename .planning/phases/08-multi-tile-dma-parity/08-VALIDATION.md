---
phase: 8
slug: multi-tile-dma-parity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-10
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (already installed; conftest + MockProcessor land in P3/P4/P5) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` + `tests/gtx/conftest.py` |
| **Quick run command** | `pytest tests/gtx/test_multi_tile_dma.py -v --no-cov` |
| **Full suite command** | `pytest tests/gtx/test_regression_fw_full_sweep.py tests/gtx/test_multi_tile_dma.py tests/gtx/test_njit_perf.py -v --no-cov` |
| **Estimated runtime** | ~30–90 seconds (multi-tile unit ~2s, vendor sweep ~30–60s, perf benchmark ~30s) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/gtx/test_multi_tile_dma.py -v --no-cov` (always available after Wave 0)
- **After every plan wave:** Run full suite command above
- **Before `/gsd:verify-work`:** Full suite green; vendor sweep `M ≥ 12` PASS; `test_vendor_sweep_walltime_5x` not skipped
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

> Filled in by gsd-planner. Each task in PLAN.md must reference a row here.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD     | TBD  | 0    | MTDMA-03    | unit      | `pytest tests/gtx/test_multi_tile_dma.py::test_tile_boundary_state_reset -v` | ❌ W0 | ⬜ pending |
| TBD     | TBD  | 0    | MTDMA-01/VTW-01 | regression | `pytest tests/gtx/test_regression_fw_full_sweep.py -k 'abs or add_vv or relu' -v` | ✅ | ⬜ pending |
| TBD     | TBD  | 0    | VTW-04      | asset     | `python scripts/import_vendor_golden.py --all --check` | ✅ (extension) | ⬜ pending |
| TBD     | TBD  | 1    | MTDMA-01/02 | regression | `GTX_DDR_REVERSED=1 pytest tests/gtx/test_regression_fw_full_sweep.py -v` | ✅ | ⬜ pending |
| TBD     | TBD  | 1    | MTDMA-04    | unit      | `pytest tests/gtx/test_multi_tile_dma.py::test_state_reset_audit -v` | ❌ W0 | ⬜ pending |
| TBD     | TBD  | 2    | VTW-03      | benchmark | `pytest tests/gtx/test_njit_perf.py::test_vendor_sweep_walltime_5x --benchmark-only` | ✅ | ⬜ pending |
| TBD     | TBD  | 2    | VTW-04      | doc-check | `test -f tests/gtx/data/firmware/README.md && grep -E 'BE FP16\|GTX_DDR_REVERSED\|_find_elf' tests/gtx/data/firmware/README.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/gtx/test_multi_tile_dma.py` — RED scaffolds for MTDMA-03/04 (tile-boundary state reset audit, ABS compute, MockProcessor)
- [ ] `scripts/import_vendor_golden.py` — `--all` flag extension (84-op iteration; preserve existing 9-op mapping)
- [ ] `tests/gtx/data/firmware/README.md` — VTW-04 contract scaffold (4 sections per D-08; can land empty body in W0, fill in W2)
- [ ] `tests/gtx/test_regression_fw_full_sweep.py:_find_elf` — multi-path search extension (D-05) with vendor-path candidate
- [ ] `tests/gtx/data/golden/<op>.hex` (84 entries) — golden hex bundle generated via `import_vendor_golden.py --all`
- [ ] `tests/gtx/data/baseline_walltime.txt` — placeholder if missing (D-12 will overwrite in Wave 2 under HAS_NUMBA=False)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| HAS_NUMBA=False baseline rerecording | VTW-03 | Requires deactivating numba in venv (env-level, not pytest-level); needs human to confirm walltime is realistic before commit | 1) `python -m pip uninstall numba` (in fresh venv) 2) `pytest tests/gtx/test_regression_fw_full_sweep.py --benchmark-only --benchmark-warmup=on > tests/gtx/data/baseline_walltime.txt` 3) Verify file > 30s threshold |
| Vendor `.elf` directory presence | MTDMA-01, VTW-01 | Local dev path `/mnt/e/14_NIGHTLY/pyspike/test/` only exists on dev machine; CI cannot fully exercise without `GTX_VENDOR_TEST_DIR` | 1) `ls /mnt/e/14_NIGHTLY/pyspike/test/<OP>/n1s16/n1s16_<op>.elf` 2) Cross-check `find /mnt/e/14_NIGHTLY/pyspike/test/ -name 'n1s16_*.elf' \| wc -l` reports ~79 |
| `wheel size ≤ 50MB after firmware/ exclusion` | VTW-04 (D-07) | Wheel build is environment-sensitive; verify post-build artifact | 1) `python -m build --wheel` 2) `du -h dist/*.whl` 3) Confirm ≤ 50MB |
| diff-matrix audit (vendor C++ ↔ pyspike Python) | MTDMA-04 (D-02) | Reading 7 vendor `.cc` files line-by-line is slow + judgement-heavy | 1) Read `08-RESEARCH.md` §"Vendor C++ ↔ pyspike Python Diff Matrix" 2) Confirm verdicts MATCH on every row |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (test_multi_tile_dma.py, README.md, golden bundle, _find_elf patch)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
