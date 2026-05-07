---
phase: 6
slug: verification-wheel
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-07
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (already installed; project uses `pytest --pylint --mypy`) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/gtx/ -x --no-header -q` |
| **Full suite command** | `pytest tests/gtx/ -v` |
| **Estimated runtime** | ~30s for full P3+P4+P5+P6 suite (~280 tests) |

Wheel + cibuildwheel verification (PKG-04) runs out-of-band:
- **Local cp310 venv smoke:** `python -m venv /tmp/spike-test && /tmp/spike-test/bin/pip install dist/spike-*.whl && /tmp/spike-test/bin/python -c "from riscv.gtx import GtxNpu; from riscv.gtx import _verify; print(_verify.bundled_elfs())"` (~30s)
- **cibuildwheel matrix:** `cibuildwheel --platform linux` (3 builds × ~5min = ~15min total; CI only)

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/gtx/ -x --no-header -q` (quick — fail-fast)
- **After every plan wave:** Run `pytest tests/gtx/ -v` (full suite, all P3+P4+P5+P6 tests)
- **Before `/gsd:verify-work`:** Full suite green + local cp310 venv smoke green
- **Before milestone close:** cibuildwheel matrix green (PKG-04)
- **Max feedback latency:** 30s for unit suite, 30s for venv smoke

---

## Per-Task Verification Map

> Tasks are placeholders — populated when planner generates PLAN.md files.
> Each plan's `<acceptance_criteria>` directly maps to "Automated Command" column.

### Wave 1a (parallel × 3 plans)

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 (VRF-01) | 1a | VRF-01 | unit | `pytest tests/gtx/test_verify_compare_hex.py -v` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 (VRF-01) | 1a | VRF-01 | smoke | `pyspike-verify --help; echo "exit=$?"` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 (VRF-01) | 1a | VRF-01 | smoke | `python -m riscv.gtx._verify --help; echo "exit=$?"` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 (atexit) | 1a | (P5 deferred) | integration | `pytest tests/gtx/test_atexit_ddr_dump.py -v` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 (atexit) | 1a | (P5 deferred) | regression | `pytest tests/gtx/test_regression_fw_act.py::test_act_strict_mode_pass -v` (now hard-PASS, no longer 5-tier skip #5) | ✅ | ⬜ pending |
| 6-03-01 | 03 (VRF-03) | 1a | VRF-03 | asset | `ls tests/gtx/data/golden/*.hex \| wc -l` (≥10) | ✅ | ⬜ pending |
| 6-03-02 | 03 (VRF-03) | 1a | VRF-03 | asset | `ls tests/gtx/data/elf/*.elf \| wc -l` (≥10) | ✅ | ⬜ pending |
| 6-03-03 | 03 (VRF-03) | 1a | VRF-03 | parse | `python scripts/import_vendor_golden.py --verify` | ❌ W0 | ⬜ pending |

### Wave 1b (1 plan)

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-04-01 | 04 (VRF-04) | 1b | VRF-04 | regression | `pytest tests/gtx/test_regression_fw_full.py -v` (zero failures, zero `within_tolerance`) | ❌ W0 | ⬜ pending |

### Wave 2 (1 plan)

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-05-01 | 05 (PKG) | 2 | PKG-01 | unit | `pytest tests/gtx/test_wheel_data_present.py -v` | ❌ W0 | ⬜ pending |
| 6-05-02 | 05 (PKG) | 2 | PKG-03 | smoke | `python -m venv /tmp/p && /tmp/p/bin/pip install dist/spike-*.whl && /tmp/p/bin/python -c "from riscv.gtx import GtxNpu; from riscv.gtx._verify import compare_hex, bundled_elfs, load_golden"` | ❌ W0 | ⬜ pending |
| 6-05-03 | 05 (PKG) | 2 | PKG-04 | matrix | `cibuildwheel --platform linux` (cp310-cp312 green, all 3) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 1a Plan 01 (or pre-wave "Plan 00" scaffold per D-17) lands ALL Wave 0 requirements as RED scaffolds:

- [ ] `tests/gtx/test_verify_compare_hex.py` — RED stubs for VRF-01 (strict/tolerant `compare_hex`, BE bit-pair, ULP edge cases)
- [ ] `tests/gtx/test_verify_cli.py` — RED stubs for `pyspike-verify --help`, `python -m riscv.gtx._verify --help`, exit codes
- [ ] `tests/gtx/test_atexit_ddr_dump.py` — RED stubs for `test_atexit_dump_fires_on_systemexit` (subprocess pyspike + GTX_DDR_DUMP env vars set + dump file exists with size = GTX_DDR_DUMP_SIZE), `test_atexit_dump_skips_when_env_unset`
- [ ] `tests/gtx/test_regression_fw_full.py` — RED parametrize stub `@pytest.mark.parametrize('elf_path', sorted(BUNDLED_ELFS))` deferring to import-time `BUNDLED_ELFS = sorted((REPO_ROOT / 'tests/gtx/data/elf').glob('*.elf'))`
- [ ] `tests/gtx/test_wheel_data_present.py` — RED stubs for `importlib.resources.files('riscv.gtx').joinpath('data/firmware').iterdir()` returns ≥1 .elf
- [ ] `scripts/import_vendor_golden.py` — RED CLI stub (entry point + argparse + pass-through `--verify` mode)
- [ ] `src/main/python/riscv/gtx/_verify.py` — RED skeleton (compare_hex re-export from `_verify_minimal` + argparse stub + main() stub)

**Existing infrastructure (no new install):**
- pytest 7.x already installed via `pytest --pylint --mypy`
- `tests/gtx/conftest.py` already provides `_RISCV_AVAILABLE` detection (P2 lineage)
- `tests/gtx/_verify_minimal.compare_hex` already provides BE bit-pair compare core (D-01 source)
- `tests/gtx/_mocks.MockProcessor` already exposes `state` property (P4 04-05 patched)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `cibuildwheel` matrix actually green on cp310/cp311/cp312 | PKG-04 | Requires Docker + manylinux2014 image (~15min, sandbox cost). | After Plan 05 commit: `cibuildwheel --platform linux 2>&1 \| tail -50` + verify last line shows `3 wheels successfully built`. |
| Local cp310 venv smoke install + one-liner | PKG-03 | Requires fresh venv + `dist/*.whl` build artifact. | After Plan 05 commit: `rm -rf /tmp/spike-test && python -m venv /tmp/spike-test && /tmp/spike-test/bin/pip install dist/spike-*.whl && /tmp/spike-test/bin/python -c "from riscv.gtx import GtxNpu; from riscv.gtx import _verify; import importlib.resources as r; assert any(p.name.endswith('.elf') for p in r.files('riscv.gtx').joinpath('data','firmware').iterdir())"; echo "exit=$?"` (must exit 0). |
| Wheel size ≤50MB | ROADMAP success #4 | Final wheel artifact requires Plan 05 build. | `ls -la dist/spike-*.whl \| awk '{print $5}'` (must be ≤52428800 bytes; 50MB = 52428800B). |
| Vendor `_ref.txt` provenance correctness | VRF-03 quality | Sub-validation that vendor source files are unmodified copies. | After Plan 03 commit: `sha256sum vendor/gtx_cpp_reference/test/<OP>/n1s16/data/n1s16_<op>_ref.txt tests/gtx/data/golden/<elf>.hex` and compare lines (after format conversion, sizes must match). |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies (planner enforces via `<acceptance_criteria>`)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (Wave 1a parallel = each plan ends with green test)
- [ ] Wave 0 covers all MISSING references (7 file scaffolds enumerated above)
- [ ] No watch-mode flags (use `pytest -x` not `pytest -f`)
- [ ] Feedback latency < 30s for unit suite, < 30s for venv smoke
- [ ] `nyquist_compliant: true` set in frontmatter (after planner produces PLANs honoring this strategy)

**Approval:** pending
