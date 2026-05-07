---
phase: 06-verification-wheel
plan: 05
subsystem: packaging
tags: [setuptools, build_py, package-data, manifest, importlib-resources, cibuildwheel, wheel-bundle, gtx]

# Dependency graph
requires:
  - phase: 06-verification-wheel/01
    provides: "riscv.gtx._verify production module + bundled_elfs() / load_golden() helpers + [project.scripts] pyspike-verify"
  - phase: 06-verification-wheel/03
    provides: "tests/gtx/data/elf/*.elf (12 firmware) + tests/gtx/data/golden/*.hex (11 goldens) — single source-of-truth (D-13)"
  - phase: 06-verification-wheel/04
    provides: "test_regression_fw_full.py parametrized strict-mode regression — validates assets before they ship"
provides:
  - "setup.py build_py.build_package_data() extension that copies tests/gtx/data/{elf,golden} to <pkg>/gtx/data/{firmware,golden} at wheel build time"
  - "pyproject.toml [tool.setuptools.package-data] 'riscv.gtx' globs (data/firmware/*.elf, data/golden/*.hex)"
  - "pyproject.toml [tool.cibuildwheel] test-command extension: helper-API import smoke + pyspike-verify --help (PKG-04)"
  - "MANIFEST.in sdist coverage for tests/gtx/data/elf/*.elf|*.S|Makefile + tests/gtx/data/golden/*.hex"
  - "tests/gtx/test_wheel_data_present.py 4 GREEN tests (importlib.resources + helper API smoke; skip-if-editable, PASS-if-wheel)"
affects: [phase-07-optimization, release-engineering, downstream-pip-install-spike]

# Tech tracking
tech-stack:
  added: [shutil (stdlib, newly imported in setup.py)]
  patterns:
    - "tests/-as-source-of-truth + build-time-copy-to-src/ (D-13): tests/gtx/data is canonical; setup.py build_py mirrors a renamed (elf->firmware) snapshot into the wheel package tree at build time, preserving the src/ tree as wheel-only sync target"
    - "package-data via setuptools quoted-key syntax: '\"riscv.gtx\" = [...]' for sub-packages whose name contains a dot"
    - "Editable-vs-wheel test polymorphism: helper _gtx_data_dir() returns None when riscv/gtx/data/ is absent so tests gracefully skip in editable installs and PASS hard in wheel-installed venvs without code duplication"
    - "cibuildwheel test-command shell-chained smoke: pytest && python -c \"import smoke\" && console_script --help — single-line bash continuation avoids YAML/TOML multi-line escape gotchas"

key-files:
  created: []
  modified:
    - "setup.py: +1 LOC import shutil + +24 LOC build_py.build_package_data() asset-copy loop"
    - "pyproject.toml: +4 LOC [tool.setuptools.package-data] 'riscv.gtx' key + extended test-command for cibuildwheel helper-API smoke"
    - "MANIFEST.in: +2 LOC recursive-include rules for tests/gtx/data/{elf,golden}"
    - "tests/gtx/test_wheel_data_present.py: 121 LOC (rewrite from 38 LOC RED scaffold, +83 net LOC) — 4 GREEN tests covering importlib.resources + helper API"

key-decisions:
  - "tests/-as-canonical, src/-as-build-time-mirror (D-13 confirmed in execution): the build_py shutil.copy2 loop reads from tests/gtx/data/{elf,golden} and writes into <build_lib>/riscv/gtx/data/{firmware,golden} — not the in-tree src/main/python/riscv/gtx/data/ — so editable installs deliberately do NOT see these assets. This avoids polluting the source tree and cleanly separates test assets from wheel-distributed assets."
  - "elf/->firmware/ rename happens at copy time (not in source tree). tests/gtx/data/elf/ keeps the .S kernel + Makefile + .elf naming (matching vendor build artifacts) while wheel ships only .elf under data/firmware/ to align with ROADMAP P6 success #3 expectation. Documented inline in setup.py build_py comment."
  - "Test polymorphism via _gtx_data_dir() helper. The 4 tests gracefully skip when riscv.gtx.data/ is absent (editable install) and PASS when it exists (wheel install). Validated locally by both code paths: 4 SKIP in editable, 4 PASS after staging assets via build_py simulation, then cleaning up. No --noconftest needed because test bodies do not depend on fixtures."
  - "MANIFEST.in rule order matters: more-specific tests/gtx/data/{elf,golden} rules go AFTER the broader 'recursive-include tests *.py *.pyi' so they extend (not override) coverage. Verified by pyproject.toml/MANIFEST.in coexistence with sdist."
  - "Task 3 cibuildwheel matrix verification is checkpoint:human-verify NOT auto-runnable — requires Docker + manylinux2014 image (~15 min) and is documented in plan as a manual operator step (matches VALIDATION.md §Manual-Only Verifications)."

patterns-established:
  - "Build-time asset copy via build_py override: when tests/ assets must ship in the wheel without polluting src/, extend build_py.build_package_data() with a shutil.copy2 loop into <build_lib>/<pkg>/<sub>. Pattern reusable for any future phase that wants tests/ to remain canonical."
  - "Sub-package package-data via quoted-key TOML: [tool.setuptools.package-data] '\"riscv.gtx\" = [...]' (NOT 'riscv = [\"gtx/data/...\"]'). The dotted key targets the sub-package directly so globs are resolved relative to that sub-package's directory."
  - "Editable-vs-wheel test polymorphism: tests that depend on wheel-staged data should use a _data_dir() helper that returns None when missing, then early-skip — keeps editable-install CI green while still hard-PASSing in wheel-installed venvs."

requirements-completed: []  # Wave 3 Plan 05 completes the packaging-side work but PKG-01/PKG-03/PKG-04 close ONLY after Task 3 manual cibuildwheel matrix verification (checkpoint:human-verify). Marked complete on user "approved" reply.

# Metrics
duration: ~4min (Tasks 1-2; Task 3 awaiting manual checkpoint)
completed: 2026-05-07 (partial — Task 3 outstanding)
---

# Phase 6 Plan 5: Wheel-Bundled GTX Assets + cibuildwheel Surface Smoke Summary

**setup.py build_py copies tests/gtx/data/{elf,golden} to <pkg>/gtx/data/{firmware,golden} at wheel build time + pyproject.toml package-data globs for riscv.gtx + MANIFEST.in sdist coverage + 4 GREEN importlib.resources tests; cibuildwheel matrix gate awaits manual operator verification.**

## Performance

- **Duration:** ~4 min (Tasks 1-2 only; Task 3 is checkpoint:human-verify pending)
- **Started:** 2026-05-07T13:51:08Z
- **Completed (Tasks 1-2):** 2026-05-07T13:55:13Z
- **Tasks:** 2 of 3 complete (Task 3 outstanding — manual cibuildwheel matrix gate)
- **Files modified:** 4 (setup.py, pyproject.toml, MANIFEST.in, tests/gtx/test_wheel_data_present.py)

## Accomplishments

- **setup.py build_py extended** (+24 LOC): shutil.copy2 loop copies tests/gtx/data/elf/*.elf → <pkg>/gtx/data/firmware/ and tests/gtx/data/golden/*.hex → <pkg>/gtx/data/golden/ at wheel build time. Renames elf→firmware during copy (so wheel layout matches ROADMAP P6 success #3 expectation while tests/ keeps the vendor-aligned elf/ name).
- **pyproject.toml package-data** registers `"riscv.gtx" = ["data/firmware/*.elf", "data/golden/*.hex"]` so the build_py-staged files actually land in the wheel.
- **pyproject.toml cibuildwheel test-command extended** with helper-API import smoke (`from riscv.gtx._verify import compare_hex, bundled_elfs, load_golden; assert callable(...)`) + `pyspike-verify --help` (PKG-04).
- **MANIFEST.in** adds sdist coverage for tests/gtx/data/{elf,golden} (cibuildwheel pulls from sdist in some configurations).
- **tests/gtx/test_wheel_data_present.py GREEN-filled** (38 → 121 LOC, +83 net LOC): 4 tests — `test_firmware_data_dir_present_in_wheel`, `test_golden_data_dir_present_in_wheel`, `test_bundled_elfs_helper_returns_list`, `test_load_golden_helper_returns_bytes`. Polymorphic skip-vs-PASS based on whether riscv.gtx.data/ exists (editable vs wheel install).
- **Local validation done:**
  - **Editable install (current state)**: 4 tests SKIP gracefully → `4 skipped, 0 failed`.
  - **Wheel-install simulation** (manually staged via build_py copy logic, then cleaned up): 4 tests PASS hard → `4 passed`.
  - **Quick suite (full tests/gtx/)**: `283 passed, 15 skipped, 0 failed` — no regression vs Plan 04 baseline.
  - **Asset count + size**: 12 .elf (15.4 KiB) + 11 .hex (3.0 KiB) = **18.4 KiB total**, well below the D-15 50 MB cap.
- **Task 3 (checkpoint:human-verify) NOT executed by agent** — manual cibuildwheel matrix gate documented; orchestrator surfaces to user.

## Task Commits

1. **Task 1: Extend setup.py build_py + pyproject.toml package-data + cibuildwheel test-command + MANIFEST.in** — `48e6eeb` (feat)
2. **Task 2: GREEN-fill tests/gtx/test_wheel_data_present.py** — `900f301` (test)
3. **Task 3: Manual cibuildwheel matrix run + venv smoke + wheel-size gate** — **PENDING (checkpoint:human-verify)**

**Plan metadata commit:** TBD after Task 3 approval

## Files Created/Modified

- `setup.py` — `import shutil` added (alphabetical); `build_py.build_package_data()` extended with build-time asset copy loop (tests/gtx/data/{elf,golden} → <pkg>/gtx/data/{firmware,golden}) before delegating to super().
- `pyproject.toml` — `[tool.setuptools.package-data]` gains quoted-key sub-package entry `"riscv.gtx" = ["data/firmware/*.elf", "data/golden/*.hex"]`. `[tool.cibuildwheel] test-command` extended with helper-API import smoke + `pyspike-verify --help` chained via `&&`.
- `MANIFEST.in` — adds two `recursive-include tests/gtx/data/{elf,golden} ...` rules for sdist coverage; preserves `prune vendor/gtx_cpp_reference` exclusion.
- `tests/gtx/test_wheel_data_present.py` — full rewrite (38 → 121 LOC). Replaces 3 `pytest.skip(...)` Plan 01 RED stubs with 4 GREEN tests. Adds `_gtx_data_dir()` helper for skip-vs-PASS polymorphism. New 4th test `test_load_golden_helper_returns_bytes` validates the load_golden() helper API (D-14) — covers the mm_basic→mm_basic_n1s16 stem mapping documented in Plan 03.

## Decisions Made

1. **tests/-as-canonical, src/-as-build-time-mirror confirmed (D-13).** The build_py copy lands in `<build_lib>/<pkg>/...` (the setuptools build output, not `src/main/python/...`) so editable installs deliberately do NOT see the assets. This prevents pollution of the source tree, keeps `git status` clean across rebuilds, and means the editable-install CI must skip — handled by the `_gtx_data_dir()` polymorphic helper in the test file.
2. **elf/→firmware/ rename happens at copy time.** `tests/gtx/data/elf/` keeps the vendor-aligned naming (matches `vendor/test/<OP>/n1s16/data/` source), while the wheel ships under `data/firmware/` to align with ROADMAP P6 success #3 (`r.files('riscv.gtx').joinpath('data','firmware').iterdir()`). Documented inline in setup.py build_py comment.
3. **cibuildwheel test-command shell-chained via `&&`.** Single-line form (`pytest ... && python -c "..." && pyspike-verify --help`) avoids YAML/TOML multi-line escape gotchas. Each step's failure short-circuits the next. Verified pyproject.toml parses with the embedded escaped quotes (`\"...\"` for the double-quoted Python `-c` string inside the TOML double-quoted value).
4. **Sub-package package-data via TOML quoted key.** Used `"riscv.gtx" = [...]` (the literal quoted dotted name) rather than nesting under `riscv = ["gtx/data/firmware/*.elf"]` — the quoted-dotted form is the canonical setuptools idiom for a sub-package, with globs resolved relative to that sub-package's directory.
5. **MANIFEST.in rule ordering preserved.** Placed `recursive-include tests/gtx/data/elf` and `recursive-include tests/gtx/data/golden` AFTER `recursive-include tests *.py *.pyi` (extending) and BEFORE `recursive-include examples *` (preserving section structure). The `prune vendor/gtx_cpp_reference` rule is unchanged — vendor stays excluded from sdist.
6. **4-test surface (NOT 3 as in Plan 01 RED scaffold).** Added `test_load_golden_helper_returns_bytes` as the 4th test — Plan 01 only stubbed 3, but Plan 05 success criteria explicitly call out the load_golden() helper (PKG-03 wheel-install one-liner) plus the build_py asset copy needs both `*.elf` (firmware) and `*.hex` (golden) coverage. The 4th test exercises the elf-stem→golden-stem mapping (mm_basic → mm_basic_n1s16) so the test catches the legacy-name divergence Plan 03 documented.

## Deviations from Plan

None - plan executed exactly as written. The 4-test surface (vs Plan 01's 3-stub RED scaffold) was the explicit acceptance-criteria target in Plan 05's `<acceptance_criteria>` section ("`grep -c 'def test_load_golden_helper_returns_bytes' tests/gtx/test_wheel_data_present.py == 1`"), so this is per-plan, not a deviation.

**Total deviations:** 0
**Impact on plan:** Plan executed verbatim. No auto-fixes triggered.

## Issues Encountered

- **Local validation of "wheel-installed venv" PASS path required temporary asset staging.** Could not run actual `python3 -m build --wheel` + `pip install dist/spike-*.whl` from within the executor without invoking the full Spike C-extension build (~20 min on this machine, plus full RISC-V toolchain availability). Instead, simulated the build_py.copy logic directly into `src/main/python/riscv/gtx/data/` via a Python one-liner, ran pytest (4 PASSED), then cleaned up the staged tree via shutil.rmtree. This validates the test PASS path (importlib.resources + bundled_elfs() helper return non-empty) without requiring a full wheel build inside the agent. The actual cibuildwheel matrix run is Task 3 (checkpoint:human-verify) — left to the operator.

## User Setup Required

**Task 3 manual checkpoint** — see CHECKPOINT message at end of this execution. The operator runs (on a developer machine with Docker + cibuildwheel + Python 3.10/3.11/3.12 + /opt/riscv toolchain):

1. Clean rebuild + wheel: `python3 -m build --wheel`
2. Wheel size gate: `ls -la dist/spike-*.whl` (must be ≤ 50 MB; expected ~baseline + ~80 KB)
3. Local cp310 venv smoke: `python3 -m venv /tmp/spike-test-venv && /tmp/spike-test-venv/bin/pip install dist/spike-*.whl && /tmp/spike-test-venv/bin/python -c "from riscv.gtx import GtxNpu; from riscv.gtx._verify import compare_hex, bundled_elfs, load_golden; print('OK')" && /tmp/spike-test-venv/bin/pyspike-verify --help`
4. cibuildwheel matrix: `cibuildwheel --platform linux 2>&1 | tail -50` — expected last line: `3 wheels successfully built`
5. Wheel-installed regression sweep: `/tmp/spike-test-venv/bin/pytest tests/gtx/test_wheel_data_present.py tests/gtx/test_regression_fw_full.py -v` — expected: ≥7 PASS regressions + 4 PASS wheel data presence

## Next Phase Readiness

- **PKG-01 + PKG-03 + PKG-04 packaging-side work complete.** All source modifications committed. Manual matrix gate (Task 3) is the only remaining step before requirements close.
- **Quick suite stable**: 283 passed / 15 skipped / 0 failed.
- **Phase 7 (optimization) entry condition** — per PROJECT.md "Phase 7 added: numba 등의 라이브러리를 통해 동적 최적화 기술... 정상 동작 확인 후 핫스팟 가속; 진입 조건 = P6 회귀 그린". Phase 7 unblocks ONLY after Task 3 manual checkpoint succeeds.
- **Known stubs:** None. Plan 05 introduces no stubs — every test body either skips gracefully (editable install) or asserts on real data (wheel install).

## Self-Check: PASSED

- setup.py: FOUND
- pyproject.toml: FOUND
- MANIFEST.in: FOUND
- tests/gtx/test_wheel_data_present.py: FOUND
- 48e6eeb (Task 1 commit): FOUND
- 900f301 (Task 2 commit): FOUND
- 4-test pytest collection works in editable install (4 skipped, 0 failed)
- 4-test pytest PASSes hard after wheel-install simulation (4 passed)
- Quick suite tests/gtx/ unchanged: 283 passed / 15 skipped / 0 failed

---
*Phase: 06-verification-wheel*
*Completed: 2026-05-07 (Tasks 1-2; Task 3 checkpoint:human-verify pending)*
