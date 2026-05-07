---
phase: 06-verification-wheel
plan: 01
subsystem: testing
tags: [verify, fp16, argparse, console-script, importlib-resources, pyproject, hybrid-port]

# Dependency graph
requires:
  - phase: 04-mm-subsystem
    provides: tests/gtx/_verify_minimal.compare_hex (P4 78 LOC strict-validated core absorbed verbatim)
  - phase: 05-vec-act-pool
    provides: P5 strict-pass discipline (activation_relu_gelu.elf returncode=0; 5-tier graceful-skip pattern reused)
provides:
  - riscv.gtx._verify production module (compare_hex, bundled_elfs, load_golden, _print_report_fp16, main)
  - pyspike-verify console_script entry (D-02)
  - python -m riscv.gtx._verify entry (D-02)
  - Vendor verify.py argparse 1:1 + --strict flag (D-03)
  - Stats dict with BOTH mini-port keys + vendor verbose-report aliases (RESEARCH §Stats Dict Mapping)
  - tests/gtx/test_verify_compare_hex.py (6/6 PASS — Plan 01 GREEN)
  - tests/gtx/test_verify_cli.py (2/2 PASS + 1 SKIP-pending-PATH — Plan 01 GREEN)
  - tests/gtx/test_regression_fw_full.py (RED scaffold for Plan 04 Wave 1b)
  - tests/gtx/test_wheel_data_present.py (RED scaffold for Plan 05 Wave 2)
affects: [06-02-atexit, 06-03-vendor-assets, 06-04-regression-matrix, 06-05-packaging]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mini-port → production hybrid promotion: P4 _verify_minimal absorbed verbatim into riscv.gtx._verify with vendor argparse wrapper"
    - "Stats dict dual-keying: mini-port keys (back-compat) + vendor aliases (verbose report) coexist"
    - "Module-local _VERIFY_AVAILABLE detection (NOT conftest fixture) per P5 plan-05 D-1"
    - "RED scaffold for cross-wave dependencies (D-17): Plan 01 lands stubs only for plans in DIFFERENT waves; same-wave siblings own own files (D-18)"

key-files:
  created:
    - src/main/python/riscv/gtx/_verify.py
    - tests/gtx/test_verify_compare_hex.py
    - tests/gtx/test_verify_cli.py
    - tests/gtx/test_regression_fw_full.py
    - tests/gtx/test_wheel_data_present.py
    - .planning/phases/06-verification-wheel/deferred-items.md
  modified:
    - pyproject.toml (added [project.scripts] block with pyspike-verify entry)

key-decisions:
  - "D-01 hybrid base: copied _verify_minimal._parse_hex + compare_hex VERBATIM into _verify.py; only stats dict was extended with vendor aliases"
  - "D-02 dual CLI entry: pyproject [project.scripts] pyspike-verify + module __main__ block both wired"
  - "D-03 vendor argparse 1:1 + --strict flag added; --strict default False (parser default), but compare_hex default strict=True (D-14 mini-port lineage preserved)"
  - "D-14 helpers: bundled_elfs() catches FileNotFoundError + ModuleNotFoundError + AttributeError + NotADirectoryError to return [] gracefully when wheel asset dir missing (editable install before Plan 05)"
  - "D-18 zero-overlap: Plan 01 created NO files owned by Plans 02/03 (test_atexit_ddr_dump.py, test_assets_present.py, scripts/import_vendor_golden.py untouched)"

patterns-established:
  - "Hybrid port: Take P4 _verify_minimal mini-port (proven 78 LOC) + add vendor verify.py argparse + --strict wrapper (~120 LOC) = production module without re-implementing core"
  - "Stats dict aliasing: Single stats dict carries both back-compat keys (failures, exact_matches) and vendor verbose-report keys (mismatches, first_mismatch as byte-offset, size_result, size_golden, total_bytes, trailing_bytes). failures == mismatches invariant enforced"
  - "Module-local availability detection: try-except ImportError + _VERIFY_AVAILABLE flag + per-test pytest.skip guard; works with --noconftest acceptance commands"
  - "Console-script + module __main__ dual entry: same main() function called from both pyproject [project.scripts] and `if __name__ == '__main__': sys.exit(main())` block at module bottom"
  - "Cross-wave RED scaffold: BUNDLED_ELFS = sorted(...) parametrize anchor with `or [placeholder]` fallback so collection survives even before Plan 03 lands assets"

requirements-completed: [VRF-01]

# Metrics
duration: ~22min
completed: 2026-05-07
---

# Phase 6 Plan 01: VRF-01 _verify hybrid port + dual CLI entry Summary

**Production riscv.gtx._verify module with vendor argparse 1:1 + --strict + bundled_elfs/load_golden helpers; compare_hex stats dict carries both mini-port back-compat keys and vendor verbose-report aliases**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-05-07T13:00:00Z (approx)
- **Completed:** 2026-05-07T13:22:19Z
- **Tasks:** 2 (Task 1: _verify module; Task 2: cross-wave RED scaffolds)
- **Files created:** 6 (1 module + 4 test files + 1 deferred-items.md)
- **Files modified:** 1 (pyproject.toml)

## Accomplishments

- **riscv.gtx._verify module landed** (202 LOC): `_parse_hex` + `compare_hex` absorbed VERBATIM from P4 `_verify_minimal` (zero re-implementation of proven core); stats dict extended with vendor aliases (mismatches, first_mismatch as byte-offset, size_result, size_golden, total_bytes, trailing_bytes); `bundled_elfs()` + `load_golden()` helpers hide importlib.resources from end users (D-14); `_print_report_fp16()` prints vendor verify.py:312-343 1:1 + new strict-mode line; `main()` argparse vendor verify.py:350-388 1:1 + `--strict` flag (D-03).
- **Dual CLI entry wired**: `pyproject.toml [project.scripts] pyspike-verify = "riscv.gtx._verify:main"` (D-02 entry #1) + `if __name__ == '__main__': sys.exit(main())` block (D-02 entry #2). `python3 -m riscv.gtx._verify --help` exits 0 with `--strict / --ulp / --atol / --fp16` all visible.
- **Plan 01 GREEN tests (8 PASS)**: `test_verify_compare_hex.py` 6/6 PASS (strict-zeros, stats-keys-present mini+vendor, BE bit-pair, tolerant-within-ULP, strict-rejects-tolerance, @-and-#-line skip); `test_verify_cli.py` 2/2 PASS + 1 SKIP (pyspike-verify console_script awaits PATH from Plan 05 wheel install or `pip install -e .`; `python -m` and `main()` self-compare both PASS).
- **Cross-wave RED scaffolds (7 SKIP, all collected cleanly)**: `test_regression_fw_full.py` with `BUNDLED_ELFS` parametrize anchor (3 placeholder skips, Plan 04 Wave 1b GREEN-fills); `test_wheel_data_present.py` with 3 PKG-01 stubs (Plan 05 Wave 2 GREEN-fills).
- **D-18 zero-overlap respected**: Plan 01 did NOT touch `tests/gtx/test_atexit_ddr_dump.py` (Plan 02), `tests/gtx/test_assets_present.py` (Plan 03), `scripts/import_vendor_golden.py` (Plan 03), `riscv/gtx/{ddr,npu,__init__}.py` (Plan 02), `tests/gtx/data/{elf,golden}/*` (Plan 03), or `[tool.setuptools.package-data]` (Plan 05).
- **Mini-port back-compat preserved**: `from tests.gtx._verify_minimal import compare_hex` still works unchanged; P4/P5 regressions that import this directly are zero-regression.

## Task Commits

Each task was committed atomically (with `--no-verify` per parallel-wave protocol):

1. **Task 1 (TDD): Author riscv.gtx._verify with absorbed core, helpers, argparse, console-script entry** — `67d4297` (feat)
   - RED: created test_verify_compare_hex.py + test_verify_cli.py with 9 tests; ran (all skip due to ImportError as expected).
   - GREEN: created src/main/python/riscv/gtx/_verify.py (202 LOC) + edited pyproject.toml [project.scripts]; tests went 8 PASS + 1 SKIP.
   - All Task 1 acceptance criteria PASS: ≥150 LOC, all 5 grep counts == 1, --help exits 0 with all 4 flags, real-asset self-compare returns mini-port + vendor keys with `failures == mismatches`, [project.scripts] singleton, [tool.setuptools.package-data] singleton (Plan 01 did not duplicate).
2. **Task 2: GREEN-fill test_verify_compare_hex.py + test_verify_cli.py + RED scaffolds for cross-wave deps** — `b8d1a53` (test)
   - test_verify_compare_hex.py + test_verify_cli.py already authored as TDD-RED in Task 1; converted to GREEN by Task 1's _verify.py landing.
   - test_regression_fw_full.py + test_wheel_data_present.py created as RED scaffolds for cross-wave Plans 04/05.
   - deferred-items.md logs Plan 02 WIP edit to test_regression_fw_act.py (out-of-scope for Plan 01).

## Files Created/Modified

- `src/main/python/riscv/gtx/_verify.py` (NEW, 202 LOC) — Production verify CLI module with hybrid base
- `tests/gtx/test_verify_compare_hex.py` (NEW) — 6 GREEN unit tests for compare_hex
- `tests/gtx/test_verify_cli.py` (NEW) — 3 CLI smoke tests (2 PASS + 1 SKIP)
- `tests/gtx/test_regression_fw_full.py` (NEW) — RED scaffold with BUNDLED_ELFS anchor for Plan 04
- `tests/gtx/test_wheel_data_present.py` (NEW) — RED scaffold for Plan 05 wheel-install smoke
- `.planning/phases/06-verification-wheel/deferred-items.md` (NEW) — Out-of-scope log
- `pyproject.toml` (MODIFIED) — Added 3-line [project.scripts] block between [project.optional-dependencies] and [tool.setuptools]

## Decisions Made

- **stats dict trailing_bytes formula**: `min(len(a_bytes), len(g_bytes)) % 2` — zero in normal case (32-byte aligned), only nonzero if odd-length input causes one trailing byte. Matches vendor verify.py:329 invariant.
- **first_mismatch encoding**: vendor reports byte-offset (`first_mismatch: 0x<offset:08x>`); _verify_minimal reports FP16 idx tuple. Plan 01 stores `first_mismatch = first_failure[0] * 2` (FP16 idx → byte offset since BE bit-pair = 2 bytes). Both representations available.
- **bundled_elfs() exception breadth**: catches `FileNotFoundError, ModuleNotFoundError, AttributeError, NotADirectoryError` — broader than RESEARCH minimum because importlib.resources behavior on missing data dirs varies across Python 3.10/3.11/3.12 (`AttributeError` on `.iterdir()` of MultiplexedPath in some configurations). Returns `[]` gracefully so editable installs (no Plan 05 build_py copy yet) don't crash imports.
- **Module-local _VERIFY_AVAILABLE pattern**: replicated from P5 plan-05 D-1 — acceptance commands use `--noconftest`, so we can't rely on conftest's `riscv_available` fixture. Each test file does its own try-except ImportError detection.
- **--strict flag default = False** in argparse parser, but **strict=True** is the compare_hex Python API default (D-14 lineage preserved). User must pass `--strict` on CLI to opt into strict mode; programmatic callers get strict by default.

## Deviations from Plan

None — plan executed exactly as written.

The plan body mandated 6 GREEN tests in test_verify_compare_hex.py and ≥1 PASS in test_verify_cli.py. Both targets exceeded:
- test_verify_compare_hex.py: 6/6 PASS (target ≥4, achieved 6).
- test_verify_cli.py: 2/3 PASS + 1 SKIP-pending-PATH (target ≥1 PASS, achieved 2).

Plan instructions for `_verify.py` Section 1-4 followed verbatim: imports identical, _parse_hex VERBATIM from _verify_minimal lines 10-19, compare_hex VERBATIM from _verify_minimal lines 22-72 with stats dict extended per RESEARCH §Stats Dict Mapping, bundled_elfs/load_golden per Step A skeleton, _print_report_fp16 per RESEARCH Pattern 1, main() per vendor verify.py:350-388 1:1 + --strict.

## Issues Encountered

- **Pre-existing test failure in test_regression_fw_act.py** (NOT introduced by Plan 01): During Task 2 broader regression sweep, `test_act_strict_mode_pass` failed with "P6 D-04 broken: subprocess clean-exited (rc=0) with GTX_DDR_DUMP set, but no dump file was written". Root cause: Plan 02 parallel agent (Wave 1a sibling) has WIP-edited test_regression_fw_act.py to convert tier #5 graceful-skip into a hard PASS gate, but the atexit hook (Plan 02's own deliverable) is not yet committed. Verified by `git stash`: stashing leaves test passing. Logged in `.planning/phases/06-verification-wheel/deferred-items.md`. Out-of-scope for Plan 01 — Plan 02 owns the GREEN.

## User Setup Required

None — no external service configuration required.

The `pyspike-verify` console_script becomes available on PATH only after `pip install -e .` (developer) or `pip install dist/*.whl` (Plan 05 wheel install). The 1 SKIP in test_verify_cli.py reflects this — it's intentional and converts to a hard PASS once Plan 05's packaging completes.

## Next Phase Readiness

- **For Plan 02 (atexit)**: `riscv.gtx._verify.compare_hex(strict=True)` is available for atexit subprocess regression tests to call.
- **For Plan 03 (vendor assets)**: `bundled_elfs()` + `load_golden()` API surface available for asset-presence tests once Plan 03 lands assets and Plan 05 lands build_py copy.
- **For Plan 04 (regression matrix)**: `tests/gtx/test_regression_fw_full.py` parametrize anchor (`BUNDLED_ELFS`) is in place — Plan 04 will tighten the `or [placeholder]` fallback away and GREEN-fill the per-elf body. The strict-mode compare_hex API is fully usable.
- **For Plan 05 (packaging)**: `tests/gtx/test_wheel_data_present.py` 3 stubs are in place — Plan 05 will replace `pytest.skip` with assertions on `r.files('riscv.gtx').joinpath('data', 'firmware').iterdir()`. `[project.scripts]` block exists; Plan 05 only needs to add `[tool.setuptools.package-data]` extension (NOT replace).

## Self-Check: PASSED

- src/main/python/riscv/gtx/_verify.py — FOUND
- pyproject.toml [project.scripts] entry — FOUND (`grep -E '^pyspike-verify\s*=\s*"riscv\.gtx\._verify:main"' pyproject.toml` matches)
- tests/gtx/test_verify_compare_hex.py — FOUND
- tests/gtx/test_verify_cli.py — FOUND
- tests/gtx/test_regression_fw_full.py — FOUND
- tests/gtx/test_wheel_data_present.py — FOUND
- .planning/phases/06-verification-wheel/deferred-items.md — FOUND
- Commit 67d4297 (Task 1 feat) — FOUND in git log
- Commit b8d1a53 (Task 2 test) — FOUND in git log
- D-18 zero-overlap: tests/gtx/test_atexit_ddr_dump.py / test_assets_present.py / scripts/import_vendor_golden.py NOT created by Plan 01 — VERIFIED via git status

---
*Phase: 06-verification-wheel*
*Plan: 01*
*Completed: 2026-05-07*
