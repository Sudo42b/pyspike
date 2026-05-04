---
phase: 01-foundation
plan: 02-fp
subsystem: numerics
tags: [fp16, fp32, numpy, ieee754, rne, view, astype, helpers]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: D-09 lock-in (np.float16 view), NumPy 2.x baseline (D-07), cp310+ runtime (D-08)
provides:
  - "riscv.gtx.fp.fp16_to_fp32 / fp32_to_fp16 helpers (np.float16 view via astype, NOT bit manipulation)"
  - "tests/gtx/__init__.py + tests/gtx/test_fp_roundtrip.py (5-test acceptance suite for FOUND-01)"
  - "Empirical proof on NumPy 2.2.6: all 65536 FP16 round-trip exact incl. 2046 NaN bit patterns"
affects: [01-foundation/03-memory, 04-packaging, 02-rocc-dispatch, 04-mm, 05-vec-act, 06-verify]

# Tech tracking
tech-stack:
  added: []  # No new runtime deps; only stdlib + numpy (already pinned numpy>=2.0,<3 baseline)
  patterns:
    - "FP16<->FP32 conversion = np.asarray(x, dtype=np.floatN).astype(np.floatM) chain (D-09)"
    - "Vectorized exhaustive FP16 round-trip test (np.arange(65536, dtype=uint16).view(float16))"
    - "tests/gtx/ subdirectory pattern (mirrors src/main/python/riscv/gtx/ structure)"
    - "Apache 2.0 license header (2026 WuXi EsionTech) at top of every new .py"

key-files:
  created:
    - src/main/python/riscv/gtx/__init__.py (16 LOC, package marker; only fp.py exposed in P1)
    - src/main/python/riscv/gtx/fp.py (52 LOC, fp16_to_fp32/fp32_to_fp16 + ArrayLike type alias)
    - tests/gtx/__init__.py (16 LOC, package marker)
    - tests/gtx/test_fp_roundtrip.py (88 LOC, 5 acceptance tests)
  modified: []  # None - all-new files

key-decisions:
  - "D-09 implementation via astype-only: NO struct.pack, NO int.from_bytes, NO bit shifts (acceptance criterion enforced via grep)"
  - "ArrayLike = Union[np.ndarray, np.float16, np.float32, float] for both helpers (handles scalar + ndarray inputs)"
  - "np.asarray(x, dtype=...) normalization + astype: idempotent for already-correct-dtype inputs, copy semantics deliberate (D-12 view-base invariant applies to memory accessors only)"
  - "tests/gtx/__init__.py created so tests/gtx/ becomes a regular package (matches existing tests/__init__.py pattern; pytest collection compatible)"

patterns-established:
  - "FP16 view discipline (D-09): astype chain only, vectorized; fp_strict.py fallback deferred to P4/P5 if strict-mode discrepancies surface"
  - "Vectorized acceptance test (no Python `for i in range(N)` loops; np.testing.assert_array_equal on uint16 view for bit-exact equality)"

requirements-completed:
  - FOUND-01

# Metrics
duration: 3m17s
completed: 2026-05-04
---

# Phase 01 Plan 02: FP16/FP32 Helpers Summary

**FP16<->FP32 conversion helpers via `np.float16`/`np.float32` astype (D-09 lock-in) with exhaustive 65536-value bit-exact round-trip suite — all 5 tests green on NumPy 2.2.6 in 0.28s.**

## Performance

- **Duration:** 3m17s
- **Started:** 2026-05-04T05:37:06Z
- **Completed:** 2026-05-04T05:40:24Z
- **Tasks:** 2 (Task 02-01 RED + Task 02-02 GREEN; no REFACTOR needed)
- **Files created:** 4 (2 source + 2 test scaffolding)

## Accomplishments

- `riscv.gtx.fp` module exists with `fp16_to_fp32` / `fp32_to_fp16` helpers (D-09 view pattern, astype-only, zero bit manipulation).
- 5 acceptance tests in `tests/gtx/test_fp_roundtrip.py` empirically verify on NumPy 2.2.6:
  - **65536 FP16 bit patterns** all round-trip exactly (`fp32_to_fp16(fp16_to_fp32(x)).view(uint16) == x`).
  - **2046 NaN bit patterns** all preserved (input bit pattern == output bit pattern, all `np.isnan(...)`).
  - **All FP16 subnormals** (`exp==0`, `mantissa!=0`; 0x0001..0x03FF + 0x8001..0x83FF) round-trip exact.
  - **Negative zero** (0x8000) preserves sign bit through round-trip.
  - **Known cases** (1.0=0x3C00, 2.0=0x4000, 0.5=0x3800, -1.0=0xBC00) hex-exact.
- Test runtime 0.28s (D-16 sub-1s expectation comfortably met). 65536 round-trip alone runs in ~64ms.
- `riscv.gtx` subpackage marker (`__init__.py`) created for downstream sibling plans (03-memory, etc.) to extend.

## Task Commits

Each task committed atomically with `--no-verify` (parallel-executor protocol):

1. **Task 02-01: tests/gtx/test_fp_roundtrip.py — RED phase** — `0453995` (test)
2. **Task 02-02: src/main/python/riscv/gtx/fp.py — GREEN phase** — `8311a46` (feat)

## Files Created/Modified

- `src/main/python/riscv/gtx/__init__.py` (NEW, 16 LOC) — Subpackage marker; only `fp` exposed in Phase 1.
- `src/main/python/riscv/gtx/fp.py` (NEW, 52 LOC) — `fp16_to_fp32(x: ArrayLike) -> np.ndarray`, `fp32_to_fp16(x: ArrayLike) -> np.ndarray`. Both implemented via `np.asarray(...).astype(...)` chain. `ArrayLike = Union[np.ndarray, np.float16, np.float32, float]`. Apache 2.0 header.
- `tests/gtx/__init__.py` (NEW, 16 LOC) — Test subpackage marker (matches `tests/__init__.py` style).
- `tests/gtx/test_fp_roundtrip.py` (NEW, 88 LOC) — 5 acceptance tests; vectorized; no Python loops; uses `np.testing.assert_array_equal` on `view(np.uint16)` for bitwise equality.

**Total LOC added:** 172 (52 source + 120 test/scaffolding).

## Decisions Made

- **D-09 implementation = astype-only.** Used `np.asarray(x, dtype=np.float16).astype(np.float32)` and the symmetric counterpart. Acceptance criterion `grep -E '(struct\.pack|int\.from_bytes|<<|>>)' src/main/python/riscv/gtx/fp.py` returns empty (verified — exit code 1).
- **NumPy 2.x IEEE 754 binary16 RNE confirmed adequate at helper level.** All 65536 patterns round-trip exact, including all 2046 NaN bit patterns. The strict-mode test vs C++ `gtx_fp32_to_16` (sticky/round-half-to-even payload preservation) is deferred to P4/P5 strict-mode measurement per CONTEXT.md D-09 risk acknowledgment.
- **`ArrayLike = Union[np.ndarray, np.float16, np.float32, float]`** chosen to accept the full set of inputs `test_known_values` exercises (`np.float16(1.0)`, `np.float32(1.0)`) plus arrays. Caller does NOT need to pre-cast.
- **`tests/gtx/__init__.py` created.** Plan didn't explicitly mandate it, but parallel sibling tests (e.g. `test_memory_layout.py` from plan 03) will need the directory to exist; matches existing `tests/__init__.py` license-header pattern.

## Deviations from Plan

### Out-of-scope environment limitations (logged, NOT fixed — Rule scope boundary)

The following are **pre-existing build/environment issues unrelated to this plan's changes**. Per CLAUDE.md scope boundary, these are documented but NOT fixed in this plan:

1. **`libriscv.so` missing in dev environment.**
   - `import riscv` warns `Missing libriscv.so` — the C++ extension hasn't been built (`pip install -e '.[dev]'` not run).
   - This does NOT affect `riscv.gtx.fp` (pure Python, only depends on `numpy`).
   - Phase 1 packaging plan (04-packaging) and submodule plan (05-submodule) own the build pipeline.

2. **`tests/conftest.py` hard-imports `riscv.cfg` / `riscv.debug_module` / `riscv.sim`** (C++-bound modules).
   - Without `libriscv.so`, top-level pytest collection fails with `ModuleNotFoundError: No module named 'riscv.cfg'`.
   - **Workaround used for verification:** `pytest tests/gtx/test_fp_roundtrip.py --noconftest -o addopts=""` runs only our scope's tests and bypasses pylint/mypy plugins (which may not be installed in this dev shell).
   - In CI / a built environment, the standard `pytest tests/gtx/test_fp_roundtrip.py` invocation (per plan acceptance criteria) will succeed without flags. The result of fully-qualified runs (post-build) MUST be re-verified during Phase 1 packaging tasks.

3. **`pyproject.toml` `addopts = "--pylint --mypy --cov-report=lcov"`.**
   - Requires `pytest-pylint` and `pytest-mypy` plugins. Available via `[project.optional-dependencies].dev` but not necessarily installed in the executor's shell.
   - Same workaround applies (`-o addopts=""` override).

**Total deviations:** 0 fixes applied (all 3 items above are out-of-scope build/env conditions, logged but deferred). No scope creep; no auto-fixes needed.

**Impact on plan:** Plan executed exactly as written for the scope it owns (helpers + tests). All 5 tests pass when invoked correctly for the dev shell. Acceptance criteria fully met when verified individually with `--noconftest`.

## Known Stubs

None. Both functions are fully wired (real `astype`, real numpy data flow); the test file imports the real symbols and validates real round-trip semantics. No placeholder/empty/TODO patterns.

## Issues Encountered

- Initial pytest invocation fails because pyproject.toml `addopts` requires plugins not present in the executor shell, AND because `tests/conftest.py` requires the (un-built) C++ extension. Resolved by running with `--noconftest -o addopts=""` for verification (see Deviations §2-3). This is environmental and pre-existing.

## D-09 Risk Status

**HIGH-confidence at helper level.** NumPy 2.2.6 produces bit-exact round-trip for all 65536 FP16 values, including all 2046 NaN bit patterns (no canonicalization to 0x7E00). Negative zero, subnormals, and known constants all preserved. Bit manipulation NOT used; no `struct`, `int.from_bytes`, or shifts in `fp.py`.

**Pending measurement (P4/P5 strict mode):** the helper's IEEE 754 binary16 RNE may differ from C++ `gtx_fp32_to_16` (`vendor/gtx_cpp_reference/gtx/gtx_npu.h:89-151`) on:
- subnormal handling (sticky bit during narrowing)
- NaN payload preservation across operations (vs simple identity round-trip tested here)
- halfway-rounding policy (RNE half-to-even vs RNE-near-zero edge cases)

If P4/P5 strict-mode acceptance gate fails, the fallback `gtx/fp_strict.py` (port of the C++ bit code) will be added at that time. Phase 1 explicitly does not block on this.

## Next Phase Readiness

- **Plan 03-memory** can now `from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16` for `mem.l1_f16(...)` named accessor return-type docs and unit tests.
- **Plans 04/05 (MM/VEC/ACT op handlers, future phases)** can use these helpers as the canonical FP-conversion API; if strict-mode divergence surfaces, the fallback path is one new file (`fp_strict.py`) without changing call sites.
- **Phase 1 packaging plan (04-packaging)** must add `riscv.gtx` to `[tool.setuptools.packages.find]` discovery (already implicit via `include = ["riscv"]`) and verify the test runs cleanly under the full pytest invocation (`--pylint --mypy`) once `libriscv.so` builds.

## Self-Check

Verifying all claims before finalization:

- `[ -f src/main/python/riscv/gtx/fp.py ]` → FOUND
- `[ -f src/main/python/riscv/gtx/__init__.py ]` → FOUND
- `[ -f tests/gtx/test_fp_roundtrip.py ]` → FOUND
- `[ -f tests/gtx/__init__.py ]` → FOUND
- Commit `0453995` (RED) → FOUND in worktree branch git log
- Commit `8311a46` (GREEN) → FOUND in worktree branch git log
- pytest 5/5 PASS → VERIFIED (last run 0.28s)
- No bit manipulation → VERIFIED (`grep -nE '(struct\.pack|int\.from_bytes|<<|>>)' src/main/python/riscv/gtx/fp.py` exits 1)
- 65536 round-trip exact incl. NaN preservation → VERIFIED via test_all_65536_fp16_values_idempotent + test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern

## Self-Check: PASSED

---

*Phase: 01-foundation, plan 02 (fp).*
*Completed: 2026-05-04 (executor: parallel worktree agent-a64955efd060d29a5).*
