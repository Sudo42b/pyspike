---
phase: 07-numba
plan: 02
subsystem: testing
tags: [numba, njit, jit, fp32-only, gemm, ulp-0-parity, fp32-boundary, cache]

# Dependency graph
requires:
  - phase: 07-numba
    provides: _jit.py (HAS_NUMBA + njit shim) + _njit_helpers registry rows for 3 gemm kernels + test_njit_parity RED scaffold
  - phase: 04-mm-subsystem
    provides: gemm_core public API (3 functions) + explicit 3-loop FP32 accumulate baseline (mm_engine callers)
provides:
  - "Three FP32-only `_impl` functions in gemm_core.py: _gemm_core_impl, _gemm_reduce_sum_a_impl, _gemm_dot_impl (numba @njit boundary; FP32 in / FP32 out, positional args only)"
  - "Three @njit(cache=True) wrappers via re-call pattern (CONTEXT D-11 Option B): _gemm_core_njit, _gemm_reduce_sum_a_njit, _gemm_dot_njit"
  - "Public API gemm_core / gemm_reduce_sum_a / gemm_dot byte-identical to P4 signatures (FP16 in / FP16-or-float out); FP16<->FP32 cast happens in wrapper, OUTSIDE @njit boundary (NJIT-FP32-BOUNDARY)"
  - "3 gemm Tier 1 ULP-0 parity tests GREEN in test_njit_parity.py (pytest -k gemm -> 3 passed; full collection -> 4 passed + 25 skipped)"
  - "3 .nbi cache files generated under __pycache__ on first invocation (numba installed)"
  - "Pattern reference for Plans 03 (vec) + 04 (act) — additive-friendly kernel_name dispatch in test_njit_parity.py allows extension without rewriting gemm logic"
affects: [07-03, 07-04, 07-05, 07-06]

# Tech tracking
tech-stack:
  added: []  # numba already added by Plan 07-01; this plan only consumes the shim
  patterns:
    - "Re-call njit decoration: `_x_njit = njit(cache=True)(_x_impl)` instead of `@njit` on def — keeps the pure Python `_impl` callable so HAS_NUMBA=False path runs unchanged"
    - "FP32-only _impl boundary (NJIT-FP32-BOUNDARY): public wrapper handles FP16<->FP32 cast; numba CPU does not support np.float16"
    - "Positional bool + zero-fill ndarray sentinel for numba lazy dispatch: replaces P4 `*, has_bias=False, bias_fp32: Optional[ndarray]=None` because numba lazy-typed dispatch cannot accept kwonly or Optional[ndarray]"
    - "Additive-friendly test dispatch: kernel_name -> handler routing in test_njit_parity.py allows Plans 03/04 to add branches without rewriting existing gemm logic"

key-files:
  created: []
  modified:
    - src/main/python/riscv/gtx/gemm_core.py
    - tests/gtx/test_njit_parity.py

key-decisions:
  - "Use `np.ascontiguousarray(arr, dtype=np.float32)` for all FP16->FP32 input casts (numba lazy-typing produces different specializations for C-contig vs strided; ensures cache hit consistency across mm_engine call sites)"
  - "Pass real zero-fill `np.zeros((M,N), np.float32)` bias sentinel when has_bias=False instead of attempting Optional[ndarray] dispatch — accepts ~2x cold-compile time (one specialization per has_bias bool value) for now; unification deferred to v1.x/v2 per plan W7 note"
  - "Inline kernel_name dispatch in test_njit_parity.py instead of extending `_njit_helpers.generate_test_inputs` — keeps Plan 02 surgical (does not touch a file Plans 03/04 will also need to touch); helper specialization deferred per its docstring"
  - "Validation (shape/dtype check + ValueError/TypeError raise) stays in the public Python wrapper, not inside @njit `_impl` — cleaner error messages + numba bypass on user error"
  - "Public `gemm_dot` wrapper does NOT add prior_accum twice: `_gemm_dot_impl` already returns `s + prior_accum`, so the wrapper just `return float(s)`"

patterns-established:
  - "Re-call `_x_njit = njit(cache=True)(_x_impl)`: pattern reused by Plans 03/04 for vec_core / act_core kernels"
  - "FP32-only `_impl` + thin FP16-cast public wrapper: 28-kernel canonical boundary across gemm/vec/act"
  - "Test dispatch by kernel_name in test_njit_parity.py: Plans 03/04 add `elif kernel_name in VEC_KERNELS:` / `elif kernel_name in ACT_KERNELS:` branches without touching gemm logic"

requirements-completed: [NJIT-02, NJIT-05]

# Metrics
duration: 6m55s
completed: 2026-05-09
---

# Phase 7 Plan 02: GEMM JIT Promotion Summary

**Three FP32-only `_impl` kernels in gemm_core.py wrapped by `@njit(cache=True)` via re-call pattern, with public API byte-identical to P4 (FP16 in/out preserved); 3 gemm Tier 1 ULP-0 parity tests GREEN; 17 P4 strict-mode tests still PASS; 3 .nbi cache files generated on first invocation.**

## Performance

- **Duration:** 6m 55s
- **Started:** 2026-05-09T06:26:44Z
- **Completed:** 2026-05-09T06:33:41Z
- **Tasks:** 2 (both autonomous, both TDD-style)
- **Files modified:** 2

## Accomplishments

- `gemm_core.py` rewritten with 3 FP32-only `_impl` functions matching vendor `gtx_npu_mm.cc:73-79, 200-211, 262-265` line-by-line (explicit 3-loop FP32 accumulate; no `np.matmul` / `np.dot` / `np.einsum` / `np.sum` anywhere).
- 3 `@njit(cache=True)` wrappers via re-call pattern (`_gemm_core_njit = njit(cache=True)(_gemm_core_impl)`) — pure-Python `_impl` callable preserved for HAS_NUMBA=False path.
- Public API `gemm_core(A, B, *, has_bias=False, bias_fp32=None) -> FP16`, `gemm_reduce_sum_a(A, *, prior_accum=0.0) -> float`, `gemm_dot(A, B, *, prior_accum=0.0) -> float` byte-identical to P4 — `mm_engine.py` callers see ZERO behavior change. All 17 P4 strict-mode regression tests (`test_op_mm` + `test_mm_chain` + `test_regression_fw_mm`) PASS unchanged.
- Tier 1 GREEN for gemm: `pytest tests/gtx/test_njit_parity.py -k gemm -v` reports `3 passed`. Full collection reports `4 passed + 25 skipped` (3 gemm + has_numba_detection sentinel; vec/act 25 skipped pending Plans 03/04).
- `test_njit_parity.py` body now dispatches on `kernel_name`: `_run_gemm_parity` runs for the 3 gemm kernels (ULP-0 byte equality via `np.array_equal(... .view(np.uint16))`); other 25 kernels keep `pytest.skip` for additive-friendly Plan 03/04 GREEN-fill.
- 3 `.nbi` cache files materialize under `src/main/python/riscv/gtx/__pycache__/` on first invocation:
  - `gemm_core._gemm_core_impl-65.py310.nbi`
  - `gemm_core._gemm_dot_impl-118.py310.nbi`
  - `gemm_core._gemm_reduce_sum_a_impl-102.py310.nbi`

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite gemm_core.py with FP32-only `_impl` + njit wrappers** — `38c6344` (feat)
2. **Task 2: GREEN-fill 3 gemm parity tests in test_njit_parity.py** — `9e82a36` (test)

**Plan metadata:** to be added by `final_commit` step (SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md).

## Files Created/Modified

- `src/main/python/riscv/gtx/gemm_core.py` (modified, +126/-25 LOC) — Apache 2.0 header preserved verbatim. Module docstring extended with NJIT-02 / NJIT-FP32-BOUNDARY rationale. New imports: `from ._jit import njit, HAS_NUMBA`. New `_impl` section (3 FP32-only functions). New `_njit` section (3 re-call wrappers). Public API section preserves P4 signatures verbatim, casts FP16->FP32 contiguous, calls the `_njit`, casts back to FP16/float.
- `tests/gtx/test_njit_parity.py` (modified, +88/-12 LOC) — Replace single `pytest.skip` parametrize body with kernel_name dispatch. New `_generate_gemm_inputs` helper (fixed-seed FP16 16x16 matrices / length-16 vectors). New `_run_gemm_parity` dispatcher (matrix / reduce / dot kinds, ULP-0 byte equality). 25 non-gemm parametrize cases continue `pytest.skip` for Plans 03/04. `test_has_numba_detection` sentinel preserved verbatim.

## Decisions Made

- **Re-call wrapper pattern (CONTEXT D-11 Option B) over inline `@njit` decorator** — keeps the pure-Python `_impl` callable as a separate top-level name, so `HAS_NUMBA=False` execution path is exactly the original Python (no decorator wrap, no overhead). The shim's passthrough already handles `@njit(cache=True)` correctly, but the re-call pattern makes the dual-path explicit.
- **`np.ascontiguousarray(arr, dtype=np.float32)` for all FP16->FP32 input casts** — numba lazy-typing produces different specializations for C-contig vs strided inputs. Forcing C-contig at the wrapper boundary ensures all `mm_engine.py` call sites land on the same compiled specialization, maximizing cache hit rate.
- **Real zero-fill `np.zeros((M,N), np.float32)` sentinel for `has_bias=False` path** — numba cannot lazily type `Optional[ndarray]`; the `_impl` MUST always receive a real ndarray. Cold-compile time grows ~2x because lazy dispatch produces one specialization per `has_bias` bool value (False / True bytecode branches differ). Acceptable for Phase 7; deferred unification (always-add-bias path with zero sentinel collapses to a single specialization) noted as v1.x / v2 work per plan W7 note.
- **Validation (shape/dtype check + ValueError/TypeError raise) stays in the public Python wrapper** — cleaner error messages than numba's compile-time type errors. The `_impl` retains a single defensive `if K != K2: raise ValueError(...)` shape check that survives @njit (numba supports `raise` with constant string).
- **Inline kernel_name dispatch in test_njit_parity.py** instead of extending `_njit_helpers.generate_test_inputs` — Plan 02 should not touch a helper file Plans 03/04 also need to touch (avoids future merge conflicts during parallel execution). The helper's docstring already says "Plan 02/03/04 may extend this for kernels with non-trivial signatures."
- **Public `gemm_dot` wrapper just `return float(s)`** — `_gemm_dot_impl` already returns `s + prior_accum`, so the wrapper does NOT add `prior_accum` again. (Initial draft had a typo `+ prior_accum - prior_accum`; caught and removed before commit.)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Initial wrapper draft double-counted `prior_accum` in `gemm_dot`**

- **Found during:** Task 1 (gemm_core.py rewrite).
- **Issue:** First-pass draft of public `gemm_dot` wrapper had `return float(s) + float(prior_accum) - float(prior_accum)` (an unintentional scratch line left during refactor). Since `_gemm_dot_impl` already returns `s + prior_accum`, this would have been a no-op for the test path (cancels) but would silently break any caller that relied on the value. The MM_V/MMC_V chain (mm_engine.py:298) wraps the return in `np.float32(...)` and stores into `mxe_accum` — so the bug would not surface in the dot test but WOULD surface as silent corruption of `mxe_accum` in chained MMC_V calls.
- **Fix:** Replaced with simple `return float(s)`. Verified the result by running all 17 P4 strict-mode tests (`test_mm_chain` exercises MMC_V chains; would catch corruption immediately).
- **Files modified:** `src/main/python/riscv/gtx/gemm_core.py` (single line in `gemm_dot`).
- **Verification:** `pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py tests/gtx/test_regression_fw_mm.py` reports 17 passed.
- **Committed in:** `38c6344` (Task 1 commit) — fix applied before commit, no separate commit needed.

**2. [Rule 1 - Cosmetic / Tooling] gsd-tools `verify key-links` reports false-negative regex pattern errors**

- **Found during:** Plan-level verification (post-Task 2).
- **Issue:** `gsd-tools verify key-links .planning/phases/07-numba/07-02-PLAN.md` reports all 3 plan key-links as "not verified":
  - Link 1: `from \\._jit import njit` — "not found in source or target" (false; line 57 of gemm_core.py contains `from ._jit import njit, HAS_NUMBA`).
  - Link 2: `_gemm_core_njit = njit\\(` — "Invalid regex pattern" (parser error from over-escaped backslashes).
  - Link 3: `from \\._njit_helpers import` — "not found in source or target" (false; line 26 of test_njit_parity.py contains `from ._njit_helpers import (`).
- **Fix:** No code change. The verifier appears to double-escape the `\.` regex sequence, producing a literal `\\.` that does not match the source. This is the SAME false-negative issue documented in 07-01-SUMMARY Deviation #2 (`numba>=0\.61\.2` link). Manually verified all 3 patterns are present in the actual files using `grep -E`:
  ```
  $ grep -E "from \._jit import njit" src/main/python/riscv/gtx/gemm_core.py
  from ._jit import njit, HAS_NUMBA  # noqa: F401  (HAS_NUMBA re-exposed for callers)
  $ grep -E "_gemm_core_njit = njit\(" src/main/python/riscv/gtx/gemm_core.py
  _gemm_core_njit = njit(cache=True)(_gemm_core_impl)
  $ grep -E "from \._njit_helpers import" tests/gtx/test_njit_parity.py
  from ._njit_helpers import (
  ```
- **Files modified:** None.
- **Verification:** Direct grep above confirms all 3 patterns present.
- **Committed in:** N/A (verifier tool quirk, not source code).

---

**Total deviations:** 2 (1 self-caught typo fixed before commit; 1 verifier-tool false negative documented for posterity)
**Impact on plan:** Zero. Plan executed exactly as specified at the source-code level. Both deviations were caught + handled cleanly.

## Issues Encountered

- **Pre-existing dirty repo state** (`STATE.md`, deleted `setup.py`, modified `mm_engine.py`, untracked `.claude/`, `example_abs_check.py`, `src/main/python/riscv/gtx/data/`, `test/`, `uv.lock`, `vendor/spike` submodule pointer). Each task commit staged ONLY the task-specific file via `git add <explicit path>` (never `git add -A` or `git add .`) per CLAUDE.md / Karpathy "Surgical Changes" guideline. Pre-existing dirty state untouched.
- **`pytest --pylint --mypy` flags from `pyproject.toml [tool.pytest.ini_options] addopts`** rejected by the local pytest install (those plugins not installed in the dev env). Verification commands ran with `-o addopts=` to override. Does NOT affect cibuildwheel CI matrix where all dev extras are installed.
- **rtk shell wrapper munges `grep` quoting in some commands** — workaround: invoke `grep` directly without `rtk` for pattern matching, use `rtk proxy` only for pytest commands. Identical to 07-01-SUMMARY environment quirk.

## Known Stubs

The following items are INTENTIONALLY left for downstream plans per the Plan 02 wave-gating contract:

| File | Stub | Resolved by |
|------|------|-------------|
| `tests/gtx/test_njit_parity.py` (25 entries) | `pytest.skip(f"Plan 03/04 GREEN-fills parity body for {kernel_name}")` for vec/act kernels | Plans 07-03 (vec) + 07-04 (act) |
| `tests/gtx/_njit_helpers.py` line 109 | `generate_test_inputs` returns Wave 0 placeholder for ALL kernels | Plans 07-03/04 specialize per-kernel signature (or each plan inlines as Plan 02 did) |

These stubs are appropriate for Plan 02's gemm-only scope. Plan 03 (vec) and Plan 04 (act) will GREEN-fill their respective branches via the additive-friendly `kernel_name` dispatch already in place.

## User Setup Required

None - no external service configuration required.

**W7 documentation note (per plan output spec):** Cold-start compile time grows ~2× for `_gemm_core_njit` because numba lazy-typed dispatch produces a separate specialization per `has_bias` value (False / True paths trace different bytecode branches). Acceptable for Phase 7 (28 specializations × ~640ms = ~18s aggregate cold compile, amortized via cache=True). Potential future optimization (deferred to v1.x or v2): unified always-add-bias path with zero-fill `bias_fp32` sentinel — collapses to a single specialization. Not a Phase 7 fix; documented here only.

## Next Phase Readiness

- **Plan 07-03 (Wave 1a, VEC JIT)** can now use the established re-call pattern + test dispatch as a template for the 7 vec kernels. Adds `elif kernel_name in VEC_KERNELS:` branch to `test_kernel_parity` without touching gemm logic.
- **Plan 07-04 (Wave 1a, ACT JIT)** independent of Plan 03. Same pattern for the 18 act kernels (5 will use `objmode` escape per D-09).
- **Plan 07-05 (Wave 1b, vendor 84-op sweep + perf)** depends on Plans 02-04 GREEN — Plan 02's gemm portion is now ready.
- **Plan 07-06 (Wave 2, docs/CI sync)** independent — no new blockers.
- **Zero regressions:** 287 P4/P5/P6 tests still pass + 17 P4 strict-mode mm tests pass.
- **No blockers.**

## Self-Check: PASSED

Verified after writing SUMMARY:

- `src/main/python/riscv/gtx/gemm_core.py` — FOUND (modified, contains `_gemm_core_impl`, `_gemm_reduce_sum_a_impl`, `_gemm_dot_impl`, 3× `njit(cache=True)`, `from ._jit import njit`)
- `tests/gtx/test_njit_parity.py` — FOUND (modified, contains `_run_gemm_parity`, `GEMM_KERNELS`, `view(np.uint16)`)
- Commit `38c6344` — FOUND in `git log`
- Commit `9e82a36` — FOUND in `git log`
- 3 `.nbi` cache files under `src/main/python/riscv/gtx/__pycache__/` — FOUND
- `pytest tests/gtx/test_njit_parity.py -k gemm -v` — 3 passed
- `pytest tests/gtx/test_njit_parity.py --no-cov -q` — 4 passed + 25 skipped
- `pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py tests/gtx/test_regression_fw_mm.py --no-cov -q` — 17 passed
- Wider P4/P5/P6 suite — 287 passed + 11 skipped (zero regression)

---
*Phase: 07-numba*
*Completed: 2026-05-09*
