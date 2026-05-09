---
phase: 07-numba
plan: 01
subsystem: testing
tags: [numba, pytest-benchmark, jit, fp32-only, vendor-golden, red-scaffold]

# Dependency graph
requires:
  - phase: 04-mm-subsystem
    provides: gemm_core 3-loop FP32 explicit-accumulate (P7 _impl boundary)
  - phase: 05-vec-act-pool
    provides: vec_core 7 kernels + act_core 7 act / 2 pool / 9 cvt (P7 _impl boundary)
  - phase: 06-verification-wheel
    provides: scripts/import_vendor_golden.py P6 9-op vendor->golden converter (extension target)
provides:
  - riscv.gtx._jit module exposing HAS_NUMBA + njit shim (passthrough when numba absent, real numba.njit when present)
  - pyproject.toml [project.optional-dependencies] fast = ["numba>=0.61.2,<0.66"]
  - pyproject.toml dev extras += pytest-benchmark>=4.0,<6
  - tests/gtx/_njit_helpers.py 28-kernel registry (gemm 3 + vec 7 + act 18) with lazy-import discipline
  - tests/gtx/test_njit_parity.py Tier 1 RED scaffold (28 parametrized + 1 active HAS_NUMBA detection)
  - tests/gtx/test_regression_fw_full_sweep.py Tier 2 RED scaffold (84 parametrized vendor ops)
  - tests/gtx/test_njit_perf.py Tier 3 RED scaffold (2 placeholder benchmarks)
  - tests/gtx/conftest.py + _numba_available + baseline_walltime fixtures (existing fixtures preserved)
  - scripts/import_vendor_golden.py + VENDOR_OPS_84 list (84 entries) + _discover_vendor_ops + --all flag stub
affects: [07-02, 07-03, 07-04, 07-05, 07-06]

# Tech tracking
tech-stack:
  added: [numba (optional, via spike[fast] extras), pytest-benchmark (dev only)]
  patterns:
    - Lazy import + auto NumPy fallback (HAS_NUMBA gate; mirrors P3 _RISCV_AVAILABLE pattern)
    - Lazy importlib lookup in test registries (parametrize collection works before _impl aliases land)
    - Auto-discovery + inlined constant cross-validation (VENDOR_OPS_84 vs filesystem walk)

key-files:
  created:
    - src/main/python/riscv/gtx/_jit.py
    - tests/gtx/_njit_helpers.py
    - tests/gtx/test_njit_parity.py
    - tests/gtx/test_regression_fw_full_sweep.py
    - tests/gtx/test_njit_perf.py
  modified:
    - pyproject.toml
    - tests/gtx/conftest.py
    - scripts/import_vendor_golden.py

key-decisions:
  - "njit shim handles both call patterns (@njit bare and @njit(kw) parenthesized) without runtime branching cost when numba is installed"
  - "VENDOR_OPS_84 inlined as a literal list (W2 fix) but cross-validated against filesystem at runtime via _discover_vendor_ops to catch vendor submodule drift"
  - "_njit_helpers uses lazy importlib at call-site (not module-import) so Wave 0 collection succeeds before Plans 02-04 land _impl aliases"
  - "test_njit_perf.py uses pytest.importorskip on pytest_benchmark so the file collects cleanly in environments without dev extras"
  - "test_has_numba_detection is the only ACTIVE test in the parity scaffold — sentinel that guards NJIT-01 contract from regression"

patterns-established:
  - "Lazy import boundary: HAS_NUMBA module-top in central _jit.py (not per-core) — single source-of-truth for the optional dep"
  - "Test registry lazy-resolution: get_public_fn / get_impl_fn use importlib at call-time so parametrize names exist before targets do"
  - "Auto-discovery cross-validation: every inlined constant that mirrors filesystem state validates against the live tree on import"
  - "Stub flag clearly marks deferred work: every pytest.skip mentions the GREEN-filling plan number"

requirements-completed: [NJIT-01, NJIT-07]

# Metrics
duration: 8m48s
completed: 2026-05-09
---

# Phase 7 Plan 01: Wave 0 NJIT Scaffold Summary

**Lazy numba shim (`HAS_NUMBA` + passthrough njit), pyproject `fast` extras (`numba>=0.61.2,<0.66`), and three RED test tiers (28 parity + 84 vendor sweep + 2 perf benchmarks) wired through a lazy 28-kernel registry — Wave 1 plans can now drop in `_impl` aliases without bootstrapping infrastructure.**

## Performance

- **Duration:** 8m 48s
- **Started:** 2026-05-09T06:08:13Z
- **Completed:** 2026-05-09T06:17:01Z
- **Tasks:** 3 (all autonomous)
- **Files modified:** 8 (5 created + 3 modified)

## Accomplishments

- `riscv.gtx._jit` module ships HAS_NUMBA detection + njit passthrough — `@njit(cache=True)` works identically whether numba is installed or absent.
- `pyproject.toml` declares `[project.optional-dependencies] fast = ["numba>=0.61.2,<0.66"]` (NJIT-07 install path; CI test-extras integration deferred to Plan 06) and adds `pytest-benchmark>=4.0,<6` to `dev`.
- 28-kernel registry (`tests/gtx/_njit_helpers.py`) wires gemm_core (3) + vec_core (7) + act_core (18) to lazy-resolved (`module_path`, `public_fn`, `_impl_fn`) tuples; constants `ALL_NJIT_KERNEL_NAMES` and `TRANSCENDENTAL_KERNELS = {gelu, tanh_act, sigmoid, softmax, esum}` exposed for downstream parametrize use.
- Three RED test tiers landed and collection-clean:
  - **Tier 1** `test_njit_parity.py` — 28 parametrized `pytest.skip` placeholders + 1 active `test_has_numba_detection` sentinel guarding NJIT-01.
  - **Tier 2** `test_regression_fw_full_sweep.py` — auto-discovers 84 vendor op directories, parametrizes one skip per op (Plan 05 GREEN-fills the strict-mode subprocess body).
  - **Tier 3** `test_njit_perf.py` — `importorskip("pytest_benchmark")` + 2 placeholder benchmarks (Plan 05 GREEN-fills).
- `scripts/import_vendor_golden.py` extended with `VENDOR_OPS_84` (84 entries inlined; cross-validated against `vendor/gtx_cpp_reference/test/` filesystem on every load via `_discover_vendor_ops()`) + `--all` argparse flag that raises `NotImplementedError` until Plan 05 GREEN. Existing 9-op `VENDOR_TO_PYSPIKE_OPS` map and `--verify` flow preserved verbatim (P6 lineage intact).
- Conftest fixtures `_numba_available` + `baseline_walltime` appended; existing `riscv_available`, `proc`, `insn_factory`, `proc_with_addra_addrr_seeded` preserved verbatim.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create `_jit.py` shim + extend `pyproject.toml` extras** — `a159365` (feat)
2. **Task 2: Create 28-kernel registry + extend conftest with fixtures** — `4076097` (test)
3. **Task 3: Create 3 RED test scaffolds + extend `scripts/import_vendor_golden.py`** — `a4726bb` (test)

**Plan metadata:** to be added by `final_commit` step (SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md).

## Files Created/Modified

- `src/main/python/riscv/gtx/_jit.py` (created, 57 LOC) — Lazy numba shim. `HAS_NUMBA: bool` + `njit(*args, **kwargs)` passthrough decorator handling both `@njit` and `@njit(cache=True)` call patterns.
- `pyproject.toml` (modified, +4 LOC) — `fast` extras group + `pytest-benchmark` in dev (alphabetically positioned between `pytest-asyncio` and `pytest-cov`).
- `tests/gtx/_njit_helpers.py` (created, 111 LOC) — `ALL_NJIT_KERNELS` 28-tuple registry + `get_public_fn` / `get_impl_fn` / `generate_test_inputs` lazy helpers + `TRANSCENDENTAL_KERNELS` set.
- `tests/gtx/conftest.py` (modified, +25 LOC at file end) — `_numba_available` + `baseline_walltime` fixtures.
- `tests/gtx/test_njit_parity.py` (created, 48 LOC) — Tier 1 scaffold; 1 active sentinel + 28 parametrized skips.
- `tests/gtx/test_regression_fw_full_sweep.py` (created, 48 LOC) — Tier 2 scaffold; auto-discovers 84 vendor ops at collection time.
- `tests/gtx/test_njit_perf.py` (created, 37 LOC) — Tier 3 scaffold; `importorskip` + 2 placeholder benchmarks.
- `scripts/import_vendor_golden.py` (modified, +46 LOC) — `VENDOR_OPS_84` list + `_discover_vendor_ops()` + `--all` argparse flag + NotImplementedError stub.

## Decisions Made

- **Lambda-cache acceptance criterion is unsatisfiable in numba-installed environments** — used a real `def double(x)` in a temp file for verification (see Deviations §1). The shim itself is correct and supports both `@njit` and `@njit(cache=True)` for normal `def`-functions.
- **`_njit_helpers.py` uses lazy `importlib.import_module` inside `get_public_fn`/`get_impl_fn`** rather than module-top imports. This is the *only* way Wave 0 collection can succeed before Plans 02-04 land the `_impl` aliases — module-top would AttributeError at import time.
- **`test_regression_fw_full_sweep.py` auto-discovers from filesystem** rather than importing `VENDOR_OPS_84` from the script. Reason: the test should reflect actual vendor submodule state, not a stale inlined list. The script's inlined list is a separate document for ergonomics + audit trail.
- **`test_has_numba_detection` is the only test that ACTIVELY runs in the parity tier today** — chosen as the NJIT-01 sentinel because it guards the most important contract (HAS_NUMBA reflects environment), yet costs ~zero (no compilation, no FP work).
- **`importorskip("pytest_benchmark")` at module-top of perf scaffold** rather than per-test `@pytest.mark.skipif` — collection skips entire file when dev extras absent, keeping CI matrices clean.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan acceptance criterion `njit(cache=True)(lambda x: x*2)` is unsatisfiable when numba is installed**

- **Found during:** Task 1 verification.
- **Issue:** Plan acceptance criterion line:
  > `python -c "from riscv.gtx._jit import njit; fn = njit(cache=True)(lambda x: x*2); assert fn(21)==42"` exits 0 — works whether numba is installed or not (W1: single-line lambda).

  When numba is installed, this raises `RuntimeError: cannot cache function '<lambda>': no locator available for file '<string>'` because numba's disk cache requires a real source-file locator (lambdas have `__name__ == '<lambda>'` and `python -c` source location is `<string>`). The plan even called this out as "(W1: single-line lambda)" — a known quirk left in the criterion.
- **Fix:** Verified the shim functionally with a real `def double(x): return x * 2` in a tempfile, exercising both `@njit(cache=True)` and bare `@njit` forms. The shim itself is **unchanged** — it correctly delegates to `numba.njit` when present and to passthrough when absent. Only the test command needed updating.
- **Files modified:** None (shim correct as-shipped per RESEARCH §"Pattern 1").
- **Verification:** Both `@njit(cache=True) def double(x): return x*2` and bare `@njit def triple(x): return x*3` return correct results in the numba-installed env (tested in `/tmp/test_jit_shim.py`). All other Task 1 acceptance criteria pass: `HAS_NUMBA=True`, `pyproject.toml` declares `fast = ["numba>=0.61.2,<0.66"]`, `pytest-benchmark>=4.0,<6` in dev.
- **Committed in:** `a159365` (Task 1 commit) — no code change needed; this is a verification-script issue documented for Plan 02 (which will write the first real `@njit(cache=True) def _gemm_core_impl(...)`).

**2. [Rule 1 - Cosmetic / Tooling] gsd-tools `verify key-links` reports false-negative on third link's regex pattern**

- **Found during:** Plan-level verification (post-Task 3).
- **Issue:** `gsd-tools verify key-links` reports the third link (`pyproject.toml -> [project.optional-dependencies] fast`) as not verified because the pattern `numba>=0\\.61\\.2` (escaped-dot regex) is being treated as a literal-string search rather than a regex match. The actual file contains `numba>=0.61.2,<0.66` and a regex-aware verifier would match.
- **Fix:** No code change. The plan's other two key-links verify correctly. Functional verification is unambiguous (`tomllib.load(...).optional-dependencies.fast == ['numba>=0.61.2,<0.66']`).
- **Files modified:** None.
- **Verification:** Direct `tomllib`/`tomli` parse confirms the extras group is exactly `['numba>=0.61.2,<0.66']`. Plain `grep -c 'numba>=0\.61\.2,<0\.66' pyproject.toml` returns 1.
- **Committed in:** N/A (verifier tool quirk, not source code).

---

**Total deviations:** 2 (both verification-tooling issues; no source-code changes required)
**Impact on plan:** Zero. Plan executed exactly as specified at the source-code level. Both deviations are documentation of acceptance-criterion / tooling quirks that future plans should be aware of.

## Issues Encountered

- **Plan executes against pyspike repo with several pre-existing unrelated dirty files** (`STATE.md`, `setup.py` D-deletion, `mm_engine.py` modifications, untracked `.claude/`, `example_abs_check.py`, `src/main/python/riscv/gtx/data/`, `test/`, `uv.lock`, `vendor/spike` submodule pointer). Each task commit staged ONLY task-related files via `git add <explicit paths>` (never `git add -A` or `git add .`) per CLAUDE.md / Karpathy "Surgical Changes" guideline. Pre-existing dirty state untouched.
- **`pytest --pylint --mypy` flags from `pyproject.toml [tool.pytest.ini_options] addopts`** rejected by the local pytest install (those plugins not installed in the dev env). Verification commands ran with `-o "addopts="` to override. Does NOT affect cibuildwheel CI matrix where all dev extras are installed.
- **`tomllib` not available in Python 3.10** (stdlib starting 3.11). Used `tomli` fallback in verification commands. The shim and pyproject.toml are unaffected — only the verification probe needed adjustment.

## Known Stubs

The Wave 0 RED scaffold INTENTIONALLY leaves the following as placeholders. These are NOT bugs — each is the explicit hand-off point for a future plan:

| File | Line | Stub | Resolved by |
|------|------|------|-------------|
| `tests/gtx/test_njit_parity.py` | 40 | `pytest.skip(f"Plan 02/03/04 GREEN-fills parity body for {kernel_name}")` | Plans 07-02 (gemm), 07-03 (vec), 07-04 (act) |
| `tests/gtx/test_regression_fw_full_sweep.py` | 48 | `pytest.skip(f"Plan 05 GREEN-fills vendor 84-op sweep (op={op_dir})")` | Plan 07-05 |
| `tests/gtx/test_njit_perf.py` | 32, 37 | 2× `pytest.skip("Plan 05 GREEN-fills ...")` | Plan 07-05 |
| `tests/gtx/_njit_helpers.py` | 109 | `generate_test_inputs` returns `(rng.random(16,...).astype(fp16),)` placeholder for ALL kernels | Plans 07-02/03/04 specialize per-kernel signature |
| `tests/gtx/conftest.py` | 113 | `baseline_walltime` returns `0.0` when `tests/gtx/data/baseline_walltime.txt` absent | Plan 07-05 first task records the real measurement |
| `scripts/import_vendor_golden.py` | 142 | `--all` flag raises `NotImplementedError` | Plan 07-05 GREEN-fills the 84-op converter |

All stubs are appropriate for Wave 0's scaffold-only scope (per plan objective) and downstream plans have explicit GREEN-fill task assignments.

## User Setup Required

None - no external service configuration required. The `numba` optional dependency is automatically installed via `pip install spike[fast]` when users opt in. CI test-extras integration is deferred to Plan 06 per NJIT-07.

## Next Phase Readiness

- **Plan 07-02 (Wave 1a, GEMM JIT)** can begin immediately — has access to `_jit.py` (`from ._jit import njit, HAS_NUMBA`), `_njit_helpers.ALL_NJIT_KERNELS` registry rows for the 3 gemm kernels, and a parity test scaffold to drop GREEN bodies into.
- **Plan 07-03 (Wave 1a, VEC JIT)** independent of Plan 02 — same scaffolding ready for the 7 vec kernels.
- **Plan 07-04 (Wave 1a, ACT JIT)** independent of Plans 02/03 — same scaffolding ready for the 18 act kernels (5 will use `objmode` escape per D-09).
- **Plan 07-05 (Wave 1b, sweep + perf)** depends on Plans 02-04 GREEN — has access to `VENDOR_OPS_84` list, `_discover_vendor_ops()` cross-validator, `--all` argparse stub, `baseline_walltime` fixture wired, and `pytest_benchmark` `importorskip`.
- **Plan 07-06 (Wave 2, docs/CI sync)** independent — has access to all delivered scaffolding.
- **No blockers.** Existing 287 P4/P5/P6 tests still pass (0 regressions).

## Self-Check: PASSED

Verified after writing SUMMARY:

- `src/main/python/riscv/gtx/_jit.py` — FOUND
- `tests/gtx/_njit_helpers.py` — FOUND
- `tests/gtx/test_njit_parity.py` — FOUND
- `tests/gtx/test_regression_fw_full_sweep.py` — FOUND
- `tests/gtx/test_njit_perf.py` — FOUND
- `scripts/import_vendor_golden.py` — FOUND (modified)
- `pyproject.toml` — FOUND (modified)
- `tests/gtx/conftest.py` — FOUND (modified)
- Commit `a159365` — FOUND
- Commit `4076097` — FOUND
- Commit `a4726bb` — FOUND

---
*Phase: 07-numba*
*Completed: 2026-05-09*
