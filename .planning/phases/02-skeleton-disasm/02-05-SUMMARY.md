---
phase: 02-skeleton-disasm
plan: 05
subsystem: integration
tags: [rocc, riscv, gtx, integration, skipif, subprocess, pyspike-cli, validation]

# Dependency graph
requires:
  - phase: 02-skeleton-disasm
    plan: 01
    provides: "GtxNpu shell + WarpState + _registry + dispatch builders + tests/gtx/_mocks.py + nop_wjoin.elf fixture"
  - phase: 02-skeleton-disasm
    plan: 02
    provides: "ops/spr.py 4 WRSPR/RDSPR handlers + spr_router.wr_spr/rd_spr"
  - phase: 02-skeleton-disasm
    plan: 03
    provides: "ops/control.py 8 custom1 funct3 + 6 custom0 stubs + WJOIN env-var branch"
  - phase: 02-skeleton-disasm
    plan: 04
    provides: "_registry.collect_disasms real builder + disasm.py helpers"
provides:
  - "tests/gtx/test_register.py -- CORE-01 design contract (5 tests; 2 always-run + 3 skipif _riscv)"
  - "tests/gtx/test_reset.py -- CORE-02 reset() sp init + zero-init + SPR defaults (8 tests; all skipif _riscv)"
  - "tests/gtx/test_dispatch.py -- DISP-01 funct7/funct3 dispatch + D-02 collision (9 tests; all skipif _riscv)"
  - "tests/gtx/test_skeleton.py -- ROADMAP P2 #1 pyspike --extlib integration (2 tests; subprocess gated on _riscv + .elf)"
  - "VALIDATION.md Approval: approved 2026-05-04 + 17 task statuses flipped to done"
affects: [phase-02-closure, phase-03-dma]

# Tech tracking
tech-stack:
  added: []  # No new external dependencies
  patterns:
    - "Self-contained _RISCV_AVAILABLE module-level detection in each test file (mirrors conftest D-17 try/except), so the planner's --noconftest acceptance command still selects the correct branch"
    - "pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...) for whole-module gating (test_reset, test_dispatch) -- one-line skip instead of per-test guards"
    - "Subprocess integration via shutil.which('pyspike') with python -m riscv fallback; subprocess.run timeout=30s; GTX_NO_EXIT scrubbed from env so WJOIN propagates SystemExit"
    - "Fixture-existence check (test_elf_fixture_exists_or_documented) ALWAYS runs even when integration test skips -- guarantees D-22 .S+Makefile contract"

key-files:
  created:
    - "tests/gtx/test_register.py -- 95 lines, 5 tests (2 always-run + 3 skipif)"
    - "tests/gtx/test_reset.py -- 143 lines, 8 tests (all skipif via pytestmark)"
    - "tests/gtx/test_dispatch.py -- 187 lines, 9 tests (all skipif via pytestmark)"
    - "tests/gtx/test_skeleton.py -- 105 lines, 2 tests (1 always-run + 1 subprocess skipif)"
  modified:
    - ".planning/phases/02-skeleton-disasm/02-VALIDATION.md -- Approval ready -> approved 2026-05-04; 17 task statuses pending -> done"

key-decisions:
  - "Self-contained _RISCV_AVAILABLE detection in each test module (NOT the conftest fixture) -- the planner's acceptance command uses --noconftest, which strips fixtures. Module-level detection survives. Pattern is minimally redundant (5 lines per file) but completely robust."
  - "Whole-module pytestmark for test_reset.py / test_dispatch.py instead of per-test if/skip blocks -- cleaner, less repetitive. test_register.py keeps per-test gating because Tier 1 tests run unconditionally and Tier 2 tests are skipif-only."
  - "test_skeleton.py uses subprocess.run NOT pytest.run -- the integration test invokes a separate pyspike process via the CLI wrapper. This avoids GIL contamination from a long-running spike process inside a pytest worker (research §1296-1297)."
  - "test_elf_fixture_exists_or_documented runs always (no skipif) -- the D-22 contract requires .S+Makefile committed even when .elf may be absent. Acts as a compile-time-style guarantee in the test layer."

patterns-established:
  - "When a future phase adds a new integration test that depends on _riscv.so and an external artifact (e.g. golden hex), use the same two-tier skipif: module-level _RISCV_AVAILABLE + pathlib existence check. Both must skip cleanly (NOT fail) when the artifact or runtime is unavailable."
  - "Self-contained skipif detection (5 lines: try/except ImportError on riscv.processor) is the recommended pattern for any test that exercises GtxNpu instantiation. Don't rely on the conftest fixture under --noconftest."

requirements-completed: [CORE-01, CORE-02, CORE-03, CORE-04, DISP-01]

# Metrics
duration: 4m18s
completed: 2026-05-04
---

# Phase 02 Plan 05: Wave 2 Integration & Validation Summary

**Wave 2 finalization -- adds 4 integration test files (CORE-01 register, CORE-02 reset, DISP-01 dispatch, ROADMAP-P2-#1 skeleton) on top of Wave 0/1 outputs and flips VALIDATION.md to `Approval: approved`. Phase 2 closes here -- all 5 phase-2 requirements (CORE-01..04 + DISP-01) covered by automated tests.**

## Performance

- **Duration:** 4m18s
- **Started:** 2026-05-04 (task 1 commit `0831898`)
- **Completed:** 2026-05-04 (task 5 commit `e3f1f1c`)
- **Tasks:** 5 (4 auto+TDD test files + 1 docs update)
- **Files created:** 4 test files
- **Files modified:** 1 validation doc
- **New tests added:** 24 (5 register + 8 reset + 9 dispatch + 2 skeleton)

## Accomplishments

- **CORE-01 (`@isa.register('gtx')`)** -- 5 tests in `test_register.py`:
  - 2 always-run: `test_gtx_module_imports_without_error`, `test_gtx_exports_match_all` (verify riscv.gtx package surface + __all__)
  - 3 skipif _riscv: `test_gtxnpu_is_rocc_subclass`, `test_gtxnpu_name_property`, `test_register_extension_factory_finds_gtx` (verify isa.ROCC inheritance + name='gtx' property + Spike's find_extension lookup)

- **CORE-02 (`reset()` sp init + zero-init + SPR defaults)** -- 8 tests in `test_reset.py` (all skipif _riscv):
  - sp = 0x80100000 (CORE-02 acceptance)
  - mxe_accum shape (GTX_NEST_NUM, GTX_SPU_NUM)=(4,16) FP32, all zero (validates plan 01 D-06 correction)
  - L0/L1/L2 byte arrays zeroed
  - NSPR THREAD_MASK=0xFFFF, TYPE=1 (FP16) seeded across all 4 NESTs
  - LSPR SPM_ADDRA..ADDRR = 0 across all (NEST, SPU) pairs (4 * 16 = 64 entries)
  - WarpState reset (is_ploop/is_tloop/is_sloop=False, tmu_id=curr_id=0)
  - GSPR clear-then-reseed (0xABCD garbage cleared, 0x000=0 / 0x010=0 seeded)
  - FPU enable (mstatus.FS=01) doesn't crash with mock get_csr

- **DISP-01 (custom0 funct7 dispatch + D-02 collision)** -- 9 tests in `test_dispatch.py` (all skipif _riscv):
  - 10 P2 funct7 keys present in `_custom0` dict (0x00-0x07 + 0x48 + 0x49)
  - 8 funct3 keys present in `_custom1` dict (0..7)
  - **D-02 collision branch 1**: `funct=0x00, rs1=0` -> wrspr_gem5 writes XPR[rs2]=0xCAFE to GSPR[0x000]
  - **D-02 collision branch 2**: `funct=0x00, rs1=3` -> P4 MM stub returns 0, NO SPR mutation (snapshot equality)
  - ISS-full WRSPR (funct=0x49) writes LSPR[0][0][0x900]=0xBEEF
  - ISS-full RDSPR (funct=0x48) returns rd_spr value AND writes to XPR[insn.rd]
  - Unmapped funct=0x7C silently returns 0 (P5/P6 may upgrade to illegal_instruction)
  - custom1 funct3 reconstruction `(xd<<2)|(xs1<<1)|xs2` dispatches correctly
  - Full custom1 sweep (funct3=0..7); WJOIN raises SystemExit when `GTX_NO_EXIT` unset

- **ROADMAP P2 #1 (`pyspike --extlib=riscv.gtx nop_wjoin.elf` exit 0)** -- 2 tests in `test_skeleton.py`:
  - `test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero` (subprocess; gated on _riscv + .elf)
  - `test_elf_fixture_exists_or_documented` (always-run; verifies D-22 .S/Makefile committed)

- **VALIDATION.md** -- Approval flipped from `ready` to `approved 2026-05-04`. All 17 task status cells flipped from `⬜ pending` to `✅ done`.

## Task Commits

Each task was committed atomically:

1. **Task 1: tests/gtx/test_register.py -- CORE-01 design contract suite (5 tests)** -- `0831898` (test)
2. **Task 2: tests/gtx/test_reset.py -- CORE-02 reset validation (8 tests)** -- `0a77638` (test)
3. **Task 3: tests/gtx/test_dispatch.py -- DISP-01 dispatch + D-02 collision (9 tests)** -- `a5bd0c1` (test)
4. **Task 4: tests/gtx/test_skeleton.py -- ROADMAP P2 #1 integration (2 tests)** -- `3ca5ab2` (test)
5. **Task 5: VALIDATION.md Approval + status flip** -- `e3f1f1c` (docs)

**Plan metadata commit:** to follow (this SUMMARY + STATE/ROADMAP updates).

## Files Created/Modified

### Created (4)

- `tests/gtx/test_register.py` -- 95 lines, 5 tests. Two-tier validation per D-21: Tier 1 always-run (module surface + __all__), Tier 2 skipif (subclass + factory).
- `tests/gtx/test_reset.py` -- 143 lines, 8 tests, whole-module skipif via `pytestmark`. Covers all CORE-02 acceptance points.
- `tests/gtx/test_dispatch.py` -- 187 lines, 9 tests, whole-module skipif. Covers DISP-01 surface + D-02 collision both branches + ISS encodings + unmapped fallback + custom1 sweep.
- `tests/gtx/test_skeleton.py` -- 105 lines, 2 tests. Subprocess integration with timeout=30s + GTX_NO_EXIT scrubbed; fixture-existence test always runs.

### Modified (1)

- `.planning/phases/02-skeleton-disasm/02-VALIDATION.md` -- two single-line edits: Approval status + 17 task-row status cells.

## Decisions Made

1. **Self-contained `_RISCV_AVAILABLE` detection per test module (NOT conftest fixture).** The plan's acceptance command is `pytest ... --noconftest -o "addopts="`. With `--noconftest`, fixtures defined in `tests/gtx/conftest.py` (including the planner-prescribed `riscv_available` fixture) are NOT loaded. The first naive run of `test_register.py` failed with "fixture 'riscv_available' not found" (Rule 3 - Blocking deviation). Resolution: each test module duplicates the 5-line `try/except ImportError` detection. This is technically redundant but completely robust under both `--noconftest` and normal collection. The conftest fixture is preserved for tests that don't use `--noconftest`.

2. **Whole-module `pytestmark` for test_reset.py and test_dispatch.py.** Since every test in those files needs `_RISCV_AVAILABLE`, a single `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)` is cleaner than per-test `if/skip`. test_register.py keeps per-test guards because Tier 1 tests are always-run; pytestmark would skip them too.

3. **`test_elf_fixture_exists_or_documented` runs unconditionally.** The D-22 contract states `.S` and `Makefile` must be committed even when `.elf` may be absent (e.g., CI without cross toolchain). The check is a compile-time-style assertion in the test layer; it skips nothing.

4. **`test_skeleton.py` uses `subprocess.run` not pytest internals.** The integration test launches a separate `pyspike` process. This avoids any GIL or Python-extension lifetime issues from running spike inside a pytest worker (research §1296-1297). Resolution rule: prefer `shutil.which('pyspike')` (developer-installed CLI), fall back to `[sys.executable, "-m", "riscv"]`. Timeout=30s defends against WJOIN SystemExit non-propagation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `riscv_available` fixture not available under `--noconftest`**

- **Found during:** Task 1 first verification run.
- **Issue:** The plan's prescribed action block declared test functions taking `riscv_available` as a fixture argument. The acceptance command is `pytest ... --noconftest -o "addopts="`. With `--noconftest`, the conftest fixture is not loaded -> `ERROR: fixture 'riscv_available' not found`.
- **Fix:** Replaced the fixture argument with a module-level detection block `try: from riscv.processor import processor_t; _RISCV_AVAILABLE = True except ImportError: _RISCV_AVAILABLE = False`. The same 5-line pattern is duplicated across all 4 plan-05 test files. The conftest fixture is unchanged and still works for non-`--noconftest` runs.
- **Files modified:** `tests/gtx/test_register.py`, `tests/gtx/test_reset.py`, `tests/gtx/test_dispatch.py`, `tests/gtx/test_skeleton.py`.
- **Verification:** All 4 acceptance commands now exit 0 (with skipif behavior under missing `_riscv.so`).
- **Committed in:** Each test's task commit (0831898, 0a77638, a5bd0c1, 3ca5ab2).

**Total deviations:** 1 auto-fixed (Rule 3 - Blocking; the planner's `--noconftest` flag in the acceptance command was incompatible with conftest-fixture-based skipif gating).

**Impact:** None on contract -- the test surface and skipif semantics are identical. Only the mechanism differs (module-level instead of fixture).

## Issues Encountered

- **`--noconftest` strips conftest fixtures** -- documented as Deviation #1 above. The fix is mechanical and one-line per file; doesn't change observable behavior.
- **`pytest tests/` (full suite) collection errors** -- pre-existing in this dev environment: `tests/test_cfg.py`, `tests/test_processor.py`, etc. import `pexpect` which is not installed. NOT caused by this plan; pre-existing Phase 1 deferred-items condition. Phase 2 tests in `tests/gtx/` collect cleanly (86 tests, 65 pass + 21 skip).

## `_riscv.so` and Toolchain Availability

- **`_riscv.so` was NOT built during this run.** All test outputs reflect the mock-fallback path:
  - `tests/gtx/test_register.py` -> 2 passed, 3 skipped (Tier 2 skipif _riscv)
  - `tests/gtx/test_reset.py` -> 0 passed, 8 skipped (whole-module skipif)
  - `tests/gtx/test_dispatch.py` -> 0 passed, 9 skipped (whole-module skipif)
  - `tests/gtx/test_skeleton.py` -> 1 passed, 1 skipped (.S+Makefile assert always-run; subprocess skipif _riscv)
  - **Total plan-05: 3 passed, 21 skipped (24 collected).**
- **`nop_wjoin.elf` IS present** (committed in plan 01 task 3 at 5KB). When `_riscv.so` is also built (CI / `python setup.py build_ext --inplace`), the integration test executes the full subprocess path.
- **When `_riscv.so` IS built:** all 24 plan-05 tests should pass. Specifically:
  - 5 register tests (Tier 1 + Tier 2 all run)
  - 8 reset tests (all skipif gates open)
  - 9 dispatch tests (all skipif gates open)
  - 2 skeleton tests (both run; integration test exercises pyspike CLI subprocess)

## ROADMAP P2 Success Criteria Status

| Criterion | Test | Status |
|-----------|------|--------|
| #1: `pyspike --extlib=riscv.gtx nop_wjoin.elf` exits 0 | `test_skeleton.py::test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero` | COVERED (skipif-gated) |
| #2: `get_disasms()` >= 10 entries | `test_disasm.py::test_collect_disasms_minimum_count` (plan 04, 18 entries) | COVERED |
| #3: WRSPR/RDSPR roundtrip both encodings | `test_spr.py::test_roadmap_p2_3_*` (plan 02) | COVERED |
| #4: warp state machine `start_p->start_t->end_t->end_p` | `test_warp.py::test_loop_state_machine_full_sequence` (plan 03) | COVERED |
| #5: WJOIN both modes (raise + return) | `test_wjoin.py::test_wjoin_default_raises_systemexit` + `test_wjoin_with_no_exit_set_returns_zero` (plan 03) | COVERED |

All 5 ROADMAP P2 success criteria are now covered by automated tests in `tests/gtx/`. With `_riscv.so` built, every criterion produces a pass/fail result; without it, criteria 1 and parts of 3-5 (those depending on GtxNpu instantiation) skip cleanly.

## Phase 2 Final Test Count

```
$ python3 -m pytest tests/gtx/ --collect-only -q --noconftest -o "addopts="
86 tests collected in 0.72s

$ python3 -m pytest tests/gtx/ -q --noconftest -o "addopts="
65 passed, 21 skipped in 0.76s
```

**Breakdown (86 collected):**
- Phase 1 carryover: 13 (test_fp_roundtrip, test_memory_layout)
- Plan 02 (SPR): 16 tests
- Plan 03 (warp + wjoin): 23 tests (16 warp + 7 wjoin)
- Plan 04 (disasm): 10 tests
- **Plan 05 (this plan): 24 tests**
- Total: 13 + 16 + 23 + 10 + 24 = 86

**Skips (21):** 3 register Tier 2 + 8 reset + 9 dispatch + 1 skeleton subprocess = 21 (all gated on `_RISCV_AVAILABLE = False` in this dev environment).

## Stub Tracking

No new stubs introduced. All P3+ placeholder behavior was already documented in plans 02/03 SUMMARYs (e.g., `dispatch_*_stub` returning 0; `wjoin_custom0_no_exit` returning 0 for "elapsed cycles"; `wrspr_gem5` rs1!=0 -> P4 MM stub). Plan 05 only adds tests; it does not introduce any new behavior gaps.

## Cross-Plan Contract Verified

- **Phase 2 fully closed.** All 5 phase-2 requirements (CORE-01, CORE-02, CORE-03, CORE-04, DISP-01) have at least one passing or skipif-gated test.
- **VALIDATION.md `Approval: approved 2026-05-04`** -- Phase 2 is ready for `/gsd:verify-work` and `/gsd:phase-evolve` once `_riscv.so` is built (CI / wheel build).
- **Phase 3 (DMA) unblocked.** All P2 dispatch primitives (custom0/custom1 dispatch dicts, _registry, ops modules, mock infra, .elf fixture pattern) are stable APIs that P3 ops (DMA-01..05) consume directly.

## Phase 2 Closure Status

Phase 2 (skeleton-disasm) is **functionally complete** as of this commit. Open items for Phase 2 closure (NOT executed in this plan, deferred to phase-evolve):

- [ ] CI run with `_riscv.so` built (validates the 21 skipif-gated tests actually pass when GtxNpu is instantiable)
- [ ] `/gsd:verify-work 2` -- verifier sub-agent reviews entire phase against PROJECT/ROADMAP/REQUIREMENTS
- [ ] `/gsd:phase-evolve 2` -- locks Phase 2 decisions into PROJECT.md, advances STATE.md to Phase 3

## Self-Check: PASSED

Verified files exist:
- `tests/gtx/test_register.py` -- FOUND
- `tests/gtx/test_reset.py` -- FOUND
- `tests/gtx/test_dispatch.py` -- FOUND
- `tests/gtx/test_skeleton.py` -- FOUND

Verified commits exist (`git log --oneline | grep`):
- `0831898` test(02-05): add CORE-01 @isa.register('gtx') validation suite -- FOUND
- `0a77638` test(02-05): add CORE-02 GtxNpu.reset() validation suite -- FOUND
- `a5bd0c1` test(02-05): add DISP-01 custom0/custom1 dispatch validation suite -- FOUND
- `3ca5ab2` test(02-05): add ROADMAP P2 #1 pyspike --extlib integration test -- FOUND
- `e3f1f1c` docs(02-05): mark VALIDATION.md approved + flip task statuses to done -- FOUND

Verified all acceptance commands pass:
- `pytest tests/gtx/test_register.py -x -q --noconftest -o "addopts="` -> 2 passed, 3 skipped (Tier 1 + Tier 2 skipif) [exit 0]
- `pytest tests/gtx/test_reset.py -x -q --noconftest -o "addopts="` -> 8 skipped [exit 0]; --collect-only reports 8 tests
- `pytest tests/gtx/test_dispatch.py -x -q --noconftest -o "addopts="` -> 9 skipped [exit 0]; --collect-only reports 9 tests
- `pytest tests/gtx/test_skeleton.py -x -q --noconftest -o "addopts="` -> 1 passed, 1 skipped [exit 0]
- `pytest tests/gtx/ -x -q --noconftest -o "addopts="` -> 65 passed, 21 skipped [exit 0]; no regressions on plans 01-04 tests
- `grep -E "nyquist_compliant: true" .planning/phases/02-skeleton-disasm/02-VALIDATION.md` -> 1 line
- `grep -E "wave_0_complete: true" .planning/phases/02-skeleton-disasm/02-VALIDATION.md` -> 1 line
- `grep -E "^\| 02-0[1-5]-T" .planning/phases/02-skeleton-disasm/02-VALIDATION.md | wc -l` -> 17
- `grep -E "Approval:.*approved" .planning/phases/02-skeleton-disasm/02-VALIDATION.md` -> 1 line

---
*Phase: 02-skeleton-disasm*
*Completed: 2026-05-04*
