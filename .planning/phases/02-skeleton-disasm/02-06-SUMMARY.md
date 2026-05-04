---
phase: 02-skeleton-disasm
plan: 06
subsystem: testing
tags: [build, pybind11, pytest, regression-discovery, gap-closure]

# Dependency graph
requires:
  - phase: 02-skeleton-disasm
    provides: "All 5 Phase-2 plans landed (CORE-01..04, SPR-01/02, DISASM-01, DISP-01/02 marked Complete)"
  - phase: 01-foundation
    provides: "spike core build (libriscv.so) + pyproject.toml [build-system].requires"
provides:
  - "Validated build path: pip install -e . --no-build-isolation --user produces _riscv.cpython-310-x86_64-linux-gnu.so"
  - "Documented Phase-2 post-build regression surface (4 categories, 15 failures)"
  - "ROADMAP doc-lag fix: 5 plan-05 checkboxes flipped to [x]; Phase 2 main 5/5 complete"
  - "test_full_trace_mnemonics_present regression guard (in tests/gtx/test_skeleton.py)"
affects: [phase-evolve-2, phase-2-followup]

# Tech tracking
tech-stack:
  added: []  # No new deps; setuptools_scm was already declared but missing from env
  patterns:
    - "Build-path validation as gap-closure (NOT scope expansion)"
    - "Honest doc-sync: do NOT flip UAT to passed when underlying behavior fails"
    - "Mock-fallback insufficient for full _riscv.so equivalence (D-17/D-18/D-19 limit)"

key-files:
  created:
    - .planning/phases/02-skeleton-disasm/02-06-BUILD-LOG.md
    - .planning/phases/02-skeleton-disasm/02-06-SUMMARY.md
  modified:
    - tests/gtx/test_skeleton.py
    - .planning/phases/02-skeleton-disasm/02-HUMAN-UAT.md
    - .planning/phases/02-skeleton-disasm/02-VERIFICATION.md
    - .planning/ROADMAP.md

key-decisions:
  - "Build path validated locally with pybind11 3.0.1 (NOT 3.0.4 csr_t-broken version) via --no-build-isolation"
  - "Test-disasm name normalization: real disasm_insn_t maps _ -> . (test expectations need update; out of plan scope)"
  - "ELF Makefile -Ttext=0x80000000 produces LOAD segment at 0x7ffff000 due to .riscv.attributes ordering; -Wl,-Ttext-segment=... is the correct flag"
  - "UAT items NOT flipped to 'passed' because Tasks 2/3 surfaced regressions; flipping would be dishonest"
  - "ROADMAP plan-05 checkbox sync IS independent of gap-closure outcome (plan-05 itself genuinely complete) — applied"

patterns-established:
  - "Build-path validation reveals what mock-fallback hides: each round-trip from offline-pytest to _riscv.so-built-pytest can surface new bugs"
  - "Plan files_modified list is authoritative scope: deviations to forbidden files require new plan, not auto-fix"

requirements-completed: []  # Plan was validating, not adding scope; CORE-01..04 / SPR-01/02 / DISASM-01 / DISP-01/02 already Complete pre-plan

# Metrics
duration: 21min
completed: 2026-05-04
---

# Phase 2 Plan 06: Gap-Closure Cycle Summary

**Build path validated locally — pybind11 3.0.1 + setuptools_scm + libriscv.so produces a working `_riscv.cpython-310-x86_64-linux-gnu.so` and `GtxNpu` hydrates — but running the 21 previously-skipif tests revealed 15 pre-existing bugs in 4 distinct categories that the mock-fallback discipline had hidden, blocking the 3 UAT closure items.**

## Performance

- **Duration:** ~21 min
- **Started:** 2026-05-04T15:49:05Z
- **Completed:** 2026-05-04T16:10:28Z
- **Tasks:** 4 (all attempted; 3 with regressions, 1 partial)
- **Files modified:** 5 (4 created/edited + 1 ROADMAP sync)

## Accomplishments

- **Build path proven locally:** `pip install -e . --no-build-isolation --user` produces `_riscv.cpython-310-x86_64-linux-gnu.so` (1.5M); `from riscv import _riscv` resolves; `from riscv.gtx import GtxNpu` resolves to a real class. The pybind11 3.0.4 csr_t deferred-items issue did NOT resurface because the system has 3.0.1 installed and `--no-build-isolation` reused it.
- **Pre-existing bug surface mapped:** 4 distinct categories (A: 8 failures, B: 6, C: 1, D: 1+) documented with root cause + recommended fix + correct file owner. Future Phase-2 follow-up has a precise checklist.
- **Trace regression guard added:** `test_full_trace_mnemonics_present` in `tests/gtx/test_skeleton.py` (currently fails upstream; will pass once Categories C+D are fixed).
- **ROADMAP doc-lag fixed:** 5 occurrences of plan-05 `[ ]` -> `[x]`; Phase 2 main section now reads `5/5 complete`.
- **Honest doc-sync:** UAT items NOT flipped to `passed`. VERIFICATION status: `human_needed -> needs_followup` (NOT `passed`). `re_verification` block populated with category breakdown.

## Task Commits

1. **Task 1: Build _riscv.so + capture build log** — `761b970` (chore)
2. **Task 2: Run gtx suite + capture pre-existing bug surface** — `afc6e56` (test)
3. **Task 3: Add trace mnemonic regression guard** — `b81b000` (test)
4. **Task 4: Honest doc-sync after gap-closure attempt** — `dde17f9` (docs)

## Files Created/Modified

- `.planning/phases/02-skeleton-disasm/02-06-BUILD-LOG.md` — created. Source of truth for the closure-cycle evidence (build outputs, pytest counts, category analysis).
- `.planning/phases/02-skeleton-disasm/02-06-SUMMARY.md` — created. This file.
- `tests/gtx/test_skeleton.py` — added `test_full_trace_mnemonics_present` (Task 3). 105 -> ~165 lines.
- `.planning/phases/02-skeleton-disasm/02-HUMAN-UAT.md` — frontmatter `status: partial -> needs_followup`; 3 items result -> `blocked / partial / blocked` (NOT `passed`); summary reflects 0 passed / 3 blocked.
- `.planning/phases/02-skeleton-disasm/02-VERIFICATION.md` — top-level `status: human_needed -> needs_followup`; `re_verification` block populated; "Re-Verification" section appended.
- `.planning/ROADMAP.md` — 5 occurrences of plan-05 `[ ]` -> `[x]`; Phase 2 main `4/5 complete -> 5/5 complete`.

## Decisions Made

1. **Use `--no-build-isolation` flag** — critical to keep pybind11 at 3.0.1 (already installed) and avoid pip pulling latest 3.0.4 (which would reproduce the deferred-items.md csr_t issue).
2. **Lower trace-mnemonic threshold from >=3 to >=1** — Step 3.4 fallback. Spike `--log` dumps executed instructions only; committed ELF executes 1 RoCC instruction; richer-ELF fixture deferred to P3+.
3. **DO NOT flip UAT items to `passed`** — Plan 02-06 Step 4.1 directive: "only after Tasks 1-3 succeed". Tasks 2/3 surfaced regressions. Flipping would be dishonest and would mask real bugs from `/gsd:phase-evolve 2`.
4. **DO apply ROADMAP doc-lag fix** — independent of gap-closure outcome. Plan-05 IS genuinely complete (5 task commits, SUMMARY.md, VALIDATION.md approved). The unchecked checkboxes were stale; flipping is correct.
5. **Document, do NOT auto-fix Categories A-D** — these require modifying files outside `02-06-PLAN.md` `files_modified` (Wave 0/1/2 owned). Per the executor's scope-boundary rule and the plan's explicit scope discipline, these route to Phase-2 follow-up.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing setuptools_scm build dependency**
- **Found during:** Task 1 (initial pip install attempt)
- **Issue:** `pip install -e . --no-build-isolation` failed with `ModuleNotFoundError: No module named 'setuptools_scm'`. The flag skips fetching build deps, but the system Python lacked this declared build dep.
- **Fix:** `python3 -m pip install --user setuptools_scm` (10.0.5 + vcs-versioning 1.1.1)
- **Files modified:** None (system pip user-site)
- **Verification:** Re-ran `pip install -e .` — build succeeded; `_riscv.cpython-310-x86_64-linux-gnu.so` created.
- **Committed in:** Documented in BUILD-LOG.md (no code change)

**2. [Rule 3 - Blocking] Use python3 -m pip install --user (not bare pip install -e .)**
- **Found during:** Task 1 (first invocation)
- **Issue:** Bare `pip install -e .` failed with `error: No virtual environment found; run uv venv to create an environment, or pass --system to install into a non-virtual environment`. Default pip on this WSL machine is `uv`-controlled.
- **Fix:** Switched to `python3 -m pip install -e . --no-build-isolation --user` (forces system Python's pip, installs to user site-packages).
- **Files modified:** None
- **Verification:** Build succeeded.
- **Committed in:** Documented in BUILD-LOG.md (Task 1 commit context)

### Architectural (Rule 4) — STOPPED, documented for follow-up

**3. [Rule 4 - Architectural] 15 pre-existing test/production bugs surfaced**
- **Found during:** Task 2 (pytest run)
- **Issue:** Building `_riscv.so` exposed bugs the mock-fallback hid:
  - **Category A** (8 failures): `test_reset.py` — `super().reset(proc)` C++ strict-type rejects `MockProcessor` (line 74 of `npu.py`). Resolution: remove the no-op `super().reset(proc)` (extension_t::reset is `virtual void reset(processor_t &) {}` — empty default).
  - **Category B** (6 failures): `test_disasm.py` — `disasm_insn_t` C++ ctor normalizes `_` -> `.` in mnemonic names; tests assert `_`-form. Resolution: update test expectations to dot-form.
  - **Category C** (1 failure): `nop_wjoin.elf` LOAD segment at 0x7ffff000 (not 0x80000000). Resolution: replace `-Ttext=0x80000000` with `-Wl,-Ttext-segment=0x80000000` in `tests/gtx/data/elf/Makefile` and rebuild.
  - **Category D** (1+ failure): even with corrected ELF, sp not initialized via `XPR.write` and `custom1` doesn't dispatch (instruction traps as illegal). Resolution: investigate spike's reset/dispatch lifecycle integration with `@isa.register('gtx')` wrapped subclass.
- **Fix:** NOT auto-fixed. Categories A-D require modifying files outside plan 02-06's `files_modified` list (forbidden by scope discipline). Documented thoroughly in BUILD-LOG.md and routed to Phase-2 follow-up.
- **Files modified:** Only documentation (BUILD-LOG.md, VERIFICATION.md, HUMAN-UAT.md).
- **Verification:** Failures captured by `pytest tests/gtx/ -q -o "addopts="` showing `15 failed, 71 passed in 3.06s`.
- **Committed in:** afc6e56 (Task 2 evidence) + dde17f9 (doc-sync without UAT flip)

---

**Total deviations:** 2 auto-fixed (Rule 3 blocking) + 1 architectural-stop (Rule 4)

**Impact on plan:** Plan succeeded at Task 1 (build) and Task 4 (ROADMAP doc-lag). Tasks 2-3 surfaced unhideable pre-existing bugs that block UAT closure. The plan's strict `files_modified` discipline meant auto-fixing was forbidden; the executor honored that and documented the work for follow-up. No silent failure; no UAT items dishonestly flipped.

## Issues Encountered

1. **Build environment quirks (WSL2):**
   - WSL `pip` defaults to `uv`-controlled mode, requiring `python3 -m pip ... --user`.
   - `setuptools_scm` was a missing build dep despite being declared (resolved by user-install).
2. **Mock-fallback insufficiency:** D-17/D-18/D-19 mock discipline was supposed to make tests pass under both `_riscv.so`-built and `_riscv.so`-absent modes. In practice, 15 of 21 newly-running tests fail under `_riscv.so`-built mode. This is a genuine gap in the mock-equivalence claim, NOT a Phase-2 implementation bug per se — but the tests + production code need adjustments to converge.
3. **ELF linker behavior change:** GCC 15.2.0 `.riscv.attributes` placement causes `-Ttext` to position the LOAD segment one page below the requested address. This may not have been an issue with older toolchains.
4. **Production dispatch bug (Category D):** Even with a correctly-loaded ELF, sp initialization and custom1 dispatch are broken. This is a real bug that mock-only testing could never detect, validating the gap-closure intent (find the unknowns) even though the closure itself failed.

## Known Stubs

None introduced by this plan.

The plan only added `test_full_trace_mnemonics_present` (a regression guard, gated correctly) and updated documentation. No new stubs in production code.

## User Setup Required

None. Build environment was prepared automatically (Rule-3 deviations).

## Next Phase Readiness

**NOT READY for `/gsd:phase-evolve 2` without follow-up.**

Recommended next action — pick ONE of:

**Option A (preferred): Roll into `/gsd:phase-evolve 2` cleanup.** When evolve runs, it can prescribe a single follow-up plan that fixes Categories A-D in one pass:
1. Remove `super().reset(proc)` from `npu.py:74` (no-op upstream; line removal trivially fixes 8 reset tests).
2. Update `test_disasm.py` mnemonic expectations from `_`-form to `.`-form (matches real `disasm_insn_t` behavior; 6 tests fix).
3. Replace `-Ttext=0x80000000` with `-Wl,-Ttext-segment=0x80000000` in `tests/gtx/data/elf/Makefile` and rebuild ELF (1 test fix + unblocks Task 3 trace test).
4. Investigate Category D: sp init lifecycle (XPR.write timing) and custom1 dispatch under spike. Likely requires reading `src/main/cpp/riscv_extension.cc` reset() trampoline + `processor_t::reset()` to confirm extension reset() ordering.

**Option B: Create plan 02-07** — dedicated post-build-fix plan with explicit `files_modified` covering `npu.py` + `test_reset.py` + `test_disasm.py` + ELF Makefile.

After follow-up resolution, re-run `pytest tests/gtx/ -q -o "addopts="` (target: 87 passed / 0 failed / 0 skipped) and re-verify the 3 UAT items in `02-HUMAN-UAT.md`.

## Self-Check: PASSED

**Created files exist:**
- `.planning/phases/02-skeleton-disasm/02-06-BUILD-LOG.md` — FOUND
- `.planning/phases/02-skeleton-disasm/02-06-SUMMARY.md` — FOUND (this file)

**Commits exist:**
- `761b970` (Task 1) — FOUND
- `afc6e56` (Task 2) — FOUND
- `b81b000` (Task 3) — FOUND
- `dde17f9` (Task 4) — FOUND

**Verification commands (from plan):**
- `ls src/main/python/riscv/_riscv*.so` — _riscv.cpython-310-x86_64-linux-gnu.so present
- `from riscv import _riscv; from riscv.gtx import GtxNpu` — both resolve, GtxNpu is real class
- `pytest tests/gtx/ -q -o "addopts="` — 71 passed, 15 failed, 0 skipped (skips assertion satisfied; failures documented)
- `grep -cE "^- \[x\] 02-skeleton-disasm/02-05-PLAN.md" .planning/ROADMAP.md` — 5 (doc-lag fixed)
- `grep -E '^status:' .planning/phases/02-skeleton-disasm/02-VERIFICATION.md` — needs_followup (intentionally not 'passed')
- `grep -cE 'result: passed' .planning/phases/02-skeleton-disasm/02-HUMAN-UAT.md` — 0 (intentionally not 3)

The plan's automated `<verify>` block expectations of `result: passed = 3` and `status: passed` are intentionally NOT met. Flipping them would lie about system state. The doc-lag fix (`[x] = 5`) IS met as planned.

---
*Phase: 02-skeleton-disasm*
*Plan: 06 (gap-closure cycle, Wave 3)*
*Completed: 2026-05-04*
