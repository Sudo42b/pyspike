---
phase: 01-foundation
plan: 01
subsystem: infra
tags: [python, package-skeleton, numpy-fp16, byte-order-guard, rocc, gtx-npu]

# Dependency graph
requires:
  - phase: 01-foundation (research/context)
    provides: D-13/D-14 module layout lock, D-09 LE host assumption, gtx_params.h reference values
provides:
  - "riscv.gtx import path (5 sibling submodule re-exports)"
  - "Little-endian host tripwire (RuntimeError on non-LE)"
  - "HW topology + memory size + SPR base constants (gtx_params.h verbatim port)"
  - "Phase 1 funct7 stub constants (WRSPR/RDSPR/WSPLIT/WJOIN/DISPATCH_*)"
  - "ops/ subpackage marker (P2-P5 fills mm/vec/act/dma)"
  - "tests/gtx/ pytest collection root"
affects:
  - 01-foundation/02-fp (consumes riscv.gtx import path; lands fp.py)
  - 01-foundation/03-memory (consumes params.GTX_NEST_NUM/SPU/L1; lands memory.py + ddr.py)
  - 01-foundation/04-packaging (consumes riscv.gtx package presence for setuptools discovery)
  - 02-rocc-dispatch (consumes encoding.GTX_F7_*; adds full ISS-full disasm table)
  - 03-dma (consumes params.GTX_DDR_DEFAULT_SIZE_BYTES + GTX_DDR_BUS_WORD_BYTES)
  - 04-mm / 05-vec-act (consumes params.GTX_NEST_NUM x GTX_SPU_NUM topology)

# Tech tracking
tech-stack:
  added: []  # Pure Python skeleton; NumPy/setup unchanged in this plan
  patterns:
    - "Module-load tripwire (sys.byteorder check at __init__.py top)"
    - "C++ macro -> Python constant ports keep the C++ identifier verbatim (CONTEXT Claude's Discretion #2)"
    - "Phase-staged constants: P1 funct7 stubs, P2 fills full disasm table"
    - "Sibling-module re-export pattern (`from . import fp`) -- Wave 1 plans land bodies"

key-files:
  created:
    - src/main/python/riscv/gtx/__init__.py
    - src/main/python/riscv/gtx/params.py
    - src/main/python/riscv/gtx/encoding.py
    - src/main/python/riscv/gtx/ops/__init__.py
    - tests/gtx/__init__.py
  modified: []  # riscv/__init__.py intentionally untouched (open-question 3)

key-decisions:
  - "Removed literal token 'GtxNpu' from __init__.py docstring (acceptance criterion: grep exit 1) -- intent preserved by paraphrasing as 'ROCC subclass'"
  - "Followed plan exactly: __init__.py re-exports fp/memory/ddr eagerly; sibling Wave 1 plans land those modules. Standalone import will fail until Wave 1 closes (documented in plan <verification>)"
  - "Verified params.py / encoding.py constants via importlib.util.spec_from_file_location to bypass partially-initialized __init__.py"

patterns-established:
  - "LE-host tripwire: sys.byteorder != 'little' raises RuntimeError before any NumPy import (defends np.float16 view semantics on theoretical non-x86_64 host)"
  - "Phase-staged encoding constants: encoding.py owns funct7 in P1; disasm.py owns full ISS-full table in P2 (D-13)"
  - "C++ macro name preservation: GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES match gtx_params.h verbatim for grep-friendly cross-reference during P4/P5 op porting"

requirements-completed: [FOUND-03]

# Metrics
duration: 4min
completed: 2026-05-04
---

# Phase 1 Plan 1: Skeleton Summary

**riscv.gtx package skeleton with LE byte-order tripwire, gtx_params.h constants port, RoCC funct7 stubs, and tests/gtx pytest root**

## Performance

- **Duration:** 4 min (3m 52s)
- **Started:** 2026-05-04T05:36:42Z
- **Completed:** 2026-05-04T05:40:41Z
- **Tasks:** 3
- **Files modified:** 5 (all created)

## Accomplishments
- `riscv.gtx` package entrypoint with LE host guard (raises RuntimeError on big-endian)
- HW topology + memory + SPR base constants ported from `gtx_params.h` (NEST=4, SPU=16, L1=384KB, L2=16MB, DDR=4GiB default, GSPR/NSPR/LSPR @ 0x000/0x400/0x800)
- Phase 1 funct7 stubs (8 constants: WRSPR, RDSPR, WSPLIT, WJOIN, 4 DISPATCH variants)
- `riscv.gtx.ops` package marker (P2-P5 fills handlers)
- `tests/gtx/` pytest collection root with Apache header
- `riscv/__init__.py` left untouched (verified empty diff) -- preserves no-NumPy-on-import-riscv invariant

## Task Commits

Each task was committed atomically:

1. **Task 01-01: riscv.gtx package entrypoint + LE guard** - `d55a82a` (feat)
2. **Task 01-02: HW params, funct7 constants, ops marker** - `30a50d6` (feat)
3. **Task 01-03: tests/gtx package marker** - `7284080` (chore)

_Plan metadata commit added after summary._

## Files Created/Modified

- `src/main/python/riscv/gtx/__init__.py` (39 lines) - Package entry; LE tripwire; re-exports `encoding/fp/memory/params/ddr`
- `src/main/python/riscv/gtx/params.py` (43 lines) - HW topology (4 NEST x 16 SPU), memory sizes (L0=1KB, L1=384KB, L2=16MB), DDR (4GiB default, 32B bus word), SPR address ranges (GSPR=0x000-0x3FF, NSPR=0x400-0x7FF, LSPR=0x800-0xBFF)
- `src/main/python/riscv/gtx/encoding.py` (36 lines) - funct7 stubs: `GTX_F7_WRSPR=0x00`, `GTX_F7_RDSPR=0x01`, `GTX_F7_WSPLIT=0x02`, `GTX_F7_WJOIN=0x03`, `GTX_F7_DISPATCH_MM=0x04`, `GTX_F7_DISPATCH_VEC=0x05`, `GTX_F7_DISPATCH_ACT=0x06`, `GTX_F7_DISPATCH_DMA=0x07`. ISS-full per-op funct7 deferred to disasm.py (P2)
- `src/main/python/riscv/gtx/ops/__init__.py` (16 lines) - Empty package marker
- `tests/gtx/__init__.py` (15 lines) - Apache header only; pytest auto-discovers via `testpaths=["tests"]`

### Constants defined (params.py)

| Name | Value | Purpose |
| --- | --- | --- |
| `GTX_NEST_NUM` | 4 | NEST count |
| `GTX_SPU_NUM` | 16 | SPUs per NEST |
| `GTX_SPUS_PER_NEST` | 16 | Alias of `GTX_SPU_NUM` |
| `GTX_L0_SIZE_BYTES` | 1024 | 1 KB per SPU |
| `GTX_L1_SIZE_BYTES` | 393216 (384*1024) | 384 KB per SPU |
| `GTX_L2_SIZE_BYTES` | 16777216 (16*1024*1024) | 16 MB per NEST |
| `GTX_DDR_DEFAULT_SIZE_BYTES` | 4294967296 (4*1024**3) | 4 GiB default cap |
| `GTX_DDR_BUS_WORD_BYTES` | 32 | Bus word for `GTX_DDR_REVERSED` reversal |
| `GSPR_BASE` / `GSPR_END` | 0x000 / 0x3FF | Global SPR range |
| `NSPR_BASE` / `NSPR_END` | 0x400 / 0x7FF | NEST-scoped SPR range |
| `LSPR_BASE` / `LSPR_END` | 0x800 / 0xBFF | SPU-scoped (local) SPR range |

### Constants defined (encoding.py)

| Name | Value | Purpose |
| --- | --- | --- |
| `GTX_F7_WRSPR` | 0x00 | WRSPR (gem5 simplified) / MM ISS-full (rs1!=0 disambiguation in P4) |
| `GTX_F7_RDSPR` | 0x01 | RDSPR |
| `GTX_F7_WSPLIT` | 0x02 | custom1 warp split |
| `GTX_F7_WJOIN` | 0x03 | custom1 warp join (exit semantics in P2) |
| `GTX_F7_DISPATCH_MM` | 0x04 | Dispatch MM op |
| `GTX_F7_DISPATCH_VEC` | 0x05 | Dispatch VEC op |
| `GTX_F7_DISPATCH_ACT` | 0x06 | Dispatch ACT op |
| `GTX_F7_DISPATCH_DMA` | 0x07 | Dispatch DMA op |

## Decisions Made

- **Docstring rewrite to satisfy strict `grep -q 'GtxNpu'` exit-1 acceptance:** The plan's `<action>` block included a docstring with the literal token `GtxNpu`, but the `<acceptance_criteria>` required `grep` exit 1 (token absent) per D-14 ("P1에서 GtxNpu 노출 금지"). Resolved by paraphrasing the docstring to "The ROCC subclass is added in Phase 2 (D-14)" -- intent preserved, criterion satisfied.
- **Verification methodology for params/encoding constants:** The plan's `<automated>` test in Task 01-02 imports through `riscv.gtx.params`, which transitively executes `riscv.gtx.__init__.py` -- and that file eagerly imports `fp/memory/ddr` (Wave 1 sibling plans, not yet landed in this worktree). To verify the constants without circular dependency on Wave 1 sibling plans, used `importlib.util.spec_from_file_location` to load each module file in isolation. All 17 constants verified with correct values. The plan-level `<verification>` (line 331) explicitly documents this Wave 1 integration boundary.
- **No modifications to `riscv/__init__.py`:** `git diff src/main/python/riscv/__init__.py` confirmed empty. This preserves the invariant that `import riscv` (existing pyspike use cases) does not pull NumPy.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed literal `GtxNpu` token from __init__.py docstring**
- **Found during:** Task 01-01 (verifying acceptance criteria)
- **Issue:** Plan `<action>` template included docstring text "`GtxNpu` (the ROCC subclass) is added in Phase 2 (D-14)." The `<acceptance_criteria>` line in the same task required `grep -q 'GtxNpu' src/main/python/riscv/gtx/__init__.py` to exit with code 1 (token absent), enforcing D-14 ("P1에서 GtxNpu 노출 금지" -- no GtxNpu mention in P1 entry point).
- **Fix:** Paraphrased the docstring sentence from "`GtxNpu` (the ROCC subclass) is added in Phase 2 (D-14)." to "The ROCC subclass is added in Phase 2 (D-14)." -- preserves intent (deferral to P2) without the literal token.
- **Files modified:** `src/main/python/riscv/gtx/__init__.py`
- **Verification:** `grep -q 'GtxNpu' src/main/python/riscv/gtx/__init__.py; echo $?` -> 1 (PASS, token absent)
- **Committed in:** `d55a82a` (Task 01-01 commit, fix applied before commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix to satisfy acceptance criterion)
**Impact on plan:** Cosmetic docstring paraphrase. No semantic change. No scope creep.

## Issues Encountered

- **Worktree base out-of-date with main:** Worktree branch `worktree-agent-ae494e55f66a06db0` was based at commit `52dbcfc` (pre-planning), so `.planning/` directory was missing. Resolved by `git rebase main` to bring worktree branch up to current main, then copying over uncommitted PLAN files and modified STATE.md/ROADMAP.md/config.json/CLAUDE.md from the parent project tree (these were untracked in main but present in `/mnt/e/14_NIGHTLY/pyspike/`). Did not commit those copies in this plan -- they will be picked up by the final metadata commit if relevant.
- **`from . import fp` (etc.) cannot succeed standalone in this worktree:** Plan's `<action>` mandated eager re-exports of `fp/memory/ddr` from sibling Wave 1 plans (02-fp / 03-memory) that have not yet been merged into this worktree. The plan's `<verification>` (line 331) explicitly accepts this -- "Wave 1 종료 시점에 통합 검증". Worked around for constant verification via `importlib.util.spec_from_file_location` (no `riscv.gtx.__init__.py` execution).

## User Setup Required

None - no external service configuration or environment variables introduced.

## Next Phase Readiness

- **Ready for Wave 1 sibling plans (02-fp, 03-memory):** They consume `from riscv.gtx.params import …` (works in isolation) and add `fp.py`, `memory.py`, `ddr.py` next to this plan's outputs.
- **Ready for Wave 2 packaging (04-packaging):** `riscv.gtx` package presence verified; `[tool.setuptools.packages.find].include = ["riscv*"]` (or equivalent) will auto-discover this package.
- **Blocker (resolves at Wave 1 close):** `python -c "import riscv.gtx"` will raise `ImportError: cannot import name 'fp' from partially initialized module 'riscv.gtx'` until Wave 1 sibling plans land. This is by design (plan `<verification>` line 331).
- **Phase 2 readiness:** Funct7 stubs in `encoding.py` provide the exact constants P2 needs for RoCC dispatch. The `# (...remaining 70+ constants in Phase 2)` comment in `encoding.py` flags the disasm.py expansion site for P2.

## Self-Check: PASSED

- File `src/main/python/riscv/gtx/__init__.py` exists: FOUND
- File `src/main/python/riscv/gtx/params.py` exists: FOUND
- File `src/main/python/riscv/gtx/encoding.py` exists: FOUND
- File `src/main/python/riscv/gtx/ops/__init__.py` exists: FOUND
- File `tests/gtx/__init__.py` exists: FOUND
- Commit `d55a82a` (Task 01-01): FOUND in `git log --oneline`
- Commit `30a50d6` (Task 01-02): FOUND in `git log --oneline`
- Commit `7284080` (Task 01-03): FOUND in `git log --oneline`
- `riscv/__init__.py` unchanged: VERIFIED (empty diff)
- All 17 params constants verified via isolated importlib.util load: PASS
- All 8 encoding funct7 constants verified: PASS

---
*Phase: 01-foundation*
*Completed: 2026-05-04*
