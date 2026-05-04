---
phase: 02-skeleton-disasm
plan: 01
subsystem: infra
tags: [pybind11, numpy, rocc, riscv, gtx, dispatch, disasm, dataclass, decorator-registry]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: GtxMemory class, FP16 helpers, params constants, encoding stub, ddr lazy alloc
provides:
  - "riscv.gtx.npu.GtxNpu(isa.ROCC) class skeleton (registered as @isa.register('gtx'))"
  - "riscv.gtx._registry decorator-based per-op registry primitive (handler/collect_for_kind/collect_disasms)"
  - "riscv.gtx.dispatch.build_custom0_table / build_custom1_table builders"
  - "riscv.gtx.warp_state.WarpState dataclass (is_ploop/is_tloop/is_sloop/tmu_id/curr_id)"
  - "riscv.gtx.encoding full P2 funct7+funct3+opcode constant set"
  - "riscv.gtx.ops.spr / riscv.gtx.ops.control stub modules (plans 02/03 fill)"
  - "tests/gtx/_mocks.py (MockProcessor/MockState/MockXPR/MockInsn) for unit tests w/o _riscv.so"
  - "tests/gtx/conftest.py hybrid fallback fixtures (proc, insn_factory, riscv_available)"
  - "tests/conftest.py D-18 try/except guard around riscv.cfg/debug_module/sim imports"
  - "tests/gtx/data/elf/nop_wjoin.elf prebuilt RISC-V ELF fixture (entry 0x80000000)"
affects: [02-02-spr, 02-03-warp-control, 02-04-disasm, 02-05-integration, phase-03-dma, phase-04-mm, phase-05-vec-act]

# Tech tracking
tech-stack:
  added: []  # No new external dependencies
  patterns:
    - "Per-op decorator registry (D-13): @handler(kind=..., funct7=..., mnemonic=...) at module-load time"
    - "Closure-binding dispatch tables: handlers wrapped in lambda capturing the GtxNpu instance"
    - "Hybrid mock fallback (D-17): same test code runs with or without _riscv.so via try/except in conftest"
    - "Try/except guard on npu import (graceful degradation when _riscv absent — Phase 1 tests still pass)"
    - "Layered SPR storage (D-11): gspr (flat dict), nspr (list of dict per NEST), lspr ([NEST][SPU] dicts)"

key-files:
  created:
    - "src/main/python/riscv/gtx/npu.py — GtxNpu(isa.ROCC) shell + dispatch + reset"
    - "src/main/python/riscv/gtx/_registry.py — handler/collect_for_kind/collect_disasms"
    - "src/main/python/riscv/gtx/dispatch.py — build_custom0_table / build_custom1_table"
    - "src/main/python/riscv/gtx/warp_state.py — WarpState dataclass"
    - "src/main/python/riscv/gtx/ops/spr.py — stub (plan 02 fills)"
    - "src/main/python/riscv/gtx/ops/control.py — stub (plan 03 fills)"
    - "tests/gtx/_mocks.py — MockProcessor/MockState/MockXPR/MockInsn"
    - "tests/gtx/conftest.py — hybrid fixtures (proc/insn_factory/riscv_available)"
    - "tests/gtx/data/elf/nop_wjoin.S — assembly source (D-22)"
    - "tests/gtx/data/elf/Makefile — reproducible build recipe"
    - "tests/gtx/data/elf/nop_wjoin.elf — prebuilt 5KB ELF binary"
    - "tests/gtx/data/elf/.gitignore — local negation rules unblocking *.elf and Makefile"
  modified:
    - "src/main/python/riscv/gtx/__init__.py — added try/except npu import + GtxNpu re-export"
    - "src/main/python/riscv/gtx/encoding.py — full P2 constant set (replaces 8-stub from Phase 1)"
    - "src/main/python/riscv/gtx/ops/__init__.py — imports spr+control submodules to fire decorators"
    - "tests/conftest.py — D-18 guard around riscv.cfg/debug_module/sim imports + mock_sim skip"

key-decisions:
  - "mxe_accum is 2D (GTX_NEST_NUM, GTX_SPU_NUM) float32 verbatim per gtx_npu.h:1254 — supersedes CONTEXT.md D-06 which stated 4D"
  - "GtxNpu npu import wrapped in try/except so riscv.gtx package still loads when _riscv.so is absent (graceful degradation; GtxNpu = None in that mode)"
  - "Makefile uses CC = (not ?=) to override Make's implicit cc default; otherwise build picks up host gcc and assembly fails"
  - "tests/gtx/data/elf/.gitignore uses negation rules (!nop_wjoin.elf, !Makefile) to override project-level *.elf and Makefile patterns"

patterns-established:
  - "Per-op decorator registry: each future op module (P3 dma.py, P4 mm.py, etc.) uses @handler(kind=..., funct7=..., mnemonic=...) — modules just need to be imported once at __init__ time"
  - "Closure-binding dispatch: dispatch.py wraps each registered fn in a lambda that captures the GtxNpu instance, so handler signatures stay clean (npu, proc, insn, xs1, xs2)"
  - "Hybrid mock fallback in tests/gtx/conftest.py + tests/conftest.py D-18 guard: enables pytest collection without _riscv.so, plans 02-04 use mocks via the proc/insn_factory fixtures"
  - "Stub-then-fill convention: ops/spr.py + ops/control.py created with documentation-only comments listing expected @handler registrations — plans 02/03 fill them"

requirements-completed: [CORE-01, CORE-02]

# Metrics
duration: 7m44s
completed: 2026-05-04
---

# Phase 02 Plan 01: Skeleton + Disasm Scaffold Summary

**Wave 0 scaffold landing the riscv.gtx package extension layer (GtxNpu shell + per-op decorator registry + dispatch builders + WarpState + test mocks + nop_wjoin.elf fixture) so plans 02/03/04 can land op handlers in parallel.**

## Performance

- **Duration:** 7m44s
- **Started:** 2026-05-04T08:38:26Z
- **Completed:** 2026-05-04T08:46:10Z
- **Tasks:** 3 (all auto, all TDD-flavored — though "test infrastructure" was the test itself in T1)
- **Files modified:** 16 (4 modified, 12 created)

## Accomplishments

- **`riscv.gtx.GtxNpu`** importable as a `riscv.isa.ROCC` subclass, registered via `@isa.register("gtx")` (CORE-01). When `_riscv.so` is absent, `GtxNpu = None` (graceful degradation; package import never fails).
- **`reset()`** sets `proc.get_state().XPR.write(2, 0x80100000)` (CORE-02), zero-fills `mxe_accum`/L0/L1/L2, reseeds GSPR/NSPR/LSPR defaults per `gtx_npu_core.cc:80-109` verbatim, and calls `warp.reset()`.
- **Per-op decorator registry** (`_registry.py`) + **dispatch builders** (`dispatch.py`) provide a 1-line API for plans 02/03 to register custom0/custom1 handlers — `@handler(kind='custom0', funct7=0x49, mnemonic='wrspr')`.
- **`WarpState` dataclass** with `is_ploop/is_tloop/is_sloop/tmu_id/curr_id` fields + `reset()` ready for plan 03 warp ops.
- **Hybrid mock infrastructure** (`tests/gtx/_mocks.py` + `tests/gtx/conftest.py` + `tests/conftest.py` D-18 guard) lets pytest collect `tests/gtx/` cleanly without `_riscv.so` (13 Phase 1 tests still green).
- **`nop_wjoin.elf` prebuilt fixture** (5KB RISC-V ELF, entry 0x80000000) committed alongside its assembly source + Makefile so plan 05's integration test resolves immediately.

## Task Commits

Each task was committed atomically:

1. **Task 1: Test infrastructure — mocks + hybrid conftest + D-18 guard** — `2170e6d` (test)
2. **Task 2: Package skeleton — encoding/WarpState/_registry/dispatch/npu/ops stubs** — `cd7c042` (feat)
3. **Task 3: nop_wjoin.elf fixture — assembly + Makefile + prebuilt binary** — `01e9737` (chore)

**Plan metadata commit:** to follow (this SUMMARY + STATE/ROADMAP updates).

## Files Created/Modified

### Created (12)
- `src/main/python/riscv/gtx/npu.py` — `GtxNpu(isa.ROCC)` class shell with custom0/custom1 dispatch, reset(), get_disasms/get_csrs/get_instructions overrides
- `src/main/python/riscv/gtx/_registry.py` — `_HANDLER_REGISTRY` list + `handler` decorator + `collect_for_kind` + `collect_disasms` stub (plan 04 fills)
- `src/main/python/riscv/gtx/dispatch.py` — `build_custom0_table` / `build_custom1_table` closure-binding builders
- `src/main/python/riscv/gtx/warp_state.py` — `WarpState` dataclass
- `src/main/python/riscv/gtx/ops/spr.py` — stub module (plan 02 fills with WRSPR/RDSPR handlers)
- `src/main/python/riscv/gtx/ops/control.py` — stub module (plan 03 fills with warp/control handlers)
- `tests/gtx/_mocks.py` — `MockXPR/MockState/MockProcessor/MockInsn` dataclasses
- `tests/gtx/conftest.py` — hybrid fallback fixtures (`proc`, `insn_factory`, `riscv_available`)
- `tests/gtx/data/elf/nop_wjoin.S` — RISC-V assembly: `addi sp,sp,-16; .insn r 0x2b,0b101,...; j .`
- `tests/gtx/data/elf/Makefile` — `riscv64-unknown-elf-gcc -nostdlib -nostartfiles -static -Ttext=0x80000000`
- `tests/gtx/data/elf/nop_wjoin.elf` — 5KB prebuilt ELF (entry 0x80000000, three insns: addi/join/j)
- `tests/gtx/data/elf/.gitignore` — local negation rules unblocking `*.elf` and `Makefile`

### Modified (4)
- `src/main/python/riscv/gtx/__init__.py` — added try/except guarded `from . import npu` + GtxNpu re-export; `__all__` extended
- `src/main/python/riscv/gtx/encoding.py` — replaced 8-stub Phase 1 set with full P2 constants: 8 gem5 funct7 + 5 ISS funct7 (0x48/0x49/0x4A/0x7D/0x7E) + 8 warp custom1 funct3 + 4 mode constants + 6 GSPR loop addresses + 2 RoCC opcodes
- `src/main/python/riscv/gtx/ops/__init__.py` — imports `spr` and `control` submodules so their `@handler` decorators fire at package load (Pitfall 6)
- `tests/conftest.py` — wrapped `riscv.cfg/debug_module/sim` imports in `try/except ImportError` with `_RISCV_AVAILABLE` flag (D-18); `mock_sim` fixture now skips when unavailable

## Decisions Made

1. **`mxe_accum` 2D shape correction** — Locked to `(GTX_NEST_NUM, GTX_SPU_NUM)` float32 verbatim per `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1254`. The plan instructed (and the planner explicitly flagged) that CONTEXT.md D-06 (which stated 4D `(NEST, SPU, M_TILE, N_TILE)`) was incorrect. Implementation follows C++ ground truth.
2. **`GtxNpu` graceful absence** — When `_riscv.so` is not built, `riscv.gtx.GtxNpu` evaluates to `None` (instead of raising at package import time). This is required so Phase 1 tests (`test_fp_roundtrip`, `test_memory_layout`) keep passing even though they import `riscv.gtx` transitively. Plans 02-05 unit tests use mocks; plan 05 integration test gates on `_RISCV_AVAILABLE`.
3. **Makefile `CC = …` not `?=`** — Make sets `CC=cc` implicitly, so `?=` is a no-op. Switched to `CC = /opt/riscv/bin/riscv64-unknown-elf-gcc` so a clean `make` picks up the cross toolchain. To use a different toolchain, callers run `make CC=/path/to/gcc`.
4. **`tests/gtx/data/elf/.gitignore` negations** — Project root `.gitignore` excludes `*.elf` and `Makefile`. Per D-22 these MUST be committed. Resolved by adding `!nop_wjoin.elf` and `!Makefile` negations in the local `.gitignore`. (`git check-ignore -v` confirms the negation rule wins.)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Makefile `CC ?= …` did not override Make's implicit `cc` default**
- **Found during:** Task 3 (`make` invocation)
- **Issue:** First `make` ran `cc -nostdlib -nostartfiles -static -Ttext=0x80000000 -o nop_wjoin.elf nop_wjoin.S`, which used the host GCC and failed with "no such instruction: addi sp,sp,-16" (host arch is x86_64).
- **Fix:** Changed `CC ?= /opt/riscv/bin/riscv64-unknown-elf-gcc` to `CC = /opt/riscv/bin/riscv64-unknown-elf-gcc`. Added a comment documenting that callers can override via `make CC=/path/to/gcc`.
- **Files modified:** `tests/gtx/data/elf/Makefile`
- **Verification:** Re-ran `make clean nop_wjoin.elf` — succeeded. `file nop_wjoin.elf` reports `ELF 64-bit LSB executable, UCB RISC-V`. `objdump -d` shows three expected instructions.
- **Committed in:** `01e9737` (Task 3 commit)

**2. [Rule 3 - Blocking] Project `.gitignore` excluded `*.elf` and `Makefile`**
- **Found during:** Task 3 (`git add tests/gtx/data/elf/Makefile tests/gtx/data/elf/nop_wjoin.elf`)
- **Issue:** `git check-ignore -v` showed `.gitignore:38:Makefile` and `.gitignore:3:*.elf` matching. D-22 mandates committing both files.
- **Fix:** Added local `.gitignore` in `tests/gtx/data/elf/` with negation rules `!nop_wjoin.elf` and `!Makefile` plus existing `*.o` / `*.tmp` blocks for build byproducts. Verified `git check-ignore -v` now reports the negation rules as the matching rules (i.e., files are no longer ignored).
- **Files modified:** `tests/gtx/data/elf/.gitignore`
- **Verification:** `git add` succeeded after the negation rules; `git status --short` shows all 4 files staged.
- **Committed in:** `01e9737` (Task 3 commit, alongside the fixture itself)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking issues that prevented Task 3 completion).
**Impact on plan:** Both fixes are mechanical/configuration adjustments. Neither changes the plan's behavioral contract — fixture content is identical to spec, just the build/commit recipe needed adjustment. No scope creep.

## Issues Encountered

- **`pytest tests/gtx/ --collect-only -q`** without `-o "addopts="` fails with `"unrecognized arguments: --pylint --mypy"` because `pyproject.toml` `[tool.pytest.ini_options].addopts` includes `--pylint --mypy --cov-report=lcov` and those pytest plugins are not currently installed in this dev environment. Workaround documented in VALIDATION.md is `--noconftest -o "addopts="` for fast local iteration. This is pre-existing (Phase 1 tests use the same workaround) and not specific to this plan.
- The first `make` run attempted to use host `cc` instead of the cross-toolchain. Resolved as documented above (deviation #1).

## `_riscv.so` and Toolchain Availability

- **`_riscv.so` was NOT available during this run.** All verification commands relied on the hybrid mock-fallback path: `riscv.gtx.GtxNpu` resolves to `None`, `riscv.gtx._registry`/`dispatch`/`encoding`/`warp_state` are pure-Python and importable directly. Phase 1 tests (`test_fp_roundtrip`, `test_memory_layout`) all 13 still pass against the mock-fallback collection (`pytest tests/gtx/ -x -q --noconftest -o "addopts="` → 13 passed in 0.42s). This matters for plan 05 integration test (which gates on `_RISCV_AVAILABLE`).
- **`/opt/riscv/bin/riscv64-unknown-elf-gcc` (15.2.0) WAS available.** `nop_wjoin.elf` was successfully built and committed. Plan 05 integration test fixture is ready immediately — no CI-time build deferral needed.

## Mxe_accum Shape Confirmation

The plan's `<interfaces>` block explicitly corrected CONTEXT.md D-06 from 4D to 2D. The implementation in `src/main/python/riscv/gtx/npu.py` declares:

```python
self._mxe_accum: np.ndarray = np.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM), dtype=np.float32
)
```

This is `(4, 16)` float32, matching `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1254` (`float mxe_accum[GTX_NUM_NESTS][GTX_SPUS_PER_NEST]`). The 4D shape from CONTEXT.md D-06 is OBSOLETE — future phases (P4 MM op) should reference this summary, not D-06.

## Next Phase Readiness

- **Plans 02 (SPR), 03 (warp/control), 04 (disasm) all unblocked.** They can land in parallel (Wave 1) using:
  - `from .._registry import handler` for op registration
  - `from ..warp_state import WarpState` for warp loop state
  - `from ..encoding import (GTX_F7_*, WARP_F3_*, GSPR_*, ...)` for constants
  - `from tests.gtx._mocks import MockProcessor, MockInsn` for unit tests
- **Plan 05 (integration)** has its `nop_wjoin.elf` fixture ready. Integration test still gates on `_RISCV_AVAILABLE` since pyspike CLI requires `_riscv.so` to actually run the firmware.
- **No blockers** for Wave 1.

## Self-Check: PASSED

Verified files exist:
- `src/main/python/riscv/gtx/npu.py` — FOUND
- `src/main/python/riscv/gtx/_registry.py` — FOUND
- `src/main/python/riscv/gtx/dispatch.py` — FOUND
- `src/main/python/riscv/gtx/warp_state.py` — FOUND
- `src/main/python/riscv/gtx/ops/spr.py` — FOUND
- `src/main/python/riscv/gtx/ops/control.py` — FOUND
- `tests/gtx/_mocks.py` — FOUND
- `tests/gtx/conftest.py` — FOUND
- `tests/gtx/data/elf/nop_wjoin.elf` — FOUND (ELF 64-bit LSB UCB RISC-V)
- `tests/gtx/data/elf/nop_wjoin.S` — FOUND
- `tests/gtx/data/elf/Makefile` — FOUND

Verified commits exist (`git log --oneline | grep`):
- `2170e6d` test(02-01): add mock infrastructure + D-18 hybrid fallback for tests/gtx/ — FOUND
- `cd7c042` feat(02-01): add riscv.gtx package skeleton (GtxNpu shell + per-op registry) — FOUND
- `01e9737` chore(02-01): add nop_wjoin.elf test fixture (assembly + Makefile + binary) — FOUND

Verified acceptance commands all pass:
- `python3 -c "from riscv.gtx import encoding; assert encoding.GTX_F7_WJOIN == 0x03"` exits 0
- `python3 -c "import riscv.gtx; assert hasattr(riscv.gtx, 'GtxNpu')"` exits 0
- `python3 -c "from tests.gtx._mocks import MockProcessor; ..."` exits 0
- `pytest tests/gtx/ --collect-only -o "addopts="` collects 13 tests, no errors
- `pytest tests/gtx/ -x -q --noconftest -o "addopts="` → 13 passed in 0.42s

---
*Phase: 02-skeleton-disasm*
*Completed: 2026-05-04*
