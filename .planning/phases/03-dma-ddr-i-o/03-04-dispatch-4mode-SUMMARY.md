---
phase: 03-dma-ddr-i-o
plan: 04
subsystem: dispatch
tags: [dispatch, warp-state, dma, rocc, gtx, routing]

# Dependency graph
requires:
  - phase: 03-dma-ddr-i-o
    provides: dma_engine.exec_dma_2d (Plan 01) — Mode 3 DMA call target
  - phase: 02-skeleton-disasm
    provides: WarpState (P/S/T loop flags, tmu_id, curr_id) — Mode selector
provides:
  - dispatch_4mode (npu, *, opcode, op1, op2, op3, sub_op=0) — 4-mode warp router
  - dispatch_iss_opcode (npu, nest_id, spu_id, funct7, op1, op2, op3) — DMA-only stub
  - Public re-export so `from riscv.gtx.dispatch import dispatch_4mode` resolves
affects: [04-mm, 05-vec-act, 03-05-flush-roundtrip]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sibling-module pattern (dispatch_4mode.py separate from dispatch.py) to avoid Wave 2 file-write conflict between two parallel plans modifying overlapping public surface"
    - "Re-export-only edit pattern: dispatch.py keeps its body owned by Plan 02; Plan 04 only appends an import line"

key-files:
  created:
    - src/main/python/riscv/gtx/dispatch_4mode.py (120 LOC, NEW)
  modified:
    - src/main/python/riscv/gtx/dispatch.py (+8 lines: re-export block; cooperatively integrated by Plan 02 commit 38aac36)
    - tests/gtx/test_dispatch_4mode.py (+251 lines: 13 tests)

key-decisions:
  - "Splitting dispatch_4mode and dispatch_iss_opcode into their own module dispatch_4mode.py (NOT dispatch.py) per CONTEXT 'Defer to user follow-up' — eliminates Wave 2 file-write race with Plan 02's 2-level builder upgrade"
  - "Re-export line added to dispatch.py only ONCE; the sibling Plan 02 agent included the same line in its 38aac36 commit (cooperative merge — both plans agreed on the canonical form), so no duplication"
  - "dispatch_iss_opcode is a TRUE stub in P3: every funct7 NOPs and returns 0. Plan 05 fills credit_st_chk (0x53) with the deferred-store flush trigger; P4 fills GTX_OP_MM=0; P5 fills VEC/ACT. The insertion point is comment-marked in the body."
  - "Mode 3 OR-rule (Pitfall 8): is_load = (sub_op == 0) OR (opcode == GTX_OP_DMA). Three tests cover the truth-table corners: (sub_op=0, opcode=VECTOR)=load, (sub_op=1, opcode=DMA)=load, (sub_op=1, opcode=VECTOR)=store."
  - "Width/height extraction (Pitfall 8 follow-up): width = op3 & 0xFFFF, height = (op3 >> 16) & 0xFFFF — explicit assertions in two tests so the encoding can never silently regress."

patterns-established:
  - "Sibling-module split for parallel-wave conflict avoidance: when two Wave-N plans both want to add to file F, one keeps F and the other adds module G + a single re-export line in F."
  - "P3 stub-with-Plan-N-marker pattern: dispatch_iss_opcode body is empty in P3 but the comment block names exactly which line Plan 05 will replace and what will replace it."
  - "Pitfall 8 dual-encoding tests: assert BOTH the boolean derivation AND the bitfield extraction; one check alone leaves room for silent regression."

requirements-completed: [DISP-03]

# Metrics
duration: 6m 29s
completed: 2026-05-05
---

# Phase 3 Plan 04: dispatch-4mode Summary

**4-mode warp router (Mode 1 broadcast 64 / Mode 2 broadcast 16 in tmu_id / Mode 3 single-NEST DMA via dma_engine.exec_dma_2d / Mode 4 single (tmu_id, curr_id)) plus DMA-only dispatch_iss_opcode stub, both living in their own module to avoid the Wave 2 conflict with Plan 02's dispatch.py upgrade.**

## Performance

- **Duration:** 6m 29s
- **Started:** 2026-05-05T14:33:01Z
- **Completed:** 2026-05-05T14:39:30Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files created:** 1 (dispatch_4mode.py)
- **Files modified:** 1 (dispatch.py — cooperatively integrated by Plan 02)
- **Tests added:** 13

## Accomplishments

- New module `riscv/gtx/dispatch_4mode.py` (120 LOC) — direct port of `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:25-200` minus thread-pool branches.
- All four warp-loop modes route correctly:
  - **Mode 1** (`!is_ploop`) — broadcasts 4 NEST × 16 SPU = 64 `dispatch_iss_opcode` calls.
  - **Mode 2** (`is_ploop && !is_sloop && !is_tloop`) — 16 SPU calls within `tmu_id`.
  - **Mode 3** (`is_ploop && is_sloop`) — exactly one `dma_engine.exec_dma_2d` call; NO `dispatch_iss_opcode` calls (test guard).
  - **Mode 4** (`is_ploop && is_tloop`) — single `(tmu_id, curr_id)` call.
- Pitfall 8 (Mode 3 OR-rule) covered with three tests — `sub_op=0 ⇒ is_load=True` regardless of opcode; `opcode=GTX_OP_DMA ⇒ is_load=True` regardless of sub_op; `(sub_op=1, opcode=VECTOR) ⇒ is_load=False`.
- Width/height extraction (`op3 & 0xFFFF` / `(op3 >> 16) & 0xFFFF`) explicitly asserted in two tests.
- dispatch_iss_opcode P3 stub: out-of-range nest/spu silently NOP, all DMA-relevant funct7s (`0x43 / 0x45 / 0x53`) NOP, unknown funct7 NOPs — never raises.
- Public surface preserved: `from riscv.gtx.dispatch import dispatch_4mode` resolves to the same callable as `from riscv.gtx.dispatch_4mode import dispatch_4mode` (verified by `is`-comparison test).

## Task Commits

1. **Task 1 RED — failing tests** — `259c18d` (test)
2. **Task 1 GREEN — dispatch_4mode + iss_opcode + re-export** — `4831bc6` (feat)

Note: dispatch.py's re-export block was first added by my GREEN edit, then was identically committed by Plan 02's `38aac36 feat(03-02)` before my commit landed. Both agents converged on the same canonical line; my GREEN commit ended up touching only `dispatch_4mode.py` because the dispatch.py change was already in HEAD by the time I committed. End-state is correct: one re-export line, no duplication.

## Files Created/Modified

- **CREATED** `src/main/python/riscv/gtx/dispatch_4mode.py` (120 LOC) — `dispatch_4mode` + `dispatch_iss_opcode`.
- **MODIFIED** `src/main/python/riscv/gtx/dispatch.py` — appended 5-line re-export block (Plan 02 commit `38aac36` carried the actual diff; Plan 04 GREEN commit landed empty for this file).
- **MODIFIED** `tests/gtx/test_dispatch_4mode.py` — Wave 0 placeholder removed, 13 real tests added (257 LOC total).
- **CREATED** `.planning/phases/03-dma-ddr-i-o/deferred-items.md` — documents 8 pre-existing failures in `tests/gtx/test_firmware_dma.py` that belong to Plan 02's write-set.

## Decisions Made

1. **Sibling-module split.** dispatch_4mode and dispatch_iss_opcode live in `dispatch_4mode.py` (NEW) instead of `dispatch.py` per CONTEXT.md "Defer to user follow-up". This eliminated the Wave 2 file-write conflict with Plan 02's 2-level builder upgrade. dispatch.py only gets a 1-line re-export.
2. **Cooperative cross-plan re-export.** When I went to add the re-export to dispatch.py, Plan 02's commit had already landed identically. End state is correct (one re-export line); no merge intervention needed.
3. **iss_opcode is a true stub, not a partial dispatch.** P3 NOPs every funct7 (including the three DMA-relevant ones at 0x43 / 0x45 / 0x53). The body has a comment block naming exactly which lines Plan 05 will replace with the credit_st_chk flush trigger. This keeps Plan 04 self-contained — Plans 04 and 05 don't need to merge code.
4. **Pitfall 8 dual coverage.** Both the boolean OR-rule (`is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)`) and the bitfield extraction (`width = op3 & 0xFFFF`, `height = (op3 >> 16) & 0xFFFF`) get explicit test assertions. A single check would let one piece silently regress.

## Deviations from Plan

None — plan executed exactly as written. The cooperative re-export with Plan 02 was anticipated by the plan's `<parallel_execution>` block ("if a merge conflict shows up at commit time, read the post-Plan-02 version and re-add the import line") and resolved cleanly without conflict.

## Issues Encountered

- **8 failing tests in `tests/gtx/test_firmware_dma.py` after Plan 02's `38aac36` landed.** All failures are in Plan 02's territory (test_firmware_dma.py is Plan 02's write-set; failures are `KeyError` on the captured-kwargs mock dict — a Plan 02-internal mocking issue). Plan 04's adjacent test suite (test_dispatch_4mode + test_dispatch + test_dma_engine + test_ddr_modes = 72 tests) all pass green. Logged in `deferred-items.md`; out of scope for Plan 04 per executor scope-boundary rules.

## Verification

- `pytest tests/gtx/test_dispatch_4mode.py --noconftest -o "addopts="` → **13/13 pass** ✓
- `pytest tests/gtx/test_dispatch.py --noconftest -o "addopts="` → **15/15 pass** (no regression from Plan 02) ✓
- `pytest tests/gtx/test_dma_engine.py tests/gtx/test_ddr_modes.py --noconftest -o "addopts="` → **44/44 pass** (no regression from Plan 01/03) ✓
- All 9 acceptance grep patterns match (file existence, `def dispatch_4mode`, `def dispatch_iss_opcode`, OR-rule, NEST/SPU loops, exec_dma_2d call, re-export line, min-LOC).

## Next Phase Readiness

- **Plan 05 (flush-roundtrip)** can now wire `credit_st_chk` (funct7=0x53) to call `npu.flush_deferred_ddr_stores()` from `is_sloop` context — the insertion point is comment-marked in `dispatch_4mode.dispatch_iss_opcode`. Plan 05 can either (a) replace the stub body in `dispatch_4mode.py` directly, or (b) introduce a runtime hook that `dispatch_iss_opcode` calls — its choice.
- **Phase 4 (MM)** will fill `GTX_OP_MM` (funct7=0) with the four MM variants in `dispatch_iss_opcode`. The stub returns 0 today; the pattern is to dispatch by `sub_op & 0x07` for MM variants per `gtx_npu_dispatch.cc:165-184`.
- **Phase 5 (VEC/ACT)** will fill `GTX_OP_VECTOR` (funct7=1) and `GTX_OP_ACTIVATION` (funct7=2) the same way.

## Self-Check: PASSED

- File `src/main/python/riscv/gtx/dispatch_4mode.py` — FOUND ✓
- File `src/main/python/riscv/gtx/dispatch.py` — FOUND with re-export ✓
- File `tests/gtx/test_dispatch_4mode.py` — FOUND ✓
- Commit `259c18d` (RED) — FOUND ✓
- Commit `4831bc6` (GREEN) — FOUND ✓
- All `must_haves.truths` (8) satisfied — VERIFIED via grep + 13/13 test pass ✓
- All `must_haves.key_links` (3 patterns: dma_engine.exec_dma_2d / `for n in range(GTX_NEST_NUM)` / `from .dispatch_4mode import`) — VERIFIED via grep ✓
- min_lines for dispatch_4mode.py (90) and test_dispatch_4mode.py (150) — both exceeded (120 / 257) ✓

---
*Phase: 03-dma-ddr-i-o*
*Completed: 2026-05-05*
