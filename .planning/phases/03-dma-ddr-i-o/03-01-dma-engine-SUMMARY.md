---
phase: 03-dma-ddr-i-o
plan: 01
subsystem: dma
tags: [dma, dma_engine, deferred_store, gtx, numpy, byte-exact]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: GtxMemory layered API (l0/l1/l2 raw byte accessors, ensure_ddr lazy alloc)
  - phase: 02-skeleton-disasm
    provides: WarpState dataclass, encoding constants (funct7/GSPR), package skeleton
provides:
  - dma_engine.py module (372 LOC) with 11 top-level callables
  - DeferredDdrStore frozen dataclass (7 fields, gtx_npu.h:1257-1266)
  - decode_firmware_dma_args (rs1/rs2/rs3 packed-arg decoder + is_copy carve-out + HW conventions)
  - 6 byte-exact pure helpers: exec_dma_2d / exec_load_svr / exec_store_svr / exec_transpose / exec_transpose_ddr / exec_fill
  - 4 firmware_dma branch helpers: sloop_load / sloop_store / tloop_load_store / tloop_copy
  - GTX_DDR_BASE = 0x370000000 in params.py
  - GSPR_GTX_OPERAND1/2/3/OPCODE = 0x001/0x002/0x003/0x004 in encoding.py (AUTHORITATIVE)
  - LSPR_SPM_ADDRA/B/C/R = 0x900/0x901/0x902/0x903 in encoding.py (AUTHORITATIVE)
  - GTX_ISS_F7_DMA_TPOSE/FILL/LD_ST/3D + CREDIT_ST_CHK + 4 funct7 stubs
  - WarpState.wsplit_seen field (process-lifetime sentinel, NOT cleared by reset())
  - 6 Wave 0 test scaffolds for downstream plans
affects:
  - 03-02-ops-dma (will register @handler entry points delegating to dma_engine)
  - 03-03-ddr-io (uses GTX_DDR_BASE; ensure_ddr upgrade)
  - 03-04-dispatch-4mode (Mode 3 calls exec_dma_2d)
  - 03-05-flush-roundtrip (uses firmware_dma_sloop_store + DeferredDdrStore queue)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Spike-independent helper module (CONTEXT D-01): pure functions on GtxMemory"
    - "frozen=True dataclass for HW-mirrored structs (Pitfall 4 lock-in)"
    - ".copy() guard on overlapping numpy slice assignments (matches C++ memmove)"
    - "AUTHORITATIVE constants flagged in source comments to prevent drift"

key-files:
  created:
    - src/main/python/riscv/gtx/dma_engine.py
    - tests/gtx/test_dma_engine.py
    - tests/gtx/test_firmware_dma.py
    - tests/gtx/test_deferred_store.py
    - tests/gtx/test_ddr_modes.py
    - tests/gtx/test_dma_roundtrip.py
    - tests/gtx/test_dispatch_4mode.py
    - .planning/phases/03-dma-ddr-i-o/03-01-dma-engine-SUMMARY.md
  modified:
    - src/main/python/riscv/gtx/params.py
    - src/main/python/riscv/gtx/encoding.py
    - src/main/python/riscv/gtx/warp_state.py

key-decisions:
  - "AUTHORITATIVE GSPR_GTX_OPERAND1/2/3/OPCODE = 0x001/0x002/0x003/0x004 (gtx_params.h:38-41); earlier drafts at 0x110..0x113 were WRONG"
  - "DeferredDdrStore declared frozen=True with exactly 7 fields in source-locked order to defend against producer/consumer drift (Pitfall 4)"
  - "is_copy carve-out: addr_hi = (rs1>>32) if is_copy else ((rs1>>27)&0x1FFFFFFFFF) -- COPY uses 32-bit dst, LOAD/STORE use 37-bit hi (Pitfall 1)"
  - "HW conventions applied at decode: length=0 -> 65536, height=0 -> 1 (Pitfall 2)"
  - "WarpState.wsplit_seen NOT touched by reset() -- process-lifetime sentinel (Pitfall 7)"
  - "Wave 0 placeholder body uses pytest.skip() rather than assert hasattr (revision iter 1 Warning 6 fix); collection passes cleanly"

patterns-established:
  - "AUTHORITATIVE comment marker for gtx_params.h-sourced constants -- prevents drift if future planners regenerate from older drafts"
  - "DeferredDdrStore frozen=True invariant -- mutation attempts raise FrozenInstanceError; producer/consumer field drift caught by len(fields(...))==7 assertion"
  - "decode_firmware_dma_args returns dict not @dataclass -- callers (Plan 02 ops/dma.py) pick fields they need without coupling"
  - "Spike-independent helpers take GtxMemory + kwargs; addr_a/addr_r read by caller from spu.lspr -- helper stays pure"
  - ".copy() on numpy source slice for overlapping memmove (firmware_dma_tloop_copy + exec_transpose + exec_transpose_ddr)"

requirements-completed: [DMA-01]

# Metrics
duration: ~9min
completed: 2026-05-05
---

# Phase 03 Plan 01: DMA Engine Summary

**Spike-independent DMA helpers + DeferredDdrStore + firmware_dma decoder, byte-exact ports of vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc with all 4 critical pitfalls (is_copy carve-out, length/height HW conventions, frozen 7-field struct, wsplit_seen sentinel) defended by tests.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-05T14:05:00Z (approx, parallel wave 1)
- **Completed:** 2026-05-05T14:14:09Z
- **Tasks:** 2
- **Files created:** 7
- **Files modified:** 3
- **Tests added:** 27 (Task 1: 7 const/state, Task 2: 20 dma_engine helpers)
- **Tests passing:** 27/27

## Accomplishments

- Phase 3 DMA bedrock landed: dma_engine.py (372 LOC, 11 top-level callables) is the spike-independent layer Plans 02/04/05 depend on
- 6 Wave 0 test scaffolds created with pytest.skip() placeholders so downstream plans can target their `<verify>` commands
- AUTHORITATIVE constants (GTX_DDR_BASE, GSPR_GTX_OPERAND1/2/3, LSPR_SPM_ADDRA/B/C/R) sourced from gtx_params.h:24/38-41/64-67 and locked with explicit comments
- Pitfall 1 (is_copy rs1>>32 carve-out), Pitfall 2 (length=0->65536, height=0->1), Pitfall 4 (DeferredDdrStore frozen + 7-field assertion), Pitfall 7 (wsplit_seen NOT reset) -- all four exercised by dedicated tests
- All 27 tests pass via `pytest tests/gtx/test_dma_engine.py -x --noconftest -o "addopts="` (~0.65s)

## Task Commits

Each task was committed atomically with `--no-verify` (parallel wave 1 contention guard).

1. **Task 1: Constants + WarpState.wsplit_seen + 6 Wave 0 scaffolds** -- `3928da7` (feat)
2. **Task 2: dma_engine.py (DeferredDdrStore + decode + 6 exec_* + 4 firmware_dma_*)** -- `65c31f9` (feat)

**Plan metadata commit:** pending (final docs commit at end of plan execution)

_Note: TDD followed (RED -> GREEN). Each task ran tests-first; failing tests proved the absence of state, then minimal implementation made them pass._

## Files Created/Modified

### Created
- `src/main/python/riscv/gtx/dma_engine.py` -- 372 LOC. The spike-independent DMA bedrock: DeferredDdrStore frozen dataclass + decode_firmware_dma_args + 6 exec_* helpers + 4 firmware_dma_* branch helpers
- `tests/gtx/test_dma_engine.py` -- 454 LOC, 27 tests covering constants, WarpState.wsplit_seen persistence, all helpers, all decode pitfalls
- `tests/gtx/test_firmware_dma.py` -- Wave 0 scaffold (Plan 02 will fill)
- `tests/gtx/test_deferred_store.py` -- Wave 0 scaffold (Plan 05 will fill)
- `tests/gtx/test_ddr_modes.py` -- Wave 0 scaffold (Plan 03 will fill)
- `tests/gtx/test_dma_roundtrip.py` -- Wave 0 scaffold (Plan 05 will fill)
- `tests/gtx/test_dispatch_4mode.py` -- Wave 0 scaffold (Plan 04 will fill)

### Modified
- `src/main/python/riscv/gtx/params.py` -- Added GTX_DDR_BASE = 0x370000000 (gtx_params.h:24)
- `src/main/python/riscv/gtx/encoding.py` -- Added 9 funct7 + 4 GSPR_GTX_OPERAND/OPCODE + 4 LSPR_SPM_ADDR constants
- `src/main/python/riscv/gtx/warp_state.py` -- Added wsplit_seen field; reset() does NOT clear it (Pitfall 7)

## Decisions Made

1. **AUTHORITATIVE constant sourcing:** GSPR_GTX_OPERAND1/2/3/OPCODE = 0x001/0x002/0x003/0x004 per gtx_params.h:38-41 (orchestrator-verified, revision iteration 1). Earlier draft values 0x110..0x113 were WRONG and would silently break GSPR-staged operand reads. Source comments flag this explicitly to prevent drift.
2. **DeferredDdrStore = frozen=True dataclass with exactly 7 fields:** Mutation attempts raise FrozenInstanceError; producer/consumer field drift caught by `len(dataclasses.fields(DeferredDdrStore)) == 7` assertion. Direct port of `deferred_ddr_store_t` (gtx_npu.h:1257-1266).
3. **is_copy carve-out at decode:** `addr_hi = (rs1>>32) if is_copy else ((rs1>>27)&0x1FFFFFFFFF)`. COPY funct3=010 path uses 32-bit dst (L1 is 384KB = 19 bits, fits within 27); LOAD/STORE use 37-bit DDR hi address (Pitfall 1).
4. **HW conventions at decode (NOT at engine):** `length=0 -> 0x10000`, `height=0 -> 1` applied immediately after extracting raw 16-bit fields. Engine receives already-resolved values and never sees the raw zero-sentinel form (Pitfall 2 lock-in).
5. **WarpState.wsplit_seen is process-lifetime sentinel:** initialized once to False, set True by WSPLIT, NOT cleared by reset(). Matches C++ field initializer in gtx_npu.h:1251 (Pitfall 7). Test asserts persistence: `w.reset()` clears is_ploop but `w.wsplit_seen` remains True.
6. **Wave 0 placeholder body = `pytest.skip()` (not `assert hasattr`):** Revision iter 1 Warning 6 fix. Earlier draft used `assert hasattr(...)` placeholders which would FAIL during the verify step before downstream plans filled them. `pytest.skip()` placeholders pass cleanly until the real test body lands.
7. **`.copy()` guard on overlapping slice assignment:** firmware_dma_tloop_copy + exec_transpose + exec_transpose_ddr all use `dst = src.copy()` for the LHS operand. Matches C++ `std::memmove` semantics; bare numpy slice assignment can corrupt overlapping ranges.
8. **Wave 0 skipif strategy:** test_dma_engine.py and test_ddr_modes.py have NO skipif (pure-python helpers); the four others have module-level `_RISCV_AVAILABLE` self-detect + `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)` to match `--noconftest` acceptance command from VALIDATION.md.

## Deviations from Plan

None - plan executed exactly as written.

The plan was unusually detailed (37.5KB) with explicit signatures, line citations to C++ source, and pitfall references. Each task ran TDD (RED -> GREEN) without surprises. All acceptance criteria (grep checks + 27 tests passing) satisfied on first commit.

## Issues Encountered

- **Read-tool intermittent context dump:** Reading the C++ DMA source surfaced two CLAUDE.md `<system-reminder>` blocks (vendor/gtx_cpp_reference/CLAUDE.md and gtx/CLAUDE.md) mid-task. These were noted but did not affect implementation (already-internalized FP16 LE byte order + GTX_DDR_REVERSED conventions matched what RESEARCH.md prescribed). No state change required.
- **STATE.md modification not from this plan:** STATE.md was modified by the orchestrator before the executor was spawned (Phase 2->3 transition). Not included in any task commit; will be picked up by the final plan-metadata commit.

## User Setup Required

None - no external service configuration required. All work is pure Python module + test scaffolding within the existing pyspike build.

## Next Phase Readiness

**Plan 02 (ops/dma.py):** Ready to register `@handler(funct7=0x40, funct3=...)` entry points that read `proc.get_state().XPR[...]`, call `decode_firmware_dma_args(...)`, then dispatch to the right `firmware_dma_*` branch helper from `dma_engine.py`. The decode dict has all field names Plan 02's entry points need.

**Plan 03 (ddr.py):** Ready to upgrade `ensure_ddr` to doubling-grow + add `ddr_init_from_file` / `ddr_dump_to_file` with `GTX_DDR_REVERSED`. `GTX_DDR_BASE = 0x370000000` is already in `params.py`. test_ddr_modes.py scaffold awaits Plan 03 fill.

**Plan 04 (dispatch_4mode.py):** Mode 3 (`is_ploop && is_sloop`) needs to call `dma_engine.exec_dma_2d` from `dispatch_4mode`. WarpState fields (tmu_id, curr_id) are already in place from Phase 2. test_dispatch_4mode.py scaffold awaits Plan 04 fill.

**Plan 05 (flush + roundtrip):** Ready to use `dma_engine.firmware_dma_sloop_store(npu, ...)` for the deferred-queue push test, then add `npu.flush_deferred_ddr_stores()` consumer. test_deferred_store.py + test_dma_roundtrip.py scaffolds await Plan 05 fill.

**Wave 1 parallel safety:** Task commits used `--no-verify` to avoid pre-commit hook contention with the concurrent 03-03-ddr-io agent. Orchestrator validates hooks once after the wave completes.

## Self-Check: PASSED

- ✓ src/main/python/riscv/gtx/dma_engine.py exists (372 lines)
- ✓ tests/gtx/test_dma_engine.py exists (454 lines, 27 tests)
- ✓ All 5 Wave 0 placeholder scaffolds exist
- ✓ Commit 3928da7 (Task 1) found in `git log --oneline`
- ✓ Commit 65c31f9 (Task 2) found in `git log --oneline`
- ✓ All 27 tests pass via `pytest tests/gtx/test_dma_engine.py -x --noconftest -o "addopts="`
- ✓ All grep-based acceptance criteria match (frozen=True, is_copy carve-out, length=0 -> 0x10000, height=0 -> 1, ddr_off translation, .copy() guard, 12 top-level def/class >= 11)
- ✓ All 7 must_haves.truths satisfied (6 byte-exact helpers, DeferredDdrStore 7 fields, decode bit fields, length/height conventions, wsplit_seen sentinel, AUTHORITATIVE addresses, Wave 0 scaffolds)

---
*Phase: 03-dma-ddr-i-o*
*Completed: 2026-05-05*
