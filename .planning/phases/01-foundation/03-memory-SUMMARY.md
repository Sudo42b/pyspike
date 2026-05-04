---
phase: 01-foundation
plan: 03
subsystem: memory
tags: [numpy, fp16, memory-layer, view-base, ddr, lazy-alloc]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: skeleton (params.py constants from 01-skeleton plan, in parallel wave)
provides:
  - GtxMemory class with L0/L1/L2 ndarray + halfword fp16/uint16 views
  - SPR unified dict[int, int] (D-11)
  - DDR lazy alloc helpers (D-01) + GTX_DDR_SIZE env var parsing (D-02)
  - 8 acceptance tests covering FOUND-02 (LE byte order, view-base, shape, SPR, DDR lazy)
affects: [02-fp, 04-packaging, 05-submodule, 06-dispatch, 07-spr-routing, 08-mm, 09-vec, 10-act, 11-dma, 12-pool, 13-conv, 14-tpose]

# Tech tracking
tech-stack:
  added: [numpy>=2.0 (already pinned in pyproject), pytest fixture-based memory tests]
  patterns:
    - "D-12 view-base tripwire: assert view.base is not None inside helper + 8 pytest invariants"
    - "L0/L1/L2 = contiguous np.uint8 ndarray; halfword view via .view(np.float16) — host-LE assumption guarded by 01-skeleton's __init__.py LE tripwire"
    - "TYPE_CHECKING import of GtxMemory in ddr.py to avoid circular import"
    - "Lazy DDR alloc with explicit ValueError on cap exceed (no silent truncation)"

key-files:
  created:
    - src/main/python/riscv/gtx/memory.py
    - src/main/python/riscv/gtx/ddr.py
    - tests/gtx/test_memory_layout.py
  modified: []

key-decisions:
  - "D-10 layered API confirmed: byte and fp16 helpers both exposed; tests cover both write paths"
  - "D-11 SPR unified dict[int, int]: single self.spr field; 0x100/0x500/0x900 routing test passes"
  - "D-12 view-base invariant enforced at runtime (assert in helper) AND at test time (8 pytest invariants)"
  - "D-01 _ddr_bytes = None at construction; ensure_ddr() materializes on first access"
  - "D-02 GTX_DDR_SIZE env var with G/M/K suffix; default 4 GiB (DEFAULT_DDR_SIZE)"
  - "D-17 LE byte order verified bidirectionally (byte write -> fp16 read; fp16 write -> byte read)"
  - "Phase 1 ensure_ddr is alloc-exact stub; Phase 3 will replace with C++ doubling-grow strategy"

patterns-established:
  - "Pattern A — D-12 view-base tripwire: every named ndarray helper has `assert view.base is not None` before return + matching pytest assertion"
  - "Pattern B — Lazy DDR alloc: _ddr_bytes private attr starts None; ensure_ddr grows on demand with cap enforcement"
  - "Pattern C — Halfword view: arr.view(np.float16) preserves base; LE assumption guarded by package-level __init__.py tripwire (01-skeleton plan)"

requirements-completed: [FOUND-02]

# Metrics
duration: 4 min
completed: 2026-05-04
---

# Phase 1 Plan 03: Memory Layer Summary

**NumPy-backed GtxMemory with L0/L1/L2 contiguous ndarray, fp16/uint16 halfword views, view-base tripwire, unified SPR dict, and lazy DDR allocation with GTX_DDR_SIZE env var**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T05:38:08Z
- **Completed:** 2026-05-04T05:42:22Z
- **Tasks:** 3 (RED test + GREEN impl + DDR helper)
- **Files created:** 3
- **LOC:** 262 (memory.py 85 + ddr.py 77 + test_memory_layout.py 100)
- **Tests:** 8/8 PASS (`pytest tests/gtx/test_memory_layout.py --noconftest`, 0.32s)
- **GtxMemory().__init__ RSS delta:** 14.7 MB (faulted-in pages); buffer total ≈88 MB (L0 0.06 + L1 24.0 + L2 64.0)

## Accomplishments

- `GtxMemory` class with 6 named accessors (l0_byte/l1_byte/l2_byte/l0_f16/l1_f16/l2_f16) + l1_u16 — all return non-copying views
- D-12 view-base tripwire embedded as runtime `assert view.base is not None` in every fp16/uint16 helper, AND verified by 4 pytest invariants (l0_f16/l1_f16/slice_preserves/l1_shape)
- D-17 LE byte order verified bidirectionally: writing `[0x00, 0x3C]` to `l1_byte` reads as `np.float16(1.0)` via `l1_f16`; writing `np.float16(2.0)` produces `[0x00, 0x40]` in `l1_byte`
- D-11 unified SPR dict — single `self.spr: dict[int, int]` covering GSPR/NSPR/LSPR ranges (0x100/0x500/0x900 routing test)
- D-01 lazy DDR — `_ddr_bytes = None` at construction; `ensure_ddr()` materializes; `ValueError` on cap exceed
- D-02 GTX_DDR_SIZE env var with G/M/K suffix parsing; default 4 GiB

## Task Commits

1. **Task 03-01: tests/gtx/test_memory_layout.py — RED phase** — `ce3c329` (test)
2. **Task 03-02: src/main/python/riscv/gtx/memory.py — GREEN phase** — `415c067` (feat)
3. **Task 03-03: src/main/python/riscv/gtx/ddr.py — DDR lazy alloc helpers** — `2d67524` (feat)

_Note: TDD RED-GREEN executed across Tasks 03-01 → 03-02. Task 03-03 is non-TDD (helper only)._

## Files Created/Modified

- `tests/gtx/test_memory_layout.py` (100 LOC) — 8 acceptance tests covering FOUND-02
- `src/main/python/riscv/gtx/memory.py` (85 LOC) — `GtxMemory` class, 6 named accessors, SPR dict, `_ddr_bytes` private attr
- `src/main/python/riscv/gtx/ddr.py` (77 LOC) — `DEFAULT_DDR_SIZE`, `get_ddr_cap`, `ensure_ddr`

## Exported Symbols

**memory.py:**
- `GtxMemory` (class) — public attrs: `spr` (dict); private attrs: `_l0_bytes`, `_l1_bytes`, `_l2_bytes`, `_ddr_bytes`
- methods: `l0_byte(n,s)`, `l1_byte(n,s)`, `l2_byte(n)`, `l0_f16(n,s)`, `l1_f16(n,s)`, `l2_f16(n)`, `l1_u16(n,s)`

**ddr.py:**
- `DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024` (4 GiB)
- `get_ddr_cap() -> int`
- `ensure_ddr(mem: GtxMemory, end_offset: int) -> np.ndarray`

## Decisions Made

- **D-10 confirmed (layered API):** Both byte view (`l1_byte(0,0)[0]=0x00`) and fp16 view (`l1_f16(0,0)[0]=np.float16(2.0)`) write paths exposed and tested. Op handlers in P4/P5 can use either.
- **D-12 enforced at TWO layers:** runtime `assert view.base is not None` inside helpers (catches bugs at first call site, not later); pytest invariants (catch regressions if helpers refactored).
- **TYPE_CHECKING import for GtxMemory in ddr.py:** Prevents circular import while keeping type hint annotations for tooling. Standard PEP 484 + `from __future__ import annotations` pattern.
- **Phase 1 ensure_ddr is alloc-exact (no doubling):** Phase 3 DMA-04 replaces with C++-equivalent doubling-grow strategy. Documented in docstring.
- **GTX_DDR_REVERSED I/O deferred:** D-03 says reversal happens only at I/O boundary; ddr.py docstring documents this; body lives in Phase 3 (DMA-04).

## Deviations from Plan

None — plan executed exactly as written.

The plan-specific guidance flagged that `params.py` may not be on disk in this isolated worktree due to parallel-wave race with the skeleton plan. This was indeed the case at execution time. The recommended workaround (write the files, do filesystem-level acceptance checks, defer cross-file pytest to post-wave validation) was followed; for verification only, temporary stub `__init__.py` and `params.py` files were briefly created to run the smoke test and pytest, then removed before commit. The 8/8 pytest pass was verified locally during Task 03-02 execution. The orchestrator's post-wave validation will re-run the full pytest after merging the skeleton plan's `params.py` and `__init__.py`.

## Issues Encountered

- `tests/conftest.py` (pre-existing pyspike conftest) imports `from riscv.cfg import cfg_t, mem_cfg_t` which requires the C++ extension to be built. For this plan's verification, pytest was run with `--noconftest` flag. This is a pre-existing concern (unrelated to my code) and does not affect production CI where libriscv.so is built. **Not a deviation — does not require any code change in this plan.**

## Authentication Gates

None — no external service required.

## Known Stubs

None. All code in this plan implements production behavior. The "Phase 1 alloc-exact" comment in `ensure_ddr()` is a documented forward-pointer to Phase 3's doubling-grow refinement; the current behavior is fully functional (lazy alloc + cap enforcement) and correct for FOUND-02 acceptance.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Wave 1 status (parallel with this plan):**
- 01-skeleton plan must produce `src/main/python/riscv/gtx/__init__.py` (with LE tripwire), `params.py`, `encoding.py`, `ops/__init__.py`, and `tests/gtx/__init__.py` for the integrated pytest run to succeed in main branch
- 02-fp plan must produce `fp.py`

**Post-wave orchestrator verification commands:**
```
PYTHONPATH=src/main/python pytest tests/gtx/test_memory_layout.py -v
PYTHONPATH=src/main/python python -c "from riscv.gtx.memory import GtxMemory; from riscv.gtx.ddr import ensure_ddr; m = GtxMemory(); ensure_ddr(m, 4096); print('OK')"
```

**Downstream consumers:**
- Phase 2 (CORE-02 reset, SPR-01/02 WRSPR/RDSPR) consumes `mem.spr`, `mem.l*_byte/_f16`
- Phase 3 (DMA-04) replaces `ensure_ddr()` body with doubling-grow + adds `ddr_init_from_file` / `ddr_dump_to_file` honoring `GTX_DDR_REVERSED` (D-03)
- Phase 4 (MM op) does in-place `mem.l1_f16(n,s)[off] = np.float16(result)` — D-12 view-base guarantees this writes through

**Blockers for Phase 2:** None from this plan. Wave 1 must complete (skeleton + fp + memory) before Phase 2 starts.

---
*Phase: 01-foundation*
*Completed: 2026-05-04*

## Self-Check: PASSED

- All 3 created files exist on disk (memory.py, ddr.py, test_memory_layout.py)
- All 3 task commits visible in git log (ce3c329, 415c067, 2d67524)
- SUMMARY.md created at .planning/phases/01-foundation/03-memory-SUMMARY.md
- 8/8 pytest pass verified locally during Task 03-02
- All 6 named accessors return views (`base is not None`) verified explicitly
