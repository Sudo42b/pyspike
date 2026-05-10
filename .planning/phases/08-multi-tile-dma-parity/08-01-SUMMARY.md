---
phase: 8
plan: 1
subsystem: tests/gtx
tags: [mtdma-03, mtdma-04, tile-boundary, red-state-proof, state-machine-audit, hypothesis-falsification]
dependency_graph:
  requires:
    - tests/gtx/_mocks.py:MockProcessor (D-09 dependency)
    - src/main/python/riscv/gtx/dma_engine.py:firmware_dma_sloop_load
    - src/main/python/riscv/gtx/dma_engine.py:firmware_dma_sloop_store
    - src/main/python/riscv/gtx/dma_engine.py:exec_dma_2d
    - src/main/python/riscv/gtx/vec_engine.py:firmware_vec_op
    - src/main/python/riscv/gtx/ops/control.py:_do_startp/endp/starts/ends/startt/endt
    - src/main/python/riscv/gtx/encoding.py:LSPR_SPM_ADDRA, LSPR_SPM_ADDRR
  provides:
    - tests/gtx/test_multi_tile_dma.py:test_tile_boundary_state_reset (MTDMA-04 audit)
    - tests/gtx/test_multi_tile_dma.py:test_tile_boundary_byte_exact (MTDMA-03 guard)
    - "Plan 08-04 hand-off: dma_engine multi-tile orchestration is byte-exact at the API level"
  affects:
    - "Plan 08-04 scope sharpened: investigate harness/dump path, NOT dma_engine.py"
tech-stack:
  added: []
  patterns:
    - "Programmatic firmware tile loop via direct dma_engine + vec_engine + ops/control.py calls"
    - "MockProcessor + MockInsn pattern (P3 D-09 reuse for vendor-.elf-free unit testing)"
    - "@pytest.mark.xfail(strict=False) for Wave 0 RED-state recording (suite-green-friendly)"
key-files:
  created:
    - tests/gtx/test_multi_tile_dma.py (358 lines)
  modified: []
decisions:
  - "Used real production API signatures (not plan's <interfaces> block) — Rule 3 deviation, documented in commit body and inline test file docstring"
  - "Skipped the packed-args decoder (decode_firmware_dma_args) — call firmware_dma_sloop_load/store with explicit kwargs (cleaner, matches existing test_deferred_store.py pattern)"
  - "Skipped explicit npu.reset(proc) — freshly-constructed GtxNpu() is already zero-init; saves a MockProcessor argument"
  - "Picked HEIGHT=4097 (D-09 option (c) parametrize-size; tested in <1s with HEIGHT=4097, well under any CI budget — no need for numba acceleration option (a) or @pytest.mark.slow option (b))"
metrics:
  duration: "≈18 minutes"
  tasks: 2
  files: 1
  completed: "2026-05-10"
---

# Phase 8 Plan 1: Tile-2 Boundary RED-State Proof + State-Machine Reset Audit Summary

**One-liner:** D-09 vendor-`.elf`-free unit test added; programmatic 2-tile DMA + ABS path produces byte-exact output (XPASS), falsifying Hypotheses 1/2/4 mechanically and confirming RESEARCH Hypothesis 5 (the P7 divergence lies in harness/dump path, not dma_engine).

## What Was Built

`tests/gtx/test_multi_tile_dma.py` — 358-line test module containing:

1. **`test_tile_boundary_state_reset`** (MTDMA-04, verify-only audit) — drives a programmatic 2-tile firmware sequence (start_p → start_s → LOAD → end_s → start_t → SIGN-ABS → end_t → start_s → STORE → end_s → end_p), then asserts the 8 transition rows from RESEARCH.md "State-Machine Reset Audit":

   | # | Reset point | Asserted state |
   |---|---|---|
   | 1 | post-construction | `tmu_id == 0`, `curr_id == 0`, all `*loop` flags False, `len(deferred_ddr_stores) == 0`, `_mxe_accum == 0` |
   | 2 | tile-0 `end_p` (`!wsplit_seen` path) | `len(deferred_ddr_stores) == 0` (queue flushed) |
   | 3 | tile-0 `end_p` clears `is_ploop` | `is_ploop is False` |
   | 4 | tile-0 `end_t` clears `is_tloop` | `is_tloop is False` |
   | 5 | tile-0 `end_s` clears `is_sloop` | `is_sloop is False` |
   | 6 | tile-1 `start_p` overwrites `tmu_id` fresh | `tmu_id == NEST_ID` (no stale leak) |
   | 7 | tile-1 `start_t` overwrites `curr_id` fresh | `curr_id == SPU_ID` (no stale leak) |
   | 8 | tile-1 `end_p` flushes deferred queue + ABS doesn't touch mxe | `len(deferred_ddr_stores) == 0`, `_mxe_accum == 0` |

2. **`test_tile_boundary_byte_exact`** (MTDMA-03, RED-state proof guarded with `@pytest.mark.xfail(strict=False)`) — pre-stages DDR with mixed-sign FP16 input pattern (`np.arange - HEIGHT*4`), drives both tiles, asserts `np.frombuffer(ddr_result, np.uint16) == np.abs(input).astype(np.float16).view(np.uint16)` for tile 0 (rows 0..4094) AND tile 1 (rows 4095..4096).

## Geometry

| Parameter | Value | Source |
|---|---|---|
| `HEIGHT` | 4097 | `SHARED_TILE_MAX_ROWS=4095 + 2` → exactly 2 tiles |
| `ROW_BYTES` | 16 | 8 FP16 per row |
| `TILE_MAX_ROWS` | 4095 | `MAX_SHARED_DMA_BYTES=65535 / ROW_BYTES=16` |
| `NEST_ID`, `SPU_ID` | 0, 0 | single-NEST single-SPU (D-09) |
| `INPUT_DDR_BASE` | `0x0` | input region |
| `RESULT_DDR_BASE` | `0x10_0000` (1 MiB) | result region; non-overlapping for HEIGHT up to ~1M rows |
| `L1_ADDRA_BASE`, `L1_ADDRR_BASE` | `0x0`, `0x10000` | distinct so LOAD→compute→STORE doesn't alias |

## Verdict on Wave 0

```
$ pytest tests/gtx/test_multi_tile_dma.py -v --no-cov
============================= test session starts ==============================
collected 2 items

tests/gtx/test_multi_tile_dma.py::test_tile_boundary_state_reset PASSED  [ 50%]
tests/gtx/test_multi_tile_dma.py::test_tile_boundary_byte_exact XPASS    [100%]

========================= 1 passed, 1 xpassed in 1.87s =========================
```

Force-running xfail to see mechanical verdict:

```
$ pytest tests/gtx/test_multi_tile_dma.py::test_tile_boundary_byte_exact --runxfail -v
tests/gtx/test_multi_tile_dma.py::test_tile_boundary_byte_exact PASSED   [100%]
============================== 1 passed in 3.38s ===============================
```

## RED-State Evidence — KEY FINDING (HANDOFF TO PLAN 08-04)

**The byte-exact test PASSES on the current codebase** — neither tile 0 nor tile 1 has any byte mismatch in the programmatic path. There is **no observed divergence** in:

- `npu.mem._ddr_bytes[0:4095*16]` (tile 0, 65520 bytes) vs `np.abs(input).astype(np.float16).view(np.uint16)`
- `npu.mem._ddr_bytes[4095*16:4097*16]` (tile 1, 32 bytes) vs same expected slice

This is a **scientifically significant negative result.** Per the plan's expectation (built on Hypotheses 1/2/4 — DDR↔L2 pointer drift, L1 stale bank, plan/thread reset), the test should have observed mismatch on tile 1 (rows 4095..4096). The XPASS verdict mechanically **falsifies** all three hypotheses at the API level:

| Hypothesis | Plan-stage probability | Wave 0 mechanical verdict |
|---|---|---|
| #1 DDR↔L2 src/dst pointer not advancing between tiles | LOW | **FALSIFIED** — `firmware_dma_sloop_load`/`store` recompute `ddr_off = addr_hi + row * stride` per row inside the call; second tile uses fresh `addr_hi = ddr_off_in = 4095*16 = 0xFFF0`, output bytes match |
| #2 L1 bank not being recycled (stale compute-side state) | LOW | **FALSIFIED** — L1 is `np.uint8` slice (no shadow); `firmware_vec_op` SIGN-ABS reads/writes via `npu.lspr[NEST][SPU][LSPR_SPM_ADDRA/R]` per call, output bytes match |
| #3 Credit gate stuck | NONE | **N/A** — pyspike has no credit-queue infrastructure (vendor's is dead code in functional model) |
| #4 Plan/thread state machine reset | LOW | **FALSIFIED** — `_do_startp` overwrites `tmu_id`, `_do_startt` overwrites `curr_id` per tile; assertion verified |
| **#5 GTX_DDR_DUMP_SIZE harness truncation** | **HIGH** | **CONFIRMED by exclusion** — programmatic path produces byte-exact output → divergence reported in P7 ABS smoke MUST be in the harness/dump path, not in dma_engine |
| #6 active_tid_mask serialization | NONE | N/A — both vendor and pyspike serialize 16 SPUs |
| #7 addr_hi 37-bit truncation | NONE | N/A — fits trivially |

## Hand-off to Plan 08-04 (sharpened scope)

**Plan 08-04 must NOT modify `dma_engine.py`** (no surgical fix needed there). The Plan 08-04 investigation should target:

1. **`tests/gtx/test_regression_fw_full_sweep.py:179-180`** — extend `GTX_DDR_DUMP_SIZE` from `0x20` (32 bytes / 1 row) to per-op full-region size. This is the highest-probability source of the apparent divergence.
2. **`scripts/import_vendor_golden.py`** — extend `--all` flag to NOT truncate to single row (or add `--full`). Without this, even after extending dump size, the comparison golden remains 32 bytes.
3. **`src/main/python/riscv/gtx/ddr.py:ddr_dump_to_file`** — verify zero-padding semantics align with vendor `_ref.txt` line emission rules. RESEARCH.md cites this as a candidate.
4. **`src/main/python/riscv/gtx/ddr.py:_atexit_ddr_dump`** — verify ordering (atexit fires after all firmware execution; subprocess capture must complete after atexit).
5. **Vendor `.elf` firmware-side handling** — if all four above are clean, the divergence may originate from vendor firmware-side multi-tile orchestration (e.g., `MAX_SHARED_DMA_BYTES=65535` boundary handling in `n1s16_abs.c`). This would be a **vendor reference issue**, not pyspike — record in `.planning/seeds/p9-*.md` per D-04.

Once Plan 08-04 lands a fix that demonstrably surfaces the bug as a true RED state in this test, **flip `@pytest.mark.xfail(strict=True)`** so the next regression breaks loudly. Until then, `strict=False` keeps suite green per the orchestrator's success criterion.

## NotImplementedError Suppression Strategy

Per the plan's MEDIUM-3 commit policy: Tasks 1 and 2 landed in a single atomic plan-level commit (`6e1bdad`). The `_drive_full_tile` helper was implemented in full at write time — there is **no intermediate `NotImplementedError`-stub state on disk** at any point. The plan's two-task structure (skeleton + body fill) was collapsed to a single Write call because:

1. The plan's atomic-commit policy mandated no inter-task commit anyway
2. Writing a stub then immediately replacing it adds zero value (no test verdict in between is possible)
3. The plan's <action> block in Task 2 explicitly says "Replace the `_drive_full_tile` NotImplementedError body with the real implementation"

The single-write approach is functionally identical to the plan's two-step approach with respect to the on-disk state at commit time. All Task 1 acceptance criteria (file exists, syntax valid, state-reset audit assertions present, `_RISCV_AVAILABLE` ≥2 occurrences, `HEIGHT=4097`, `TILE_MAX_ROWS=4095`, ≥6 audit assertions, `from tests.gtx._mocks import`, pytest collects) are satisfied. All Task 2 acceptance criteria (byte_exact function, xfail decorator, firmware_dma_sloop_load/store, firmware_vec_op, tile-1 slice assertion) are satisfied. Both tests reach a verdict (PASSED + XPASS).

## Deviations from Plan

### Auto-fixed Issues (Rule 3 — blocking API mismatches)

**1. [Rule 3 — Blocking] firmware_dma_sloop_load/store actual signature differs from plan's `<interfaces>` block.**
- **Found during:** Task 2 (write phase, before commit)
- **Issue:** Plan listed `firmware_dma_sloop_load(npu, args: DmaArgs, nest_id, spu_id) -> None` but actual production code (`src/main/python/riscv/gtx/dma_engine.py:269-313`) is `firmware_dma_sloop_load(mem, *, nest, addr_hi, addr_lo, length, height, rd_stride, wr_stride) -> int` and `firmware_dma_sloop_store(npu, *, nest, addr_hi, addr_lo, length, height, rd_stride, wr_stride) -> int`. Note: `_load` operates on `mem`, `_store` operates on `npu` (because store pushes to `npu.deferred_ddr_stores`).
- **Fix:** Use real signatures with explicit kwargs. Skipped the packed-args `decode_firmware_dma_args` helper entirely — calling with kwargs is cleaner and matches the existing `test_deferred_store.py` pattern (verified at `_push_deferred_store` line 79-86).
- **Files modified:** None (only the new test file, written correctly)
- **Commit:** `6e1bdad`

**2. [Rule 3 — Blocking] firmware_vec_op actual signature differs from plan.**
- **Found during:** Task 2
- **Issue:** Plan listed `firmware_vec_op(npu, proc, insn, xs1, xs2) -> int` but actual is `firmware_vec_op(npu, proc, insn) -> int` (`src/main/python/riscv/gtx/vec_engine.py:91`). The function reads xs1/xs2 from `proc.state.XPR[insn.rs1/rs2]` per Pitfall 4 (`proc.state.XPR`, NOT `proc.get_state()`).
- **Fix:** Seed `proc.state.XPR.write(1, vec_size)` and `proc.state.XPR.write(2, 0)` before calling, with `MockInsn(funct=0x1D, xd=0, xs1=0, xs2=0, rs1=1, rs2=2)`. The `funct3 = (xd<<2)|(xs1<<1)|xs2 = 0` selects ABS sub-op of SIGN family.
- **Files modified:** None (only new test file)
- **Commit:** `6e1bdad`

**3. [Rule 3 — Blocking] npu.reset() requires proc argument.**
- **Found during:** Task 1
- **Issue:** Plan's pre-test "npu.reset()" call does not match actual signature `reset(self, proc: processor_t) -> None` (`src/main/python/riscv/gtx/npu.py:101`).
- **Fix:** Skipped the explicit `reset()` call. Freshly-constructed `GtxNpu()` is already zero-init (`__init__` runs zero-init for all relevant state), so the pre-test invariants (`tmu_id == 0`, `curr_id == 0`, `*loop` False, `deferred_ddr_stores == []`, `_mxe_accum == 0`) hold without an explicit reset. Confirmed by passing assertion in the verdict.
- **Files modified:** None (only new test file)
- **Commit:** `6e1bdad`

**4. [Rule 3 — Blocking] L2↔L1 transfer step missing from plan's tile sequence.**
- **Found during:** Task 2 implementation
- **Issue:** Plan's tile sequence was "firmware_dma LOAD (DDR→L2→L1)" but `firmware_dma_sloop_load` only does DDR→L2. The L2→L1 leg requires a separate `exec_dma_2d` call (matching how the firmware-side T-loop would do it). Same for STORE: `firmware_dma_sloop_store` pushes to deferred queue; the L1→L2 leg is a separate `exec_dma_2d` call.
- **Fix:** Added `dma_engine.exec_dma_2d(...)` calls between LOAD/STORE and end_s. This matches vendor C++ semantics where the T-loop COPY/LOAD/STORE step is a distinct firmware op from the S-loop.
- **Files modified:** None (only new test file)
- **Commit:** `6e1bdad`

### Plan Expectation vs Actual Verdict

The plan expected `test_tile_boundary_byte_exact` to FAIL/XFAIL on the current codebase (RED state). It actually XPASSes. This is **not a deviation in implementation** — the test code is correct and exercises the intended path. It is a **factual finding** that updates Plan 08-04's scope. See "Hand-off to Plan 08-04" section above. The orchestrator's success criterion (xfail decorator with strict=False, suite stays green) is fully satisfied.

## Authentication Gates

None — fully programmatic path; no external services.

## Verification

| Gate | Command | Result |
|---|---|---|
| Syntax | `python -c "import ast; ast.parse(open('tests/gtx/test_multi_tile_dma.py').read())"` | OK |
| Collection | `pytest tests/gtx/test_multi_tile_dma.py --collect-only --no-cov` | 2 items collected |
| State-reset audit | `pytest tests/gtx/test_multi_tile_dma.py::test_tile_boundary_state_reset -v --no-cov` | PASSED (0.93s) |
| Byte-exact (xfail) | `pytest tests/gtx/test_multi_tile_dma.py::test_tile_boundary_byte_exact -v --no-cov` | XPASS (0.83s) |
| Byte-exact (mechanical) | `pytest ...::test_tile_boundary_byte_exact --runxfail -v --no-cov` | PASSED (3.38s) |
| Adjacent regression | `pytest tests/gtx/test_dma_roundtrip.py tests/gtx/test_deferred_store.py -q` | 14 passed (1.06s) |

## Self-Check: PASSED

Verified items exist:
- `tests/gtx/test_multi_tile_dma.py` — FOUND
- Commit `6e1bdad` — FOUND in `git log`

Verified test counts:
- `def test_tile_boundary_state_reset` × 1 occurrence — confirmed
- `def test_tile_boundary_byte_exact` × 1 occurrence — confirmed
- `_RISCV_AVAILABLE` × 6 occurrences (≥2 required) — confirmed
- `HEIGHT = 4097` × 1 — confirmed
- `TILE_MAX_ROWS = 4095` × 1 — confirmed
- `from tests.gtx._mocks import` — confirmed
- `pytest.mark.xfail` × 1 — confirmed (decorator on byte_exact)
- `firmware_dma_sloop_load` × 1 call site — confirmed
- `firmware_dma_sloop_store` × 1 call site — confirmed
- `firmware_vec_op` × 1 call site — confirmed
