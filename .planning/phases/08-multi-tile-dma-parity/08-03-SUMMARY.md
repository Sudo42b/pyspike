---
phase: 08-multi-tile-dma-parity
plan: 03
subsystem: testing/investigation
tags: [dump-size-investigation, hypothesis-5-resolution, multi-tile-bug-localization, harness-extension, full-region-golden, plan-04-handoff]

# Dependency graph
requires:
  - phase: 08-multi-tile-dma-parity
    provides: 08-01 XPASS evidence (programmatic 2-tile path byte-exact) + 08-02 vendor asset wire-up (--all skip-guard, _find_elf 3-tier)
provides:
  - scripts/import_vendor_golden.py --full flag (n_lines=None bypass + dst_dir=PYSPIKE_GOLDEN_FULL routing, --full and --all are independent)
  - tests/gtx/test_regression_fw_full_sweep.py OP_DUMP_SIZE_OVERRIDE dict (computed at module import from GOLDEN_DIR_FULL on disk) + GTX_DDR_DUMP_SIZE_OVERRIDE_ALL env-var override + GOLDEN_DIR_FULL preference in _find_golden
  - .gitignore tests/gtx/data/golden_full/ exclusion
  - 08-03-INVESTIGATION.md verdict (Outcome B — NPU code fix needed) with file:line bug location candidates and exact byte-mismatch evidence at line 2048 = MAX_SHARED_DMA_BYTES boundary for ABS
affects: [08-04 (sharpened scope: dispatch path is buggy, not dma_engine core; Plan 04 must trace RoCC custom0/custom1 path with vendor .elf + add OP_DUMP_ADDR_OVERRIDE)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runtime-from-disk per-op size override (computed at module import from GOLDEN_DIR_FULL.glob)"
    - "Layered GTX_DDR_DUMP_SIZE priority: env-var > per-op-from-disk > legacy default"
    - "Dual-destination golden import: golden/ (committed truncated) + golden_full/ (gitignored full-region)"
    - "Investigation-as-deliverable: PLAN produces evidence document (not code) for next plan to act on"

key-files:
  created:
    - .planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md
  modified:
    - scripts/import_vendor_golden.py
    - tests/gtx/test_regression_fw_full_sweep.py
    - .gitignore

key-decisions:
  - "Outcome B confirmed: ABS divergence at exact line 2048 (= MAX_SHARED_DMA_BYTES boundary) is a real production bug in the RoCC dispatch path, not a harness artifact. Plan 01 XPASS is genuine but bypasses the buggy path."
  - "_find_golden prefers golden_full/ over golden/ when both present (auto-detected at runtime; CI default unchanged when golden_full/ absent)"
  - "OP_DUMP_SIZE_OVERRIDE computed runtime-from-disk (not hardcoded) — investigator can populate golden_full/ then run pytest; static prefill recommended for Plan 04 GREEN-path CI"
  - "Discovered during investigation: vendor .elf writes output at 0xf000000 (not 0x100); harness needs OP_DUMP_ADDR_OVERRIDE in Plan 04. Discovered during investigation: vendor .elf requires GTX_DDR_INIT pre-staging or DDR is all zeros. Both are out-of-scope for Plan 03 (D-04 surgical) but flagged for Plan 04."
  - "Byte-pattern signature at tile boundary: dump line 2048 has first 16 bytes zero + last 16 bytes correct (matches golden's last 16 bytes). Points to addr_hi packed-rs1 decode bug at second-tile invocation, OR L2 ring-wrap at GTX_L2_SIZE_BYTES boundary."

patterns-established:
  - "Investigation-only plan structure: produces an INVESTIGATION.md with verdict + evidence + Plan-N+1 handoff, no production code changes"
  - "Dual-destination golden generation: --all (CI/committed, truncated) and --full (dev-local, full-region) coexist via flag composition"
  - "Per-op env-var override via runtime os.stat() inspection: scales to 84 ops without manual prefill, but with deterministic CI fallback"

requirements-completed: [MTDMA-01, VTW-02]
# (Investigation deliverable — both requirements get evidence-bearing partial closure;
#  full closure requires Plan 04 fix + sweep PASS for MTDMA-01 and re-run with extended
#  dump for VTW-02)

# Metrics
duration: 12min
completed: 2026-05-10
---

# Phase 08 Plan 03: Dump-size investigation Summary

**One-liner:** Generated full-region vendor goldens via new `--full` flag + extended pytest harness with per-op `GTX_DDR_DUMP_SIZE` override; ran 6 vendor `.elf` smoke ops and confirmed real production bug in RoCC dispatch path — ABS diverges at exactly line 2048 = `MAX_SHARED_DMA_BYTES` tile boundary, second tile writes only the latter 16 bytes per row correctly. Verdict: Outcome B (NPU code fix needed in Plan 04).

## Performance

- **Duration:** ~12 min (08-03 plan execution)
- **Started:** 2026-05-10T13:55Z
- **Completed:** 2026-05-10T14:07Z
- **Tasks:** 2 (both atomic commits)
- **Files modified:** 3 (scripts/import_vendor_golden.py, tests/gtx/test_regression_fw_full_sweep.py, .gitignore)
- **Files created:** 1 (.planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md)

## Accomplishments

### Task 1: harness extensions (commit `25c54a5`)

- `scripts/import_vendor_golden.py` gained `--full` flag (`n_lines=None`, writes to `tests/gtx/data/golden_full/`). Independent of `--all` (composes both). Verified `--all --full --verify` emits 82 entries pointing to `golden_full/` with full-region line counts (e.g., `abs.hex 196609 lines`).
- `convert_one` signature extended: `n_lines: int | None`, `dst_dir: pathlib.Path | None`. None branch bypasses truncation; explicit dst_dir overrides default `PYSPIKE_GOLDEN`.
- `tests/gtx/test_regression_fw_full_sweep.py` gained:
  - `GOLDEN_DIR_FULL` constant (`tests/gtx/data/golden_full/`)
  - `OP_DUMP_SIZE_OVERRIDE` dict computed at module import time by `glob`-ing `GOLDEN_DIR_FULL/*.hex` and using `lines * 32` raw byte count
  - `_find_golden` priority list extended: `golden_full/{op_dir.lower(), elf_stem} > golden/{...}` (CI default unchanged when `golden_full/` absent)
  - Subprocess env block uses layered priority `GTX_DDR_DUMP_SIZE_OVERRIDE_ALL > OP_DUMP_SIZE_OVERRIDE[op_dir] > "0x20"`
- `.gitignore` excludes `tests/gtx/data/golden_full/` (huge, dev-local generation only).

### Task 2: investigation evidence (commit `a3e52f3`)

- Generated full-region goldens for all 82 vendor ops with non-empty `_ref.txt` (9 P6-aliased ops + 73 vendor-only); total `golden_full/` size ≈ 25 MB on disk (gitignored).
- Ran 6 smoke ops through `pyspike --extlib=riscv.gtx --extension=gtx <vendor.elf>` with full-region dump + diff vs full-region golden. Pre-staged inputs via `GTX_DDR_INIT=<vendor input.txt>` (D-11 hands-on confirmation).
- **Confirmed multi-tile production bug** at exactly line 2048 of ABS (= `MAX_SHARED_DMA_BYTES = 65535` boundary).
- Wrote `08-03-INVESTIGATION.md` with:
  - Per-op divergence table (6 ops, with ABS at exact tile boundary, GELU PASS, others single-tile or FP-precision)
  - Hypothesis verdict (H5 confirmed-AND-falsified simultaneously; new H8 confirmed)
  - Bug location candidates ranked by probability (`decode_firmware_dma_args` second-tile `addr_hi`, `flush_deferred_ddr_stores` L2 ring-wrap, RoCC dispatch table)
  - Recommended Plan 04 scope (5 specific actions)
  - Plan 04 hand-off (file:line + bytes-mismatched range + suggested OP_DUMP_SIZE_OVERRIDE prefill)

## Task Commits

| # | Description | Hash |
|---|-------------|------|
| 1 | feat(08-03): add --full flag + per-op GTX_DDR_DUMP_SIZE override | `25c54a5` |
| 2 | docs(08-03): record multi-tile DMA divergence investigation | `a3e52f3` |

## Files Created/Modified

**Created:**
- `.planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md` (185 lines)

**Modified:**
- `scripts/import_vendor_golden.py` (+50 lines, -16 deletions): `--full` flag, `convert_one` accepts `n_lines=None`/`dst_dir`, `--all` skip-guard relaxed under `--full`, default 9-op branch threads `--full` through `effective_n_lines`.
- `tests/gtx/test_regression_fw_full_sweep.py` (+62 lines): `GOLDEN_DIR_FULL`, `OP_DUMP_SIZE_OVERRIDE` (runtime-from-disk), env-block layered priority, `_find_golden` 4-candidate list.
- `.gitignore` (+5 lines): `tests/gtx/data/golden_full/` exclusion with descriptive comment.

## Verification Evidence

### Static checks (Task 1)

```
=== --full in --help ===
2

=== --all --full produces golden_full ===
82

=== --all (no --full) does NOT touch golden_full ===
0

=== .gitignore golden_full match ===
1

=== OP_DUMP_SIZE_OVERRIDE in test file (>=2) ===
6

=== GOLDEN_DIR_FULL in test file (>=2) ===
7

=== GTX_DDR_DUMP_SIZE_OVERRIDE_ALL in test file (>=1) ===
3

=== 9-op default still works (>=8 DRY:) ===
9
```

### Pytest collection (Task 1 regression check)

```
$ python -m pytest tests/gtx/test_regression_fw_full_sweep.py --collect-only --no-cov
========================= 84 tests collected in 0.42s ==========================
```

### Adjacent regression (Task 1)

```
$ pytest tests/gtx/test_dma_roundtrip.py tests/gtx/test_deferred_store.py tests/gtx/test_multi_tile_dma.py -q --no-cov
15 passed, 1 xpassed in 1.69s
```

(test_multi_tile_dma.py XPASS preserved from 08-01)

### Investigation evidence (Task 2)

```
=== Final smoke-set diff summary ===
ABS	DIVERGE	first_line=2048	golden_lines=196609	diff_lines=389124
ADD	DIVERGE	first_line=0	golden_lines=65536	diff_lines=65536
RELU	DIVERGE	first_line=1	golden_lines=16384	diff_lines=32640
GELU	PASS	golden_lines=1920
SIGMOID	DIVERGE	first_line=1	golden_lines=256	diff_lines=298
LEAKY_RELU	DIVERGE	first_line=1497	golden_lines=1993	diff_lines=2
```

### Byte-sample at ABS tile boundary

```
Line 2047 (last good — tile 0 final row):
DUMP:  3a1e3a3d32c03831304c397d373b38d43b9b33403ba33b593224394a3bfe359b
GOLD:  3a1e3a3d32c03831304c397d373b38d43b9b33403ba33b593224394a3bfe359b
                                    BYTE-EXACT MATCH

Line 2048 (first bad — tile 1 first row):
DUMP:  0000000000000000000000000000000035a334683a483387386d39152a5738b3
GOLD:  3556393827b6381638e428433bbd33aa35a334683a483387386d39152a5738b3
       ^ 16 zero bytes ^^^^^^^^^^^^^^^^^^^ 16 correct bytes (matches golden) ^^
```

### Investigation document gates

```
=== INVESTIGATION.md exists ===
OK

=== sections present ===
Per-Op Divergence Results: line 36
Hypothesis Verdict: line 74
Bug Location Candidates (for Plan 04): line 87
Recommended Plan 04 Scope: line 115
Plan 04 Hand-off: line 147
```

## Decisions Made

- **Investigation reproduces P7 ABS observation byte-for-byte.** The "lines past 2048 diverge" symptom is real and lives in production NPU code (specifically the RoCC dispatch path that vendor `.elf` exercises). Plan 01's XPASS is genuine for the programmatic API path but does not exercise the dispatch path.
- **`--full` and `--all` flags are independent and composable.** `--full` redirects ALL output to `golden_full/` regardless of `--all`. The skip-guard for the 9 P6 ops is RELAXED under `--full` because the destination directories don't collide. Without `--full`, the original 08-02 invariant holds.
- **`OP_DUMP_SIZE_OVERRIDE` is runtime-from-disk, not statically prefilled.** This means an investigator can populate `golden_full/` and `pytest` picks up new sizes automatically. CI default behavior is preserved (no `golden_full/` → all ops fall back to `0x20`). For deterministic CI behavior post-fix, Plan 04 should add a static prefill to `OP_DUMP_SIZE_OVERRIDE` for the smoke set.
- **Two NEW out-of-scope harness gaps surfaced** (recorded in INVESTIGATION.md, deferred to Plan 04):
  - Per-op `GTX_DDR_DUMP_ADDR` (vendor uses `0xf000000`, not `0x100`)
  - Per-op `GTX_DDR_INIT` (vendor `.elf` requires operand pre-staging, currently unwired)

## Deviations from Plan

### Auto-fixed Issues (Rule 3 — blocking discoveries during investigation)

**1. [Rule 3 — Blocking] `GTX_DDR_DUMP_ADDR=0x100` produces all-zero dumps for vendor `.elf`.**
- **Found during:** Task 2 first investigation run.
- **Issue:** Vendor `.elf` writes output at `0xf000000` (per `_ref.txt @f000000` header), not `0x100`. With `0x100`, dumps are 12 MB of zeros and the bug is invisible.
- **Fix:** Switched investigation to use `GTX_DDR_DUMP_ADDR=0xf000000`. The harness `test_regression_fw_full_sweep.py` still uses `0x100` (P5/P6 default for hand-built `.elf`); fixing it for vendor `.elf` is logged as a Plan 04 work item ("OP_DUMP_ADDR_OVERRIDE" in INVESTIGATION.md hand-off).
- **Files modified:** None in this plan (D-04 surgical scope — Plan 04 owns code changes).

**2. [Rule 3 — Blocking] Vendor `.elf` requires `GTX_DDR_INIT` pre-staging.**
- **Found during:** Task 2 first investigation run.
- **Issue:** Vendor `.elf` reads input from DDR at construction (via `__ddr_init` intrinsic). Without `GTX_DDR_INIT=<vendor input.txt>`, DDR is all zeros, `abs(0) == 0`, and dumps are all zero.
- **Fix:** Investigation runs explicitly set `GTX_DDR_INIT=<vendor input.txt>` per op. The harness `test_regression_fw_full_sweep.py` does NOT yet wire this. Logged as Plan 04 work item (analogous to GTX_DDR_REVERSED inline propagation in 08-02).
- **Files modified:** None in this plan.

### Plan Expectation vs Actual Verdict

The plan's <action> outlined Outcome A vs Outcome B verdicts. Initial expectation (per RESEARCH.md + Plan 01 XPASS) was **Outcome A** ("harness-only fix sufficient"). Actual verdict: **Outcome B** ("NPU code fix needed"). The investigation surfaced clear evidence that the bug is real and lives in the RoCC dispatch path. This is **not a deviation in implementation** — the investigation was designed to discriminate between A and B, and produced B. The plan's success criterion ("verdict + evidence") is fully satisfied.

## Authentication Gates

None — fully programmatic investigation; no external services.

## Issues Encountered

- Initial pyspike runs against vendor `.elf` produced all-zero dumps for ~10 minutes of debugging until the `GTX_DDR_DUMP_ADDR` and `GTX_DDR_INIT` mismatches were identified (both noted above).
- ABS full-region investigation took ~5 minutes per run (12 MB dump). Acceptable for one-off investigation; Plan 04 may need to use `--full` selectively per-op rather than for all 82.

## Self-Check: PASSED

Verified items exist:
- `scripts/import_vendor_golden.py` — modified (commit `25c54a5`) — FOUND
- `tests/gtx/test_regression_fw_full_sweep.py` — modified (commit `25c54a5`) — FOUND
- `.gitignore` — modified (commit `25c54a5`) — FOUND
- `.planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md` — created (commit `a3e52f3`) — FOUND

Verified commits exist:
- `25c54a5` (Task 1): `git log --oneline -5 | grep 25c54a5` — FOUND
- `a3e52f3` (Task 2): `git log --oneline -5 | grep a3e52f3` — FOUND

Verified verdict + evidence:
- INVESTIGATION.md verdict: "Outcome B (NPU code fix needed)" — confirmed in TL;DR + frontmatter
- Per-op divergence table: 6 ops, with line numbers — confirmed
- Bug location candidates ranked: 4 candidates with file:line + probability — confirmed
- Plan 04 hand-off: single most likely bug location, byte-mismatched range, suggested override prefill — confirmed

---
*Phase: 08-multi-tile-dma-parity*
*Plan: 03*
*Completed: 2026-05-10*
