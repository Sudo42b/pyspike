---
phase: 09-backend-migration-numpy-cupy
plan: 01a
subsystem: infra
tags: [numpy, cupy, xp-alias, backend-migration, memory, ddr, scratchpads, wave-1, pytorch-removal]

requires:
  - phase: 09-backend-migration-numpy-cupy
    provides: "Wave 0 xp alias + to_host/to_device helpers in config_params.py (plan 09-00)."
provides:
  - "unit/memory.py is torch-free; scratchpads (L0/L1/L2) + DDR_MEMORY allocate via xp.zeros"
  - "GtxMemory byte / fp16 view aliasing on the xp backend (LE byte order, numpy/cupy uniform)"
  - "DDR_MEMORY.read/write/view/raw API surface on xp (no .device, no .to(.device), no .numel())"
  - "ddr_save_to_hex / ddr_load_from_hex route through to_host / to_device at the file boundary (D-10 H/D bridge)"
  - "_DDR_DEVICE constant fully removed across riscv/gtx/ (H-5 audit clean)"
  - "D-10 VRAM-budget comment landed near DDR_MEMORY (consumer GPU <12 GB → GTX_DDR_SIZE=1G hint)"
  - "Wave 1a invariants for downstream waves: mem.lN_byte/lN_f16 return xp.ndarray"
affects: [09-01b-regs, 09-02a-ops, 09-02b-engines, 09-03-finalize]

tech-stack:
  added: []
  patterns:
    - "Module-level scratchpad allocation via xp.zeros frozen at import time (matches D-02 eager backend resolution)"
    - "Host-bytes → device staging via _np.frombuffer + to_device (file I/O is host-only by contract)"
    - "Slice copy across xp doubling-grow (`new_arr[:current_size] = self._bytes`) works on numpy and cupy uniformly"
    - "View aliasing via `.view(xp.float16)` for byte ↔ FP16 reinterpret (zero-copy, LE host)"

key-files:
  created:
    - tests/gtx/test_memory_layout.py
    - tests/gtx/test_dma_roundtrip.py
  modified:
    - src/main/python/riscv/gtx/unit/memory.py

key-decisions:
  - "Drop `device=` kwarg entirely in xp.zeros calls — xp is device-implicit; numpy=host, cupy=GPU. Aligns with Wave 0's `DEVICE` deprecated-alias contract and CONTEXT D-09/D-10."
  - "`ddr_save_to_hex` calls `to_host(...)` ONCE on the sliced window (start:end) before bytes conversion — minimises H/D copy size to the requested region, not full DDR."
  - "`ddr_load_from_hex` stages through `_np.frombuffer(bytearray(chunk))` then `to_device(...)` — numpy is mandatory at the file edge because cupy.frombuffer does not accept host bytes directly (per plan action note)."
  - "`DDR_MEMORY.read/write` API simplified: removed the cross-device `.to(.device)` branch from `write` (xp backend is uniform across DDR + scratchpads — no per-call device check needed)."
  - "Type annotations use `xp.ndarray` instead of `numpy.ndarray` — single source-of-truth that flips with xp under `GTX_USE_CUDA=1` (cupy.ndarray is a distinct class)."

patterns-established:
  - "xp-resident memory layer: All scratchpad + DDR allocations route through xp.zeros. Downstream waves must use `mem.lN_byte()` / `mem.lN_f16()` as xp.ndarray-returning APIs."
  - "to_host/to_device at file I/O boundaries: hex dump/load is host-only by contract; the bridge crosses once per file operation, not per byte access."
  - "VRAM-budget documentation idiom: D-10 source-comment near DDR_MEMORY.__init__ names the GTX_DDR_SIZE override knob explicitly (consumer-GPU operator guidance)."

requirements-completed: [BM-02]  # Wave 1a partial — BM-02 fully closed by plan 09-01b's gate run

duration: 17min
completed: 2026-05-18
---

# Phase 9 Plan 01a: Memory Layer xp Port Summary

**unit/memory.py ported from torch to xp (numpy default, cupy under `GTX_USE_CUDA=1`): scratchpads + DDR_MEMORY allocate via `xp.zeros`, `_DDR_DEVICE` constant removed, file I/O routes through `to_host`/`to_device`, D-10 VRAM-budget comment landed.**

## Performance

- **Duration:** 17min 15s
- **Started:** 2026-05-18T10:56:05Z
- **Completed:** 2026-05-18T11:13:20Z
- **Tasks:** 1 / 1 complete
- **Files modified/created:** 3 (1 source modified + 2 test files created)
- **Commits:** 2 task (RED + GREEN) + 1 metadata = 3 total

## Accomplishments

- `src/main/python/riscv/gtx/unit/memory.py` is fully torch-free. `import torch` removed; `from ..config_params import xp, to_host, to_device` added. All 11 prior `torch.*` call sites converted (module-level scratchpads, DDR allocation, DDR doubling-grow, byte/f16 view aliasing, read/write/view/raw API, file I/O bridge).
- Module-level scratchpads `_L2_GLOBAL` / `_L1_GLOBAL` / `_L0_GLOBAL` allocate via `xp.zeros(shape, dtype=xp.uint8)` with no `device=DEVICE` kwarg. Shapes preserved exactly: (4, 16MB), (4, 16, 384KB), (4, 16, 1KB).
- `_DDR_DEVICE = torch.device("cpu")` literal deleted. H-5 audit: zero `_DDR_DEVICE` references across `src/main/python/riscv/gtx/` tree (confirmed by `test_no_ddr_device_anywhere_in_gtx_package`).
- `DDR_MEMORY.__init__` allocates via `xp.zeros(size, dtype=xp.uint8)` with a D-10 inline comment naming the `GTX_DDR_SIZE=1G` override for consumer GPUs (<12 GB VRAM headroom).
- `DDR_MEMORY.ensure()` doubling-grow uses `xp.zeros` + slice copy (numpy & cupy-uniform).
- `GtxMemory` byte/f16 views use `.view(xp.float16)` for LE byte-reinterpret; aliasing preserved end-to-end (writes through byte view visible through f16 view and vice versa — confirmed by 3 paired tests).
- `ddr_save_to_hex` bridges `to_host(ddr_src[start:end])` once at the formatting edge — minimises H/D copy to the requested window.
- `ddr_load_from_hex` stages through `_np.frombuffer(bytearray(chunk))` then `to_device(...)` — file bytes flow host → device once per chunk.
- 21 new unit tests added: 15 in `test_memory_layout.py` (source-level + runtime invariants) + 6 in `test_dma_roundtrip.py` (DMA-pattern composition + cross-view aliasing). All GREEN.

## Task Commits

Each task was committed atomically following the TDD task protocol:

1. **Task 1 RED: test_memory_layout.py + test_dma_roundtrip.py (failing tests for xp port)** — `c162ace` (test)
2. **Task 1 GREEN: port unit/memory.py from torch to xp** — `eacf75e` (feat)

**Plan metadata commit (SUMMARY + STATE + ROADMAP):** recorded after self-check.

## Files Created/Modified

### Created
- `tests/gtx/test_memory_layout.py` — 15 tests covering source-level invariants (torch-free, xp imports, _DDR_DEVICE removal, VRAM-budget D-10 comment, ≥4 xp.zeros, to_host at file I/O boundary, H-5 package-wide audit) AND runtime invariants (scratchpad shapes/dtypes, L0/L1/L2 byte ↔ f16 view aliasing, DDR xp uint8 backing, DDR doubling-grow preserves prior bytes, ddr_save/load hex roundtrip).
- `tests/gtx/test_dma_roundtrip.py` — 6 tests for DDR.write/read byte-exact, DDR.view fp16 reinterpret, DDR → L1 byte transfer DMA pattern, FP16↔byte view aliasing (both directions), ensure() idempotent, clear() zeros in-place without realloc.

### Modified
- `src/main/python/riscv/gtx/unit/memory.py` — torch → xp port. Replaced `import torch` + `from ..config_params import (..., DEVICE)` with `import numpy as _np` (local) + `from ..config_params import xp, to_host, to_device`. Removed `_DDR_DEVICE` constant. Updated DDR_MEMORY (init, ensure, read, write, view, raw, clear, getsize, capacity, free) and GtxMemory (init, clear, reset_scratchpads, free, l0/l1/l2 byte/f16 view accessors, ddr_load_from_hex, ddr_save_to_hex) to xp idiom. Replaced torch-specific calls (`.numel()`, `.zero_()`, `len(tensor)`, `.device` checks, `.to(.device)`) with xp-uniform equivalents (`.size`, `[:] = 0`, `.shape[0]`).

## Decisions Made

1. **Drop `device=` kwarg in `xp.zeros` calls** — xp is device-implicit (numpy=host, cupy=GPU per Wave 0 contract). Keeping `device=DEVICE` would conflict with the deprecated string-alias contract of Wave 0 and force numpy to reject the kwarg (it doesn't accept `device=`).

2. **`ddr_save_to_hex` slices BEFORE to_host()** — `to_host(ddr_src[start:end])` ships only the requested window (typically tile-sized, < 1 MiB) across the H/D boundary rather than the full 4 GiB DDR backing. Important for cupy path performance.

3. **`ddr_load_from_hex` uses numpy explicitly for `frombuffer`** — `cupy.frombuffer` does not accept host bytes; staging through `_np.frombuffer(bytearray(chunk))` then `to_device(...)` is the only correct path. Documented in the action block of plan 09-01a.

4. **Simplified `DDR_MEMORY.write` — removed cross-device `.to()` branch** — the prior torch contract was "write accepts CPU or DEVICE input, applies one .cpu() / .to() on mismatch". Under xp, DDR + scratchpads share the same backend (D-10 unified), so per-call device checks are pure overhead. If a Wave 2 caller ever needs to write a numpy ndarray to a cupy-backed DDR, the explicit `to_device()` call belongs at the caller's DMA boundary, not buried inside `write`.

5. **Type annotations use `xp.ndarray` (not `numpy.ndarray`)** — single source-of-truth that flips automatically with xp under GTX_USE_CUDA=1. Both `numpy.ndarray` and `cupy.ndarray` are valid xp classes at import time.

## Deviations from Plan

None — plan executed exactly as written. All action-block edits applied 1:1 to memory.py at the documented sites. All acceptance criteria (grep counts, test PASS) hit on the first GREEN run.

The only adjustments after first port were docstring-level (two passes to remove residual `torch.device` and `_DDR_DEVICE` mentions from comments) — these are not deviations, they are part of the H-5 clean-cut work specified by the plan.

## Issues Encountered

1. **Mid-execution `git stash` reverted my staged edits.** Ran `git stash` to inspect the prior commit state during a diagnostic; the subsequent `git stash pop` silently failed (untracked-file conflicts with the large `test/**/*.nohdr.txt` data tree). The staged port reverted to the original torch state. Resolved by `git checkout stash@{0} -- src/main/python/riscv/gtx/unit/memory.py` to restore just the one file, then `git stash drop`. All tests still GREEN after restore. No work was lost.

2. **`test_deferred_store.py` pre-existing failures (11/11).** `ModuleNotFoundError: 'riscv.gtx.dma_engine'` — the test imports from the old path; the actual module is at `riscv.gtx.unit.context.dma_engine`. Pre-existing condition unrelated to Wave 1a. Documented in deferred-items below.

## Known Stubs

None. `self.spr: dict[int, int] = {}` in `GtxMemory.__init__` is the legacy SPR placeholder dict (pre-existing, predates Phase 9), not a Wave 1a stub. SPR storage proper is owned by `RegisterFile` (plan 09-01b).

## User Setup Required

None — no external service configuration required for Wave 1a.

## Next Phase Readiness

**Plan 09-01b (Wave 1b: register_file + npu + Wave 1 gate) entry condition: MET.**

- ✅ `unit/memory.py` is xp-resident and torch-free. `mem.l{0,1,2}_byte()` and `mem.l{0,1,2}_f16()` return xp.ndarray, matching the Wave 1a invariants declared in 09-01a's `<interfaces>` block.
- ✅ `_DDR_DEVICE` literal fully removed (H-5 clean cut complete in Wave 1a; no carry-forward to Wave 1b).
- ✅ DDR file I/O routes through to_host/to_device bridge — Wave 2 ops can rely on this contract.
- ✅ D-10 VRAM-budget comment landed in source. Wave 1 gate document (owned by plan 09-01b Task 4) will record the actual ABS walltime measurement.

**Wave 1a transit-state note (carry-forward to 09-01b):**

Downstream callers in `npu.py`, `unit/register_file.py`, `unit/context/dma_engine.py`, `tloop_buffer.py`, `unit/ins/ops/*.py`, and `_verify.py` still use `torch.*` and will fail at runtime when they receive numpy arrays from `mem.lN_byte()`. This is expected per CONTEXT D-06 ("Dual-import allowed but minimised — wave intermediate state").

The full 6-op smoke + tile-2 unit test gate runs at the END of plan 09-01b (Task 4 Wave-1 gate). Wave 1a's own verification was unit-level (`test_memory_layout.py` + `test_dma_roundtrip.py`), and both files passed 21/21.

## Deferred Issues (out of plan 09-01a scope)

1. **`test_deferred_store.py` import path broken** — `from riscv.gtx.dma_engine import firmware_dma_sloop_store` should be `from riscv.gtx.unit.context.dma_engine import ...`. Pre-existing failure (11/11 tests). Not introduced by Wave 1a. Track for Phase 9 wrap-up or v1.2 test-infra cleanup.

2. **`uv.lock` modified by parallel uv run invocation** — cuda-bindings 12.9.6 added by some background `uv run` (possibly during a benchmark or another agent). Not committed as part of Wave 1a. Leave untracked for now; Phase 9 finalize will reconcile.

## Self-Check: PASSED

Verified files exist:
- /mnt/e/14_NIGHTLY/pyspike/tests/gtx/test_memory_layout.py
- /mnt/e/14_NIGHTLY/pyspike/tests/gtx/test_dma_roundtrip.py
- /mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/memory.py (modified)
- /mnt/e/14_NIGHTLY/pyspike/.planning/phases/09-backend-migration-numpy-cupy/09-01a-SUMMARY.md (this file)

Verified commits in `git log --all`:
- c162ace (Task 1 RED — failing tests for xp port)
- eacf75e (Task 1 GREEN — port memory.py from torch to xp)

Verified acceptance criteria (grep):
- 0 `import torch | torch.` refs in memory.py
- 1 `from ..config_params import xp` in memory.py
- 0 `_DDR_DEVICE | torch.device` refs in memory.py
- 0 `_DDR_DEVICE` refs across `src/main/python/riscv/gtx/`
- 5 `xp.zeros` calls in memory.py (≥ 4 required)
- 6 `to_host` refs in memory.py (≥ 1 required)
- 5 `D-10` comments in memory.py (≥ 1 required)
- 21/21 tests GREEN under `uv run pytest tests/gtx/test_memory_layout.py tests/gtx/test_dma_roundtrip.py --no-cov`

---
*Phase: 09-backend-migration-numpy-cupy*
*Plan: 01a-memory*
*Completed: 2026-05-18*
