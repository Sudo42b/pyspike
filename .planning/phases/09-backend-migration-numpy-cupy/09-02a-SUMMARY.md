---
phase: 09-backend-migration-numpy-cupy
plan: 02a
subsystem: ops
tags: [numpy, cupy, xp-alias, backend-migration, ops, mm, vec, act, spr, csr, fp8-lut, wave-2, pytorch-removal, shim-removal]

requires:
  - phase: 09-backend-migration-numpy-cupy
    provides: "Wave 0 xp alias + helpers (plan 09-00); Wave 1a memory.py xp port (plan 09-01a); Wave 1b register_file.py + npu.py xp port + bridge shim (plan 09-01b)."
provides:
  - "unit/ins/ops/spr.py torch-free; CPSVR/MVSVR ported to xp.tile + xp slice-assign (replaces torch.Tensor.repeat / .clone / .zero_)"
  - "unit/ins/ops/mm.py torch-free; gemm_core + 5 MM variants on xp.matmul / xp.dot / xp.sum (BLAS-equivalent semantics preserved)"
  - "unit/ins/ops/vec.py torch-free; 7 vector kernels + _apply_unary + sub-dispatchers on xp (cumsum axis=, clamp→clip)"
  - "unit/ins/ops/act.py torch-free; 7 activations + 2 pool + 9 cvt + FP8 LUTs on xp; FP8 path is single deterministic LUT-only (Option-B, H-1)"
  - "unit/csr/register.py docstring updated for xp.ndarray storage contract"
  - "3 f16 WAVE-1-SHIM accessor sites removed from unit/memory.py (l0_f16, l1_f16, l2_f16)"
  - "Wave 2a gate doc (09-02a-WAVE-GATE.md) — smoke GREEN + ABS walltime back in D-08 budget"

affects: [09-02b-engines, 09-03-finalize]

tech-stack:
  added: []
  patterns:
    - "Shim bypass for in-Wave callers: ported op files read raw module-level storage (npu.mem.l[012][nest,spu]) instead of the shimmed accessor (mem.l*_byte / mem.l*_f16). Same pattern Wave 1b's flush_deferred_ddr_stores adopted for internal code paths."
    - "Per-wave shim sunset: after the last torch consumer of a given accessor is ported, that accessor's shim is removed in the same plan. Wave 2a removed 3 f16 shims (l0_f16, l1_f16, l2_f16) because zero non-memory.py consumers remained."
    - "FP8 LUT-only path (Option-B, locked default per 09-SCOPE-DECISION.md / H-1): single deterministic code path; FP16→FP8 via .view(xp.uint16) + FP16_TO_FP8_LUT[u16]; FP8→FP16 via FP8_TO_FP16_LUT[u8]. No conditional branches, no Option-A/C escape hatches."
    - "Explicit activation formulas for numpy parity: relu = xp.maximum(x, 0); sigmoid = 1/(1+exp(-x)); softmax = stable shift form. GELU uses tanh-approx (matches torch.gelu default + vendor C++)."
    - "xp.tile vs torch.Tensor.repeat semantic divergence: torch.Tensor.repeat tiles, xp.ndarray.repeat repeats per-element. spr.py CPSVR's byte-stream fill required xp.tile (not xp.repeat) to preserve vendor C++ memcpy semantics."

key-files:
  created:
    - .planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md
    - .planning/phases/09-backend-migration-numpy-cupy/09-02a-SUMMARY.md
  modified:
    - src/main/python/riscv/gtx/unit/ins/ops/spr.py
    - src/main/python/riscv/gtx/unit/ins/ops/mm.py
    - src/main/python/riscv/gtx/unit/ins/ops/vec.py
    - src/main/python/riscv/gtx/unit/ins/ops/act.py
    - src/main/python/riscv/gtx/unit/csr/register.py
    - src/main/python/riscv/gtx/unit/memory.py
    - tests/gtx/test_memory_torch_shim.py

key-decisions:
  - "spr.py xp import REQUIRED (deviates from plan 'no xp' assertion). CPSVR/MVSVR exercise array operations on L0 byte storage (torch.Tensor.repeat tile semantics, torch.Tensor.clone, torch.Tensor.zero_). Adding xp import is mandatory to avoid runtime AttributeError once the l0_byte shim is removed. Plan's H-3 'spr.py uses Python ints' was over-strong; only RDSPR/WRSPR/OPSET use pure Python ints — CPSVR/MVSVR exercise byte-stream array primitives."
  - "Shim bypass pattern: ported op files access raw xp storage (npu.mem.l[012][nest,spu] / .view(xp.float16)) instead of the shimmed accessors (mem.l*_byte / mem.l*_f16). Reasoning: the shim's purpose is to keep UN-PORTED torch consumers working — once a file is ported, it should read xp directly. Avoids the dead torch.from_numpy round-trip on the hot path (recoverable ~14s in ABS walltime)."
  - "FP8 LUT-only path is the SINGLE deterministic implementation (H-1). No conditional branches between Option-A (ml_dtypes), Option-B (LUT), Option-C (descope). Plan 09-SCOPE-DECISION.md locks Option-B as the default and forbids the alternative-strategy mention in any code comment that grep would flag as a banned ref."
  - "Pool kernels use reshape + axis-reduction (xp.max / xp.mean over (N, K) windows) instead of torch.{max,avg}_pool1d. count_include_pad=False is moot under stride==kernel_size (no implicit padding)."
  - "GELU uses approximate tanh formula (not erf-based). Matches torch.gelu default approx path + vendor C++ tanh-approx per 09-RESEARCH. Avoids scipy dependency for erf (D-17 wheel size constraint)."

patterns-established:
  - "Ported-file shim-bypass: ops/*.py read raw xp storage; shim remains alive only for un-ported callers."
  - "Per-plan shim removal: Wave 2a's plan-end task removes the shim sites whose torch consumers were all ported in this plan. Mechanical audit: grep all callers of the shimmed accessor across `src/main/python/riscv/gtx/`; if zero non-memory.py consumers remain, remove the shim and update its corresponding test."
  - "FP8 LUT-only code path: fp16_to_fp8_e4m3(arr) = FP16_TO_FP8_LUT[arr.view(xp.uint16)]; fp8_e4m3_to_fp16(arr) = FP8_TO_FP16_LUT[arr]. uint8-indexed LUT works identically on numpy + cupy backends. Zero new runtime deps."

requirements-completed: [BM-03]

duration: 33min
completed: 2026-05-18
---

# Phase 9 Plan 02a: Op-Handler xp Port + f16 Shim Removal Summary

**Wave 2a port complete — 4 op-handler modules (spr.py, mm.py, vec.py, act.py) + csr/register.py docstring + 3 f16 shim removals. FP8 LUT-only Option-B locked. Smoke gate GREEN with ABS strict walltime back inside D-08 85-105s budget (96.68s, 13% improvement vs Wave 1).**

## Performance

- **Duration:** 33min 27s
- **Started:** 2026-05-18T13:25:15Z
- **Completed:** 2026-05-18T13:58:42Z
- **Tasks:** 6 / 6 complete
- **Files modified/created:** 9 (5 source modified + 1 test modified + 2 docs created + memory.py modified)
- **Commits:** 7 (6 task + 1 shim removal). Plan metadata commit will follow self-check.

## Accomplishments

### Op-handler port (Tasks 1–4)

| File | torch refs (before → after) | xp import | Notes |
|------|-----------------------------|-----------|-------|
| `spr.py` | 1 → 0 | ADDED | CPSVR/MVSVR ported with xp.tile + slice-assign + .copy() |
| `mm.py` | 44 → 0 | ADDED | gemm_core + 5 MM variants on xp.matmul (BLAS-equivalent); 10 handler entries unchanged |
| `vec.py` | 52 → 0 | ADDED | 7 kernels + _apply_unary + 3 sub-dispatchers ported; cumsum axis=; clamp→clip; arange w/o DEVICE |
| `act.py` | 81 → 0 | ADDED | 7 activations + 2 pool + 9 cvt + FP8 LUTs; FP8 Option-B locked single-path |
| `csr/register.py` | 1 → 0 | n/a (docstring only) | "torch.Tensor" → "xp.ndarray (numpy default, cupy on GTX_USE_CUDA=1)" |

Total torch reduction: **179 → 0** across the 5 modified files.

### Shim removal (Task 5)

| Shim site | Status | Reason |
|-----------|--------|--------|
| `GtxMemory.l0_f16` | **REMOVED** | Zero remaining consumers (defensive shim cleared) |
| `GtxMemory.l1_f16` | **REMOVED** | act.py:312/433 + vec.py:124 ported; no other callers |
| `GtxMemory.l2_f16` | **REMOVED** | Zero remaining consumers (defensive shim cleared) |
| `GtxMemory.l0_byte` | KEPT | dma_engine.py:155/179 still consume (Wave 5) |
| `GtxMemory.l1_byte` | KEPT | dma_engine.py + tloop_buffer.py:483 still consume (Wave 5/6) |
| `GtxMemory.l2_byte` | KEPT | tloop_buffer.py:459/467/477/485 still consume (Wave 6) |
| `DDR_MEMORY.read` | KEPT | dma_engine.py:266/345/534/647/664 still consume (Wave 5) |

### Wave 2a gate (Task 6)

- Smoke set: 4 PASS + 1 SKIP per Wave 0 convention (same as Wave 1 baseline).
- Tile-2: N/A (file removed pre-Phase-9; same as Wave 0/1).
- ABS strict walltime: **96.68s** — back inside D-08 85-105s budget. Wave 1 was 110.97s (6% over); Wave 2a is 13% faster. The reduction comes from removing the per-call torch.from_numpy + tensor.view overhead on the hot vec.py + act.py + spr.py paths (each ABS tile hits these thousands of times across 96 tiles).
- 73 / 73 baseline unit tests GREEN (up from 69 — net +4 from shim-test re-flip to xp.ndarray contract).

## Task Commits

Each task committed atomically per the GSD task protocol:

1. **Task 1: spr.py port** — `de291ff` (feat)
2. **Task 2: mm.py port** — `cfc2677` (feat)
3. **Task 3: vec.py port** — `d62ba27` (feat)
4. **Task 4: act.py port (FP8 LUT-only)** — `6b2e3c1` (feat)
5. **Task 5: csr/register.py docstring** — `020ebb9` (docs)
6. **Shim removal: l0/l1/l2_f16** — `8b35f7c` (refactor)
7. **Task 6: Wave 2a gate doc** — `21feb47` (docs)

**Plan metadata commit (this SUMMARY + STATE + ROADMAP):** recorded after self-check.

## Files Created/Modified

### Created

- `.planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md` — Wave 2a gate results + per-shim removal table + sign-off + deferred-items.
- `.planning/phases/09-backend-migration-numpy-cupy/09-02a-SUMMARY.md` — This document.

### Modified (source)

- `src/main/python/riscv/gtx/unit/ins/ops/spr.py` — CPSVR/MVSVR ported; xp import added for byte-stream array primitives.
- `src/main/python/riscv/gtx/unit/ins/ops/mm.py` — gemm_core + 5 variant executors + 4 helpers; .matmul → xp.matmul; FP32-internal-accumulate preserved.
- `src/main/python/riscv/gtx/unit/ins/ops/vec.py` — 7 kernels + _apply_unary + 3 sub-dispatchers; cumsum axis=, clamp→clip, arange w/o DEVICE.
- `src/main/python/riscv/gtx/unit/ins/ops/act.py` — 7 activations + 2 pool + 9 cvt + FP8 LUTs; FP8 Option-B locked.
- `src/main/python/riscv/gtx/unit/csr/register.py` — docstring xp.ndarray.
- `src/main/python/riscv/gtx/unit/memory.py` — 3 f16 shim sites removed; module docstring updated.

### Modified (tests)

- `tests/gtx/test_memory_torch_shim.py` — 3 f16-accessor return-type tests flipped from `torch.Tensor` to `np.ndarray`; marker count threshold lowered from 7 → 4 to reflect Wave 2a's f16-shim removal.

## Decisions Made

See `key-decisions` frontmatter for the canonical list. Substantive decisions:

1. **spr.py needs xp import (Rule 3 deviation).** Plan said "1 torch ref, import only" and "uses Python ints". Actual: CPSVR (xp.tile for byte-stream replication) + MVSVR (slice-assign for zero, .copy() for clone) both exercise array operations. Adding `from ....config_params import xp` was mandatory.

2. **Shim bypass over shim consumption.** Ported op files read raw `npu.mem.l[012][nest,spu]` storage instead of the shimmed accessor (`mem.l*_byte` / `mem.l*_f16`). This is the same pattern Wave 1b's `flush_deferred_ddr_stores` adopted for internal code paths. The shim's purpose is to keep UN-PORTED torch consumers working — once a file is ported, it should read xp directly.

3. **FP8 path single-source.** Per 09-SCOPE-DECISION.md / H-1, the FP8 conversion code has ONE deterministic implementation: LUT-indexed (Option-B). No Option-A (`ml_dtypes`) or Option-C (`NotImplementedError`) branches; no even-mention-in-comments of those alternatives that would trip the banned-ref grep. The comment block in act.py was rewritten to avoid the literal `float8_e4m3fn` / `ml_dtypes` / `NotImplementedError` tokens.

4. **GELU = tanh-approx, not erf.** Matches torch.gelu default + vendor C++ per 09-RESEARCH. Avoids scipy.special.erf dependency (D-17 wheel-size constraint).

5. **Pool via reshape + reduce.** numpy has no pool1d primitive; reshape to (N, kernel_size) windows + xp.max/xp.mean over axis=1 is the idiomatic equivalent. count_include_pad=False is moot under stride==kernel_size (no implicit padding).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] spr.py xp import required (plan asserted xp-free)**
- **Found during:** Task 1 — CPSVR uses `l0[base].repeat(8)` (byte tile) and MVSVR uses `.clone()` / `.zero_()` (torch methods).
- **Issue:** Plan's H-3 "spr.py uses Python ints" was over-strong. xp.tile is needed for byte-stream replication (xp.ndarray.repeat would tile per-element instead of byte-wise, producing the wrong SVR fill pattern). xp slice-assign + .copy() replace torch's in-place .zero_() / .clone().
- **Fix:** Add `from ....config_params import xp` to spr.py; route CPSVR through xp.tile and MVSVR through xp slice-assign + .copy(). Plan's "no xp import" rule honored for the pure-Python-int arithmetic parts (RDSPR/WRSPR/OPSET) but relaxed for CPSVR/MVSVR.
- **Files modified:** `src/main/python/riscv/gtx/unit/ins/ops/spr.py`
- **Commit:** `de291ff`

**2. [Rule 3 - Blocking] All op files need shim bypass to keep xp end-to-end**
- **Found during:** Task 2 (mm.py) and subsequent tasks — the ports' xp-native operations would re-hit the shim layer and roundtrip through torch.from_numpy + back unless the call sites read raw `npu.mem.l[012][nest,spu]` storage directly.
- **Issue:** Plan acceptance criteria require torch-free op files. If the op file calls `npu.mem.l1_byte()` (shimmed → torch.Tensor return), the immediate `.view(xp.float16)` would fail with `TypeError: argument of type 'numpy.dtype[float16]' is not iterable` because torch.Tensor.view() expects a torch dtype.
- **Fix:** Replace `mem.l[01]_byte(nest, spu)` with `mem.l[01][nest, spu]` (raw module-level storage access). The shim's purpose is for un-ported torch consumers; ports of those consumers should access raw xp directly. Applied to all 4 op files.
- **Files modified:** `src/main/python/riscv/gtx/unit/ins/ops/{spr,mm,vec,act}.py`
- **Verification:** ABS strict regression PASSES at 96.68s (back in D-08 budget); 73/73 unit tests GREEN.

**3. [Rule 3 - Blocking] Banned-ref tokens in act.py comments tripped acceptance grep**
- **Found during:** Task 4 acceptance verification — `grep -c 'float8_e4m3fn|ml_dtypes|NotImplementedError' act.py` returned 2 (from a comment explaining what's NOT used).
- **Issue:** Plan's H-1 acceptance criteria require zero references to the rejected FP8 strategies (Option-A / Option-C) anywhere in act.py. The comment described the rejection, which still tripped the literal grep.
- **Fix:** Rewrote the comment block to describe the locked Option-B strategy without naming the rejected alternatives by their banned tokens.
- **Files modified:** `src/main/python/riscv/gtx/unit/ins/ops/act.py` (lines 121–127 comment block).
- **Verification:** `grep -c 'float8_e4m3fn|ml_dtypes|NotImplementedError' act.py` returns 0.

---

**Total deviations:** 3 auto-fixed (3 Rule 3 — Blocking).
**Impact on plan:** All 3 deviations necessary to deliver Wave 2a's bit-exact + torch-free contract. No scope creep — each deviation surgical to the plan's stated acceptance criteria.

## Issues Encountered

1. **Plan H-3 over-strong assertion.** "spr.py uses Python ints (no xp)" was true for the four SPR routing handlers (RDSPR/WRSPR/OPSET) but false for CPSVR/MVSVR which exercise byte-stream array operations. The H-3 assertion should be relaxed in plan templates: "ops that don't operate on memory arrays use Python ints" rather than "spr.py uses Python ints".

2. **`test_no_torch_in_npu_source` pre-existing false positive.** The test at `tests/gtx/test_npu_xp.py:23-28` greps npu.py for "torch." outside `#`-prefixed lines but doesn't exclude docstrings. Wave 1b commit 6072b37 added a docstring to `flush_deferred_ddr_stores` that mentions `torch.Tensor` while describing the WAVE-1-SHIM bridge contract. The docstring is informational and intentional; removing it would lose useful context. This is a pre-existing Wave 1b test bug, NOT introduced by Wave 2a. Deferred to plan 09-03 (Wave 6) test cleanup or a follow-up quick.

3. **`git stash` during diagnostic reverted in-progress Task 1 edits.** Mid-Task-1, I ran `git stash` to check pre-port baseline; the subsequent `git stash pop` failed with uv.lock conflict. Restored via `git checkout uv.lock && git stash pop`. No work was lost; all Task 1 edits intact.

## Known Stubs

None. All 4 op files are fully ported to xp; no placeholder text or fall-through `pass` statements. The FP8 LUT-only path is the single deterministic implementation per H-1.

## User Setup Required

None — no external service configuration required for Wave 2a.

## Next Phase Readiness

**Plan 09-02b-engines (Wave 5: dma_engine.py + dma_engine internal helpers) entry condition: MET.**

- ✅ All 4 op-handler modules + register.py docstring torch-free; 73/73 unit tests GREEN.
- ✅ 3 f16 shim sites removed; only 4 byte/ddr shims remain (Wave 5/6 obligations).
- ✅ ABS strict walltime back inside D-08 budget (96.68s vs 105s ceiling — 8% headroom).
- ✅ Wave 5 inherits the following shim-removal obligations: `l0_byte` (after porting dma_engine.py L155/179); `ddr.read` (after porting dma_engine.py L266/345/534/647/664). The `l1_byte` + `l2_byte` shims survive to Wave 6 because tloop_buffer.py:459/467/477/483/485 still consume them.

**Wave 6 (plan 09-03-finalize) carry-forward acceptance criteria:**
- All remaining shims (`l1_byte`, `l2_byte`) removed; `_torch_view` helper and module-level torch import deleted.
- npu.py's `flush_deferred_ddr_stores` docstring updated to remove the `torch.Tensor` reference (closes the `test_no_torch_in_npu_source` false positive).
- `from riscv.gtx import DEVICE` raises ImportError (D-04 clean-cut from Wave 0).

## Self-Check: PASSED

Files created (verified):
- FOUND: .planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md
- FOUND: .planning/phases/09-backend-migration-numpy-cupy/09-02a-SUMMARY.md (this file)

Files modified (verified):
- FOUND: src/main/python/riscv/gtx/unit/ins/ops/spr.py
- FOUND: src/main/python/riscv/gtx/unit/ins/ops/mm.py
- FOUND: src/main/python/riscv/gtx/unit/ins/ops/vec.py
- FOUND: src/main/python/riscv/gtx/unit/ins/ops/act.py
- FOUND: src/main/python/riscv/gtx/unit/csr/register.py
- FOUND: src/main/python/riscv/gtx/unit/memory.py
- FOUND: tests/gtx/test_memory_torch_shim.py

Commits (this plan):
- FOUND: de291ff (Task 1 — spr.py)
- FOUND: cfc2677 (Task 2 — mm.py)
- FOUND: d62ba27 (Task 3 — vec.py)
- FOUND: 6b2e3c1 (Task 4 — act.py FP8 LUT-only)
- FOUND: 020ebb9 (Task 5 — csr/register.py docstring)
- FOUND: 8b35f7c (Shim removal — l0/l1/l2_f16)
- FOUND: 21feb47 (Task 6 — Wave 2a gate doc)

Acceptance grep verification:
- mm.py torch refs: 0
- vec.py torch refs: 0
- act.py torch refs: 0
- spr.py torch refs: 0
- csr/register.py torch.Tensor: 0; xp.ndarray: 1
- act.py FP16_TO_FP8_LUT / FP8_TO_FP16_LUT: 6 (LUTs preserved)
- act.py float8_e4m3fn / ml_dtypes / NotImplementedError: 0 (H-1 deterministic single-path)
- memory.py WAVE-1-SHIM marker count: 4 (3 removed: l0_f16, l1_f16, l2_f16)

Smoke gate:
- 4 PASS + 1 SKIP (TANH) literal 6-op set
- ABS strict walltime: 96.68s (D-08 budget 85-105s — IN BUDGET)
- 73 / 73 baseline unit tests GREEN

---

*Phase: 09-backend-migration-numpy-cupy*
*Plan: 02a-ops (Wave 2 part a — op-handler xp port + f16 shim removal)*
*Completed: 2026-05-18*
