---
type: phase-seed
proposed_phase: 8
title: Multi-tile DMA orchestration parity
status: proposed
created: 2026-05-10
trigger: P7 ABS smoke test (07-HUMAN-UAT.md "Findings" 2026-05-10)
related_artifacts:
  - .planning/phases/07-numba/07-HUMAN-UAT.md
  - test/ABS/n1s16/n1s16_abs.c (vendor reference)
  - test/ABS/n1s16/data/n1s16_abs_ref.txt (golden BE FP16)
  - src/main/python/riscv/gtx/vec_engine.py (_apply_unary, dispatch)
  - src/main/python/riscv/gtx/dma.py (multi-tile DMA loop)
  - src/main/python/riscv/gtx/ddr.py (GTX_DDR_REVERSED handling)
discovered_during: P7 vendor 84-op sweep enablement attempt
---

## Goal (proposed)

Port the multi-tile DMA orchestration path from vendor `gtx_npu_dma.cc`
so that vendor `n1s16_<op>.elf` regression sweeps pass strict-mode
`compare_hex(strict=True)` against `_ref.txt` for the full output region
(not just the first `MAX_SHARED_DMA_BYTES=65535` tile).

## Symptom

`pyspike --extlib=riscv.gtx --extension=gtx test/ABS/n1s16/n1s16_abs.elf`
with `GTX_DDR_REVERSED=1` and `GTX_DDR_INIT=…/input.txt`:

- First **2047 lines** of dump (≈ 64 KB, exactly one
  `MAX_SHARED_DMA_BYTES=65535` worth of FP16 data) match `_ref.txt`
  **byte-exact** (delta_ulp == 0).
- Lines **2048..end** diverge from the reference.

So the *compute kernel* (vec_engine `_apply_unary` SIGN→ABS path) and
the *first DMA tile* are correct. The bug is in the orchestration that
walks the second and subsequent tiles.

## Working Hypotheses (in priority order)

1. **DDR↔L2 source/dest pointer not advancing** between tiles. Vendor
   does explicit `__load(BASE_DDR_A + tile_offset, L2_A, tile_bytes)` at
   the head of each tile loop iteration; pyspike's interpretation may be
   reusing tile 0's pointers.
2. **L1 bank not being recycled** — after `__store_cr` for tile 0 fires,
   subsequent `__load_cr` to the same L1 bank may be reading stale
   compute-side state instead of the new DMA payload.
3. **Credit gate stuck** — `__credit_chk` may be comparing against a
   counter that wraps incorrectly past tile 1, letting compute proceed
   on uninitialized L1 contents.
4. **Plan/thread state machine reset** — the `__split → __start_plan →
   __start_thread` sequence inside the inner tile loop may not be
   resetting NEST/SPU dispatch context, causing thread 0 of tile 2 to
   inherit tile 1's local addresses.

## Likely Investigation Steps

1. Read vendor `gtx_npu_dma.cc` shared/thread DMA tile loop verbatim and
   diff against pyspike's `dma.py` / `mm_engine.py` / `vec_engine.py`
   tile orchestration.
2. Add a per-tile assertion that L1 bank A contents match what
   `__load(...)` should have placed there; surface first divergence
   (which tile, which row).
3. If hypothesis 1 wins → audit `npu.lspr[nest][spu][LSPR_SPM_ADDR*]`
   advancement across `__store_cr` / `__load_cr` calls; check whether
   pyspike resets these or accumulates correctly.
4. Once isolated, write a tile-2 unit test (smaller HEIGHT, two tiles
   only) that fails before the fix and passes after.

## Out of scope for P7

P7 (Numba Dynamic Optimization) committed to per-kernel ULP-0 parity
(28 stateless kernels) and the JIT promotion + extras + CI sync. End-
to-end vendor regression with multi-tile DMA was always a *consequence*
of P5/P6 plumbing being complete, not a P7 deliverable. The smoke test
on 2026-05-10 surfaced this gap; capturing it here so it lands as a
proper phase rather than an undocumented loose end.

## Acceptance Criteria (proposed)

1. `pytest tests/gtx/test_regression_fw_full_sweep.py -v --no-cov` with
   `GTX_DDR_REVERSED=1` reports M >= 12 PASS (strict-mode compare against
   vendor `_ref.txt`) on a checkout that has `pyspike/test/` populated
   with vendor `.elf`.
2. ABS, ADD_VV, MUL_VV, RELU, SIGMOID, GELU all PASS as a representative
   smoke set spanning vec / act / multi-input ops.
3. Tile-2 unit test (added during fix) lives at
   `tests/gtx/test_multi_tile_dma.py` and PASSes — protects against
   regression without requiring vendor `.elf`.
4. P7 HUMAN-UAT items #1 (M >= 12) and #2 (5x walltime) close out via
   `/gsd:verify-work 7`.

## Notes

- Numba speedup is already proven (ABS in 4.8 s on this checkout) — once
  correctness lands, the 5x walltime gate should fire naturally without
  additional optimization work.
- All 79 vendor `.elf` files at `/mnt/e/14_NIGHTLY/pyspike/test/` are
  untracked. Decide during P8 planning whether to commit them as test
  fixtures or symlink from the developer's vendor checkout.
