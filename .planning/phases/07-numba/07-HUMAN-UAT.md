---
status: partial
phase: 07-numba
source: [07-VERIFICATION.md]
started: 2026-05-09T00:00:00Z
updated: 2026-05-10T00:00:00Z
---

## Current Test

[2 items pending — environment-bound, two new findings unblocked but surface a separate multi-tile DMA bug]

## Tests

### 1. Vendor 84-op sweep with M >= 12 (Success Criterion #2)
expected: GFW source tree (gtx/address.h + intrinsic headers) present → 72 vendor `n1s16_<op>.c` kernels cross-compile to `.elf` → `pytest tests/gtx/test_regression_fw_full_sweep.py -v --no-cov` reports `M passed + N skipped where M+N==84 and M >= 12` (M >= 60 with `/opt/riscv/` toolchain present). Strict-mode `compare_hex(strict=True)` PASS for at least 12 ops.
result: [pending — partial progress on 2026-05-10; see Findings below]
why_human: GFW source tree was located at `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/` (full layout with address.h, 10 GTX headers, intrinsics, 4 linker scripts). 79 pre-built vendor `.elf` + 70 `_ref.txt` golden files were also discovered at `/mnt/e/14_NIGHTLY/pyspike/test/<OP>/n1s16/n1s16_<op>.elf` (untracked). Smoke test on `n1s16_abs.elf` ran in **4.8 s with numba enabled** (vs the >90 s timeout assumption that drove the original deferral) — fully practical for sweep execution. **Endianness root cause identified and resolved**: vendor DDR format is big-endian FP16, pyspike default is little-endian; setting `GTX_DDR_REVERSED=1` produces byte-exact match against the vendor reference for the first DMA tile (~64 KB / `MAX_SHARED_DMA_BYTES=65535`). **Remaining blocker**: lines past tile 1 diverge — appears to be a multi-tile DMA orchestration bug in pyspike's NPU model (address/counter/L1-reuse path that single-tile P5/P6 hand-written `.elf` never exercised). Resolution is a focused debug effort outside the P7 scope; tracked as P8 seed (see `.planning/seeds/p8-multi-tile-dma.md`).

### 2. 5x walltime acceptance (Success Criterion #3)
expected: After Test 1 produces real `.elf` work, re-record `tests/gtx/data/baseline_walltime.txt` under `HAS_NUMBA=False` → `pytest tests/gtx/test_njit_perf.py --benchmark-only` reports `test_vendor_sweep_walltime_5x` PASSes (asserts `mean*5 <= baseline_walltime`, NOT skipped via 30s threshold).
result: [pending]
why_human: Blocked by Test 1 — once multi-tile DMA bug is fixed and ABS et al. pass strict-mode compare, baseline can be re-recorded under `HAS_NUMBA=False` and the 5x assertion will fire. Smoke benchmark already shows numba reduces full-vendor ABS walltime to ~5 s; the 5x speedup is highly likely to land naturally once correctness is in place.

## Findings (2026-05-10 ABS smoke test)

| Finding | Detail |
|---------|--------|
| GFW location | `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/` — `include/gtx/address.h`, 10 headers, 4 linker scripts, intrinsics .c — complete tree |
| Pre-built vendor assets | `/mnt/e/14_NIGHTLY/pyspike/test/` (untracked) — 79 `n1s16_<op>.elf` + 70 `_ref.txt` |
| numba speed | ABS full vendor (12 MB DDR, 16-SPU NEST) in **4.8 s** with numba enabled — sweep is operationally feasible |
| Endianness | `GTX_DDR_REVERSED=1` required for vendor BE FP16 vs pyspike default LE; byte-exact for first DMA tile |
| Multi-tile DMA bug | First tile (~64 KB / `MAX_SHARED_DMA_BYTES=65535`) byte-exact; lines past tile 1 diverge — see P8 seed |
| Dispatch coverage | `abs.v` opcode = `0x0b custom0`, funct7 `0x1D` (GTX_F7_VEC_SIGN), sub_op 0 — already wired (commit `fcf2ebc`); compute path correct |

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
