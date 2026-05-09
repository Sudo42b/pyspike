---
status: partial
phase: 07-numba
source: [07-VERIFICATION.md]
started: 2026-05-09T00:00:00Z
updated: 2026-05-09T00:00:00Z
---

## Current Test

[awaiting human testing on developer machine with GFW source tree]

## Tests

### 1. Vendor 84-op sweep with M >= 12 (Success Criterion #2)
expected: GFW source tree (gtx/address.h + intrinsic headers) present → 72 vendor `n1s16_<op>.c` kernels cross-compile to `.elf` → `pytest tests/gtx/test_regression_fw_full_sweep.py -v --no-cov` reports `M passed + N skipped where M+N==84 and M >= 12` (M >= 60 with `/opt/riscv/` toolchain present). Strict-mode `compare_hex(strict=True)` PASS for at least 12 ops.
result: [pending]
why_human: GFW source tree is missing on this checkout — vendor n1s16 kernels cannot be cross-compiled to `.elf` without it. M=0 here is environmental, not a code defect. Harness wiring (5-tier graceful skip + 84-op auto-discover + OPERAND_STAGING_REQUIRED_VENDOR set) verified correct in 07-VERIFICATION.md.

### 2. 5x walltime acceptance (Success Criterion #3)
expected: After Test 1 produces real `.elf` work, re-record `tests/gtx/data/baseline_walltime.txt` under `HAS_NUMBA=False` → `pytest tests/gtx/test_njit_perf.py --benchmark-only` reports `test_vendor_sweep_walltime_5x` PASSes (asserts `mean*5 <= baseline_walltime`, NOT skipped via 30s threshold).
result: [pending]
why_human: Current baseline_walltime.txt = 4.5s reflects pytest+numba startup overhead since M=0 vendor `.elf` is executing. Plan 05 deviation #3 added a 30s threshold-based skip so CI stays green on minimal-work environments. Assertion machinery + benchmark harness verified correct; the actual 5x measurement requires real vendor work.

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
