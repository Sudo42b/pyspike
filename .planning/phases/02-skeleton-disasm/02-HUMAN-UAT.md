---
status: partial
phase: 02-skeleton-disasm
source: [02-VERIFICATION.md]
started: 2026-05-04T09:36:29Z
updated: 2026-05-04T09:36:29Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. End-to-end pyspike CLI subprocess
expected: In an environment with `_riscv.so` built (i.e. after `pip install -e .` succeeds against a real spike), running `pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf; echo $?` outputs `0`. The 5KB committed ELF must reach `wjoin` and exit cleanly without trapping on `addi sp,sp,-16`.
result: [pending]

### 2. Skipif-gated tests in CI
expected: When `_riscv.so` is built and importable, all 21 currently-skipped tests in `tests/gtx/` execute and pass. Together with the 65 mock-fallback tests, the full Phase 2 suite shows 86 passed / 0 skipped.
result: [pending]

### 3. Disasm trace mnemonic visibility
expected: Running `pyspike --extlib=riscv.gtx --log=trace.log tests/gtx/data/elf/nop_wjoin.elf` produces a trace containing the mnemonics `wjoin`, `wrspr`, and `rdspr`. `grep -E '(wjoin|wrspr|rdspr)' trace.log` returns ≥3 matches against the sampled ELF.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
