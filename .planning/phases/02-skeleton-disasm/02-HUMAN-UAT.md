---
status: needs_followup
phase: 02-skeleton-disasm
source: [02-VERIFICATION.md]
started: 2026-05-04T09:36:29Z
updated: 2026-05-04T16:00:00Z
---

## Current Test

[blocked by Phase-2 follow-up — Categories A-D in 02-06-BUILD-LOG.md]

## Tests

### 1. End-to-end pyspike CLI subprocess
expected: In an environment with `_riscv.so` built (i.e. after `pip install -e .` succeeds against a real spike), running `pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf; echo $?` outputs `0`. The 5KB committed ELF must reach `wjoin` and exit cleanly without trapping on `addi sp,sp,-16`.
result: blocked
verified_at: 2026-05-04T16:00:00Z
evidence:
  - "02-06-BUILD-LOG.md Category C: ELF LOAD segment at 0x7ffff000 (not 0x80000000)"
  - "02-06-BUILD-LOG.md Category D: sp not initialized; custom1 dispatch broken"
  - "Commit b81b000 — test_full_trace_mnemonics_present added; currently fails upstream"
notes: |
  _riscv.so was built successfully (commit 761b970), unblocking the test gate.
  However the subprocess still exits 255 with "Memory address 0x7ffff000 is invalid"
  due to ELF Makefile using -Ttext (not -Ttext-segment). Once the Makefile is
  fixed and the ELF rebuilt, dispatch debugging (Category D) is needed to make
  WJOIN actually reach the GtxNpu.custom1 handler.

### 2. Skipif-gated tests in CI
expected: When `_riscv.so` is built and importable, all 21 currently-skipped tests in `tests/gtx/` execute and pass. Together with the 65 mock-fallback tests, the full Phase 2 suite shows 86 passed / 0 skipped.
result: partial
verified_at: 2026-05-04T16:00:00Z
evidence:
  - "02-06-BUILD-LOG.md Step 2.2: 71 passed, 15 failed, 0 skipped"
  - "Commit afc6e56 — full per-file accounting captured"
notes: |
  Skips resolved 21 -> 0 (gap-closure assertion satisfied). However 15 of the
  newly-running tests fail:
    - test_reset.py: 8 failures (Category A: super().reset(proc) C++ strict-type)
    - test_disasm.py: 6 failures (Category B: _ vs . mnemonic separator)
    - test_skeleton.py: 1 failure (Category C: ELF LOAD-segment bug)
  Categories A-C are pre-existing bugs hidden by the mock-fallback discipline.

### 3. Disasm trace mnemonic visibility
expected: Running `pyspike --extlib=riscv.gtx --log=trace.log tests/gtx/data/elf/nop_wjoin.elf` produces a trace containing the mnemonics `wjoin`, `wrspr`, and `rdspr`. `grep -E '(wjoin|wrspr|rdspr)' trace.log` returns ≥3 matches.
result: blocked
verified_at: 2026-05-04T16:00:00Z
evidence:
  - "02-06-BUILD-LOG.md Step 3.1: subprocess fails with 'Memory address 0x7ffff000 is invalid'"
  - "Commit b81b000 — test_full_trace_mnemonics_present added with threshold lowered to >=1"
notes: |
  Test added per Task 3, with threshold lowered from >=3 to >=1 (Step 3.4
  fallback rationale: spike --log dumps executed instructions only; committed
  ELF executes 1 RoCC instruction; richer-ELF fixture deferred to P3+).
  Test currently fails because of the upstream Category C+D bugs blocking
  pyspike from running the ELF at all.

## Summary

total: 3
passed: 0
issues: 3
pending: 0
skipped: 0
blocked: 3
followup_needed: true

## Gaps

All 3 UAT items remain unresolved because Plan 02-06's build-path validation
exposed pre-existing bugs (Categories A-D in 02-06-BUILD-LOG.md) that cannot be
fixed within the plan's `files_modified` scope. A follow-up plan (02-07 or roll
into `/gsd:phase-evolve 2` cleanup) is required to:
1. Remove the no-op `super().reset(proc)` from `npu.py:74` (Category A fix)
2. Update `test_disasm.py` mnemonic expectations to dot-form (Category B fix)
3. Fix `tests/gtx/data/elf/Makefile` to use `-Wl,-Ttext-segment=0x80000000` and rebuild ELF (Category C fix)
4. Investigate sp initialization lifecycle and custom1 dispatch under spike (Category D fix)

After follow-up resolution, re-run `pytest tests/gtx/ -q -o "addopts="` and
re-verify these 3 UAT items.

Closure-cycle attempted by 02-06-PLAN.md; see 02-06-BUILD-LOG.md for full evidence.
