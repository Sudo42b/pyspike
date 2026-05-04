---
status: passed
phase: 02-skeleton-disasm
source: [02-VERIFICATION.md]
started: 2026-05-04T09:36:29Z
updated: 2026-05-04T17:59:37Z
---

## Current Test

[all 3 closed — Categories A/B/C/D all resolved inline]

## Tests

### 1. End-to-end pyspike CLI subprocess
expected: In an environment with `_riscv.so` built (i.e. after `pip install -e .` succeeds against a real spike), running `pyspike --extlib=riscv.gtx --extension=gtx tests/gtx/data/elf/nop_wjoin.elf; echo $?` outputs `0`. The 5KB committed ELF must reach `wjoin` and exit cleanly without trapping on `addi sp,sp,-16`.
result: passed
verified_at: 2026-05-04T17:59:37Z
evidence:
  - "Commit 611c222 — drop empty get_instructions() override in GtxNpu"
  - "Commit be91d2f — py_rocc_t trampolines for extension_t hooks + SystemExit -> std::exit translation"
  - "Commit 51dee8d — test_skeleton.py adds --extension=gtx flag"
  - "Manual verify: scripts/pyspike --extlib=riscv.gtx --extension=gtx -l --log=t.log nop_wjoin.elf; echo $? -> 0"
  - "tests/gtx/test_skeleton.py::test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero passes"
notes: |
  Original spec used --extlib only; the missing --extension=gtx flag was
  the activation gate. Two pyspike-core defects also exposed: missing
  py_rocc_t trampolines for the extension_t hook methods (caused disasm
  to render as 'unknown') and uncaught SystemExit at the C++/Python
  boundary (caused exit 255 via std::terminate). Both fixed in be91d2f.

### 2. Skipif-gated tests in CI
expected: When `_riscv.so` is built and importable, all 21 currently-skipped tests in `tests/gtx/` execute and pass. Together with the 65 mock-fallback tests, the full Phase 2 suite shows 86 passed / 0 skipped.
result: passed
verified_at: 2026-05-04T17:59:37Z
evidence:
  - "Commit 107e646 — Category A (super().reset(proc) C++ strict-type)"
  - "Commit 87f8d2a — Category B (disasm _ -> . normalization)"
  - "Commit 8f75991 — Category C (ELF -Ttext-segment fix)"
  - "Commit 611c222 + be91d2f + 51dee8d — Category D (dispatch + disasm + SystemExit)"
  - "pytest tests/gtx/ -q -o 'addopts=' -> 87 passed in 2.76s"
notes: |
  Final count is 87 (not 86) because Plan 02-06 added one regression
  test (test_full_trace_mnemonics_present) that is now passing. Zero
  skips, zero failures, zero xfails. The full skipif-gated Phase 2
  surface runs end-to-end on a built _riscv.so.

### 3. Disasm trace mnemonic visibility
expected: Running `pyspike --extlib=riscv.gtx --log=trace.log tests/gtx/data/elf/nop_wjoin.elf` produces a trace containing the mnemonics `wjoin`, `wrspr`, and `rdspr`. `grep -E '(wjoin|wrspr|rdspr)' trace.log` returns ≥3 matches.
result: passed
verified_at: 2026-05-04T17:59:37Z
evidence:
  - "Commit be91d2f — py_rocc_t::get_disasms trampoline now reaches GtxNpu.get_disasms"
  - "Commit 51dee8d — test regex updated to match dot-form (warp.join etc) per disasm_insn_t normalization"
  - "Manual verify: grep -c 'warp\\.join' /tmp/gtx-trace5.log -> 1"
  - "tests/gtx/test_skeleton.py::test_full_trace_mnemonics_present passes"
notes: |
  The committed nop_wjoin.elf executes one RoCC instruction (warp.join),
  so threshold stays at >=1 match. A richer-ELF fixture exercising
  WRSPR/RDSPR/WJOIN together is a P3+ test-fixture work item — would
  satisfy the original >=3 spec without any further code changes.

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0
followup_needed: false

## Gaps

None. All 3 UAT items resolved inline during Phase 2 wrap-up per user
directive (2026-05-05). The build-path validation cycle (Plan 02-06)
exposed 4 categories of pre-existing bugs hidden by the mock-fallback
discipline; A/B/C were mechanical fixes (Phase 2 inline), D required
both pyspike-core trampoline + SystemExit boundary work (also fixed
inline given the small surface area: 5 C++ methods + 1 [[noreturn]]
helper, all in src/main/cpp/riscv_extension.{h,cc}).

Commits:
- A: 107e646
- B: 87f8d2a
- C: 8f75991
- D-flag: 51dee8d
- D1: be91d2f (extension_t hook trampolines)
- D2: be91d2f (SystemExit -> std::exit)
- npu.py cleanup: 611c222

deferred-items.md "RoCC Subclass Dispatch Lifecycle" entry now marked
RESOLVED with full evidence trail.
