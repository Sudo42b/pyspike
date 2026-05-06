---
phase: 04-mm-subsystem
plan: 01
subsystem: testing
tags: [pytest, fp16, be-bit-pair, riscv-elf, mm-fixture, scaffold]

requires:
  - phase: 02-skeleton-disasm
    provides: "_RISCV_AVAILABLE module-level self-detect pattern (P2 plan-05 D-1) + .elf fixture pattern (D-22)"
  - phase: 03-dma-ddr-i-o
    provides: "tests/gtx 179-test green baseline; ops/ + dispatch_4mode + DDR I/O"
provides:
  - "Wave 0 RED scaffolds for all 18 P4 named tests (per 04-VALIDATION.md)"
  - "_verify_minimal.compare_hex BE FP16 bit-pair compare (Pitfall 1 lock)"
  - "mm_basic.{S,elf} pre-built RV64 ELF fixture (D-09 fallback)"
  - "mm_basic_n1s16.hex zero-init oracle golden (Blocker 1 Option B)"
  - "test_regression_fw_mm.py 3-tier skip discipline (_RISCV/ELF/pyspike)"
affects: [04-02-gemm-core, 04-03-mm-engine, 04-04-ops-mm, 04-05-regression]

tech-stack:
  added: []
  patterns:
    - "TDD-RED Wave 0 scaffold: pytest.skip() bodies, never assert hasattr (P3 D-5 lock)"
    - "ZERO-INIT oracle for plumbing-proof golden (subprocess L1=0 path alignment)"
    - "Local .gitignore !mm_basic.elf override mirrors P2 D-22 nop_wjoin.elf pattern"

key-files:
  created:
    - "tests/gtx/_verify_minimal.py"
    - "tests/gtx/test_op_mm.py"
    - "tests/gtx/test_mm_chain.py"
    - "tests/gtx/test_funct7_routing.py"
    - "tests/gtx/test_regression_fw_mm.py"
    - "tests/gtx/data/elf/mm_basic.S"
    - "tests/gtx/data/elf/mm_basic.elf"
    - "tests/gtx/data/golden/mm_basic_n1s16.hex"
  modified:
    - "tests/gtx/data/elf/Makefile"
    - "tests/gtx/data/elf/.gitignore"

key-decisions:
  - "Test-only _verify_minimal: NO CLI / NO argparse / NO __main__ block (D-13 lock; P6 promotes)"
  - "BE bit-pair done MANUALLY via (byte[0]<<8)|byte[1] not numpy newbyteorder magic (Pitfall 1 explicit)"
  - "ISS-full WRSPR (funct7=0x49) in mm_basic.S to avoid funct7=0x00 dispatch reentry"
  - "Zero-init golden: 32 bytes of 0x00 from gemm_core(zeros @ zeros) explicit 3-loop FP32"
  - "Local .gitignore added !mm_basic.elf override (Rule 3 blocking fix mirroring nop_wjoin.elf)"

patterns-established:
  - "Subprocess pyspike CLI invocation (D-11 fallback PRIMARY): shutil.which('pyspike') -> [sys.executable, '-m', 'riscv']"
  - "Module-level _RISCV_AVAILABLE try/except in every test file (works under --noconftest)"
  - "Each scaffold has substantive docstring explaining target behavior + Pitfall reference"

requirements-completed: []  # Wave 0 scaffold only — RED contracts laid; MM-01..05 GREEN-close in Wave 1 (Plans 02/03/04) + Wave 2 (Plan 05)

duration: 7min
completed: 2026-05-06
---

# Phase 4 Plan 01: Wave 0 MM Scaffold + .elf Fixture Summary

**TDD-RED gate landed: 18 named test scaffolds (per VALIDATION) all skip cleanly, _verify_minimal BE FP16 compare verified, mm_basic.elf pre-built and committed, zero-init golden hex synthesized — Wave 1 plans (02/03/04) unblocked for parallel execution.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-06T00:10Z
- **Completed:** 2026-05-06T00:18Z
- **Tasks:** 3
- **Files modified:** 10 (8 created, 2 modified)

## Accomplishments

- 18 VALIDATION-named test functions land as `pytest.skip()` scaffolds; 0 fail, 0 error, full P3 regression unaffected (179 -> 180 passing)
- `tests/gtx/_verify_minimal.compare_hex` exposes BE FP16 bit-pair compare per `verify.py:235`; verified by round-trip on `np.float16(1.0) = 0x3C00` BE encoding
- `mm_basic.elf` (RV64, entry 0x800000b0) pre-built and committed; 4 LE-disassembled `.insn` rows match the planned encoding (3 ISS-full WRSPRs + 1 MM custom0 funct7=0x00 funct3=2 rs1=x1 rd=x10 + WJOIN)
- Zero-init oracle golden hex (32 bytes of 0x00) committed at `tests/gtx/data/golden/mm_basic_n1s16.hex`; aligns with Plan 05 subprocess L1=0 path

## Task Commits

1. **Task 1: _verify_minimal.py BE FP16 bit-pair compare** — `26b8262` (feat)
2. **Task 2: mm_basic.S + Makefile rule + pre-built .elf** — `3b95f38` (feat)
3. **Task 3: 4 test scaffold files + zero-init golden hex** — `2a03451` (test)

## Files Created/Modified

- `tests/gtx/_verify_minimal.py` (NEW, 72 LOC) — `compare_hex(actual, golden, *, ulp, atol, strict) -> (bool, dict)`; BE bit-pair via `(byte[0]<<8)|byte[1]`; signed-magnitude ULP fallback; NaN reported as max ULP
- `tests/gtx/test_op_mm.py` (NEW, 11 scaffolds) — MM-01 (gemm_core×2), MM-02 (handler-registry, exec_mm_basic, exec_mm_s/o/v/t = 6), MM-03 (decode), MM-05 (verify_minimal smoke), MM-01 (signature_stateless)
- `tests/gtx/test_mm_chain.py` (NEW, 4 scaffolds) — MM-04 ADDRC chain, mxe_accum chain, per-cell isolation, dtype-locked
- `tests/gtx/test_funct7_routing.py` (NEW, 3 scaffolds) — MM-03 funct7=0x00 collision, funct7=0x01 always-MMC, MM-05 #5 Mode 4 routing
- `tests/gtx/test_regression_fw_mm.py` (NEW, 1 strict-mode scaffold + 1 fixture-present always-runnable check) — 3-tier skip: `_RISCV_AVAILABLE` -> ELF exists -> golden exists; subprocess D-11 fallback as PRIMARY
- `tests/gtx/data/elf/mm_basic.S` (NEW, 49 LOC) — `_start`: 3× `WRSPR ADDRA/B/R` (ISS-full funct7=0x49) + 1× `mm` (custom0 funct7=0x00 funct3=2, rs1=x1=0x0004000000040004, rd=x10) + WJOIN + `j .` safety loop
- `tests/gtx/data/elf/mm_basic.elf` (NEW binary, ELF64 RV64 LSB executable, entry 0x800000b0)
- `tests/gtx/data/elf/Makefile` (MODIFIED) — added `mm_basic.elf: mm_basic.S` rule mirroring `nop_wjoin.elf`; clean covers both; new `verify-mm` phony
- `tests/gtx/data/elf/.gitignore` (MODIFIED) — added `!mm_basic.elf` override
- `tests/gtx/data/golden/mm_basic_n1s16.hex` (NEW, 1 line × 64 hex chars = 16 FP16 zeros, BE bit-pair) — synthesized from explicit-3-loop FP32 `gemm_core(zeros @ zeros) -> zeros`

## Decisions Made

- **`_verify_minimal` BE conversion done explicitly** via `(byte[0]<<8)|byte[1]`, not via numpy `>u2`/`newbyteorder`. Reason: matches `verify.py:235` line-for-line; numpy 2.x deprecates `newbyteorder` and the explicit form is bit-exact regardless of host endianness.
- **`mm_basic.S` uses ISS-full WRSPR (funct7=0x49)** instead of gem5-simplified (funct7=0x00). Reason: Wave 0 ships ELF before MM @handler is wired; using funct7=0x00 for WRSPR would reenter the not-yet-implemented MM dispatch path.
- **Zero-init golden (Blocker 1 Option B alignment)**: golden hex is 32 bytes of 0x00 because `mm_basic.elf` runs against zero-init L1 (firmware does NOT pre-load operands). gemm_core(zeros @ zeros) = zeros. Plumbing-proof: if any @handler crashes, mm_basic.elf never reaches WJOIN, subprocess returncode != 0, test fails. Non-trivial operand staging deferred to P6.
- **Local `.gitignore` `!mm_basic.elf` override** added (Rule 3 blocking fix). Project-level `.gitignore` masks `*.elf`; the existing `tests/gtx/data/elf/.gitignore` only had `!nop_wjoin.elf`, so the new ELF was being ignored. Mirrors P2 D-22 commit-binary pattern.
- **Test scaffold body = `pytest.skip(...)` (NOT `assert hasattr(...)`)** per P3 plan-01 D-5 lock. `assert hasattr` would fail the verify step before Wave 1 fills modules; `pytest.skip` passes cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `!mm_basic.elf` override to local .gitignore**
- **Found during:** Task 2 (`git add tests/gtx/data/elf/mm_basic.elf` failed with "ignored by .gitignore")
- **Issue:** Project-level `/.gitignore` line 3 (`*.elf`) masks new ELFs. Plan assumed the local `tests/gtx/data/elf/.gitignore` already overrode for any committed binary, but it only listed `!nop_wjoin.elf`.
- **Fix:** Added `!mm_basic.elf` line to `tests/gtx/data/elf/.gitignore`. Updated comment to reference both binaries (D-22 and P4 D-09).
- **Files modified:** `tests/gtx/data/elf/.gitignore`
- **Verification:** `git ls-files tests/gtx/data/elf/` now lists `mm_basic.elf`.
- **Committed in:** `3b95f38` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Surgical 2-line .gitignore edit. Same pattern as existing `!nop_wjoin.elf` override; no scope creep.

## Issues Encountered

- None during execution. Toolchain (`/opt/riscv/bin/riscv64-unknown-elf-gcc`) was available, so the D-09 toolchain-unavailable fallback (commit only .S + Makefile, let regression test skip) was not needed.
- The disassembled `mm_basic.elf` confirms encoding: instruction at `0x800000e6` is `0x0000a50b` = custom0 (opcode 0x0b), funct3=2, funct7=0x00, rs1=x1, rd=x10 — exactly per plan.

## Known Stubs

All test scaffolds in this plan are intentional `pytest.skip("Wave 1: ...")` placeholders per the TDD-RED Wave 0 contract (P3 plan-01 D-5 lock). They will be GREEN-replaced by Wave 1 plans:

| File | Tests | Resolved by |
|------|-------|-------------|
| `tests/gtx/test_op_mm.py` | 11 (test_gemm_core_*, test_handler_registry_*, test_exec_mm_*, test_decode_firmware_mm_args, test_verify_minimal_be_fp16_pairs, test_gemm_core_signature_stateless) | Wave 1: Plans 02 (gemm_core) + 03 (mm_engine) + 04 (ops/mm) |
| `tests/gtx/test_mm_chain.py` | 4 (test_mm_addrc_chain_*, test_mxe_accum_*) | Wave 1: Plan 04 (ops/mm) |
| `tests/gtx/test_funct7_routing.py` | 3 (test_funct7_*, test_mode4_*) | Wave 1: Plans 03 + 04 |
| `tests/gtx/test_regression_fw_mm.py::test_mm_basic_strict_mode_pass` | 1 | Wave 2: Plan 05 (subprocess regression) |

These are NOT "passing assertions hiding missing functionality" — every scaffold body is `pytest.skip(...)` with an explicit message naming the Wave 1/2 plan that fills it. The full P4 SUMMARY chain (this + 02 + 03 + 04 + 05) closes MM-01..05.

## Self-Check: PASSED

All 8 created files exist on disk:
- `tests/gtx/_verify_minimal.py` ✓
- `tests/gtx/test_op_mm.py` ✓
- `tests/gtx/test_mm_chain.py` ✓
- `tests/gtx/test_funct7_routing.py` ✓
- `tests/gtx/test_regression_fw_mm.py` ✓
- `tests/gtx/data/elf/mm_basic.S` ✓
- `tests/gtx/data/elf/mm_basic.elf` ✓
- `tests/gtx/data/golden/mm_basic_n1s16.hex` ✓

All 3 commits exist: `26b8262`, `3b95f38`, `2a03451`.

`pytest tests/gtx/ --noconftest -o "addopts=" -q`: **180 passed, 19 skipped, 0 failed** (P3 baseline 179 + new fixture-present 1).

## Next Wave Readiness

Wave 1 plans can now begin in parallel:

- **Plan 02 (gemm_core)** — `tests/gtx/test_op_mm.py::test_gemm_core_*` are RED scaffolds; Plan 02 GREEN-fills `riscv.gtx.gemm_core.gemm_core(A, B, *, has_bias, prior_accum)`.
- **Plan 03 (mm_engine)** — `tests/gtx/test_op_mm.py::test_decode_firmware_mm_args` + `test_funct7_routing.py::test_mode4_routes_to_tmu_curr` are RED scaffolds; Plan 03 GREEN-fills `riscv.gtx.mm_engine.{decode_firmware_mm_args, firmware_mm}`.
- **Plan 04 (ops/mm)** — `test_op_mm.py::test_handler_registry_has_all_10_mm_variants` + `test_exec_mm_*` + `test_mm_chain.py` + `test_funct7_routing.py::test_funct7_*` are RED scaffolds; Plan 04 GREEN-fills the 10 `@handler` entry points + WRSPR-collision NOP safety.
- **Plan 05 (regression)** — `test_regression_fw_mm.py::test_mm_basic_strict_mode_pass` is the final scaffold; Plan 05 wires the subprocess invocation + DDR dump + `compare_hex(strict=True)` PASS.

No blockers. The .elf fixture, golden hex, and BE FP16 compare are ready for Wave 2 strict-mode regression to consume.

---
*Phase: 04-mm-subsystem*
*Completed: 2026-05-06*
