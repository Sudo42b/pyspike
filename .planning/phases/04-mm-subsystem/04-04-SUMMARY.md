---
phase: 04-mm-subsystem
plan: 04
subsystem: compute
tags: [mm-handlers, ops-mm, funct7-collision, wrspr-redispatch, pitfall-f, pitfall-d, l0-be-mm-o, l0-le-mm-v, mode4-isolation]

requires:
  - phase: 04-mm-subsystem
    provides: "Plan 03 mm_engine.firmware_mm dispatcher (variant-string + is_accumulate API) + Plan 02 gemm_core stateless kernels + Plan 01 Wave 0 scaffolds"
provides:
  - "src/main/python/riscv/gtx/ops/mm.py: 10 @handler entries (5 MM funct7=0x00 + 5 MMC funct7=0x01), each a 2-3 line forwarder to mm_engine.firmware_mm"
  - "encoding.py: 5 new MM funct3 constants (GTX_F3_MM_S/O/M/V/T)"
  - "ops/__init__.py: mm module registration triggers @handler decorators on import"
  - "spr.py wrspr_gem5/rdspr_gem5 rs1!=0 branch wired to MM/MMC re-dispatch (Plan 04 fill of P2 stub)"
affects: [04-05-regression]

tech-stack:
  added: []
  patterns:
    - "Spike-bound shim layer mirroring P3 ops/dma.py boundary (npu/proc/insn args; pure dispatch delegated to mm_engine)"
    - "WRSPR/RDSPR funct7=0x00/0x01 collision: None-key handler from P2 (wrspr_gem5/rdspr_gem5) re-dispatches to per-funct3 MM/MMC handler when insn.rs1 != 0"
    - "Per-handler insn.rs1==0 NOP guard preserved as defensive symmetry (also documents Pitfall F intent at the routing site)"

key-files:
  created:
    - "src/main/python/riscv/gtx/ops/mm.py"
  modified:
    - "src/main/python/riscv/gtx/encoding.py"
    - "src/main/python/riscv/gtx/ops/__init__.py"
    - "src/main/python/riscv/gtx/ops/spr.py"
    - "tests/gtx/test_op_mm.py"
    - "tests/gtx/test_funct7_routing.py"
    - "tests/gtx/test_spr.py"
    - "tests/gtx/test_dispatch.py"

key-decisions:
  - "wrspr_gem5/rdspr_gem5 rs1!=0 branch re-dispatches to per-funct3 MM/MMC handler. Required because the existing P2 None-key handler at funct7=0x00/0x01 wins the 2-level dispatch precedence in npu.custom0; without re-dispatch, the 5 MM funct3-keyed handlers would be unreachable. The plan did not anticipate the existing None-key handler -- caught at first verification of Task 2."
  - "Per-handler insn.rs1==0 guard kept inside ops/mm.py despite being functionally dead-code (unreachable when reached via wrspr_gem5 re-dispatch which only forwards rs1!=0). Reasoning: keeps ops/mm.py self-documenting at the routing site, and provides defensive symmetry if a future caller bypasses wrspr_gem5 (e.g. a P5 dispatch_iss_opcode promotion)."
  - "test_dispatch.py P2-era test test_custom0_funct7_collision_rs1_nonzero_returns_zero updated to use 1x1x1 GEMM dims. Original used 0x900 packed-rs1 which decoded to 2304x65536 dims and hung the GEMM loop indefinitely. The original assertions (rc==0, no SPR mutation) remain intact; only the input was right-sized to a valid MM dispatch."
  - "test_spr.py _fake_npu shim grew _custom0={} field. Mirrors Phase 3 P5 D-3 pattern (test shim grows when production handler signature changes); production GtxNpu callers unaffected."

patterns-established:
  - "Pre-existing None-key handler at a funct7 (e.g. wrspr_gem5 at 0x00) participates in MM dispatch by re-dispatching to its sibling funct3-keyed handlers. Pattern reusable for future funct7 collisions (e.g. RDSPR-VEC if P5 introduces analogous overlap)."
  - "@handler shim discipline locked: each ops/mm.py entry is a 2-3 line forwarder. No business logic; no decode; no variant fan-out. mm_engine owns variant dispatch; gemm_core owns the kernel."
  - "Pitfall D NxM transposed write verified end-to-end via test_exec_mm_t_writes_transposed (mm_t writes C^T flat to ADDRR, not C; reading as (N, M) shape returns C^T)."
  - "L0 BE/LE asymmetry test pair (test_exec_mm_o BE 0x49 0x00 vs test_exec_mm_v LE 0x60 0x54) pins the gtx_npu_mm.cc:217-218 vs :274-275 byte-order divergence at the integration level."

requirements-completed: [MM-02, MM-03]

duration: 44min
completed: 2026-05-06
---

# Phase 4 Plan 04: ops/mm @handler Shim Layer Summary

**10 individual @handler entries (5 MM + 5 MMC) + WRSPR collision re-dispatch land the spike-bound MM routing surface; 11 Wave 0 scaffolds GREEN; suite goes from 184 passed / 15 skipped to 195 passed / 5 skipped (remaining skips owned by Plan 05).**

## Performance

- **Duration:** ~44 min (mostly time blocked on hung pytest runs from a pre-existing test_dispatch.py timeout that was unmasked by the Plan 04 collision-routing change; once isolated, fix took 5 min)
- **Started:** 2026-05-06T00:47:57Z
- **Completed:** 2026-05-06T01:31:56Z
- **Tasks:** 3
- **Files modified:** 7 (1 created, 6 modified)
- **Commits:** 3

## Accomplishments

- `src/main/python/riscv/gtx/ops/mm.py` lands as a 148-LOC pure routing module: 10 thin `@handler` forwarders, each delegating to `mm_engine.firmware_mm(npu, proc, insn, is_accumulate=<funct7==0x01>, variant=<mnemonic>)`. Zero local dispatch logic. Variant table single-sourced in `mm_engine.py` (Plan 03).
- 5 new funct3 constants added to `encoding.py` (`GTX_F3_MM_S=0, GTX_F3_MM_O=1, GTX_F3_MM=2, GTX_F3_MM_V=3, GTX_F3_MM_T=7`); used by `ops/mm.py` exclusively (no literals in handler bodies).
- `ops/__init__.py` imports `mm` to trigger the 10 `@handler` decorators at PythonBridge load time.
- `spr.py` `wrspr_gem5`/`rdspr_gem5` rs1!=0 branch evolved from "stub returns 0" into "re-dispatch into per-funct3 MM/MMC handler"; this was the documented Plan 04 fill point per spr.py docstring's "P4: firmware_mm_op -> return 0 stub for P2".
- 11 Wave 0 scaffolds GREEN-filled: 7 in `test_op_mm.py` + 3 original + 1 new in `test_funct7_routing.py`. The 4 sibling-owned scaffolds in `test_op_mm.py` (gemm_core x3 from Plan 02 + decode x1 from Plan 03) are untouched.
- All 4 routing-matrix tests in `test_funct7_routing.py` verified end-to-end: WRSPR-collision NOP behavior + MM dispatch chain + Mode 4 firmware_mm_op isolation (only `mxe_accum[tmu_id, curr_id]` mutates; other 63 cells unchanged).

## Task Commits

1. **Task 1: Add MM funct3 constants to encoding.py** -- `a5bcc25` (feat)
2. **Task 2: Implement ops/mm.py with 10 @handler entries + ops/__init__.py patch + WRSPR re-dispatch wiring** -- `edae639` (feat)
3. **Task 3: GREEN-fill 11 scaffold tests in test_op_mm.py + test_funct7_routing.py** -- `7ee15d6` (test)

## Files Created/Modified

- `src/main/python/riscv/gtx/ops/mm.py` (NEW, 148 LOC)
  - Module docstring documents Pitfall F design (no None-key, per-handler rs1==0 guard for symmetry)
  - 10 `@handler(kind='custom0', funct7=0x00|0x01, funct3=0|1|2|3|7, mnemonic='mm[c]_X', mask_funct3=True)` entries
  - Each handler body: 2-3 lines (rs1==0 NOP + delegation to firmware_mm)
- `src/main/python/riscv/gtx/encoding.py` (MODIFIED, +11 lines)
  - 5 new constants: `GTX_F3_MM_S/O/M/V/T = 0, 1, 2, 3, 7` (matches `gtx_npu_disasm.inc:39-50`)
- `src/main/python/riscv/gtx/ops/__init__.py` (MODIFIED, +1 line)
  - `from . import mm  # noqa: F401  -- triggers MM @handler decorators (Plan 04)`
  - `__all__` now includes `"mm"`
- `src/main/python/riscv/gtx/ops/spr.py` (MODIFIED, +24/-6 lines)
  - `wrspr_gem5` rs1!=0 branch now re-dispatches via `npu._custom0.get(0x00, {}).get(funct3)`
  - `rdspr_gem5` rs1!=0 branch now re-dispatches via `npu._custom0.get(0x01, {}).get(funct3)`
  - Updated docstrings cite Plan 04 reasoning
- `tests/gtx/test_op_mm.py` (MODIFIED, +175/-19 lines): 7 scaffolds filled
- `tests/gtx/test_funct7_routing.py` (MODIFIED, +143/-9 lines): 3 originals + 1 new test (test_mode4_firmware_mm_op_routes_to_tmu_curr)
- `tests/gtx/test_spr.py` (MODIFIED, +9/-1 lines): `_fake_npu` grew `_custom0={}` for re-dispatch fallback compatibility
- `tests/gtx/test_dispatch.py` (MODIFIED, +12/-4 lines): `test_custom0_funct7_collision_rs1_nonzero_returns_zero` now uses 1x1x1 dims (was hanging after re-dispatch wired); assertions intact

## Decisions Made

- **Plan deviation: WRSPR/RDSPR re-dispatch.** The plan's CRITICAL routing semantics block correctly identified that a None-key handler at funct7=0x00 would mask MM, and instructed "DO NOT add a None-key handler". But it failed to recognize that `spr.py` from Plan 02 already registers `wrspr_gem5`/`rdspr_gem5` at None-key for funct7=0x00/0x01. Without intervention, MM was unreachable. Three options were considered:
  - (A) Modify `npu.custom0` dispatcher to check funct3-keyed entries first. **Rejected** — touches the cross-cutting dispatch surface; the None-first precedence was an explicit P2 backwards-compat decision.
  - (B) Replace the None-key wrspr_gem5 handler with a funct3-keyed one. **Rejected** — wrspr_gem5 needs to handle the rs1==0 path verbatim per gtx_npu_custom0.cc:56-72.
  - (C) Have wrspr_gem5/rdspr_gem5 re-dispatch when rs1!=0. **Selected** — minimally invasive, matches the existing rs1==0 vs rs1!=0 branching the P2 stub already had, and the docstring already named "P4: firmware_mm_op" as the rs1!=0 fill target.
- **Per-handler `if insn.rs1 == 0: return 0` guard kept despite redundancy.** When MM is reached via the wrspr_gem5 re-dispatch (the only normal path), insn.rs1 is guaranteed non-zero (wrspr_gem5 only re-dispatches when rs1!=0). The plan still requires the per-handler guard. Kept it because (a) the plan's symmetry argument is sound (a future P5 caller could bypass wrspr_gem5 and call MM directly), (b) self-documents the Pitfall F intent at the routing site, (c) one-line per handler so cost is negligible.
- **MM dispatch must NOT touch SPR dicts.** test_custom0_funct7_collision_rs1_nonzero_returns_zero (test_dispatch.py) asserted "no SPR mutation" as a P2 contract; that assertion still holds for MM (mm_engine reads LSPR addresses but never writes them, and writes to L1 bytes only). The test was just hanging on huge synthesized matrix dims; the assertion is intact post-fix.
- **Disasm mnemonic canonicalization (`mm_s` -> `mm.s`).** pyspike's C++ `disasm_insn_t` constructor canonicalizes underscore-separated mnemonics into dot-separated form (`mm_s` -> `mm.s`), matching upstream Spike RISC-V disasm conventions. test_handler_registry_has_all_10_mm_variants checks both forms (canonical for real binding, underscore for offline NamedTuple fallback).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] WRSPR/RDSPR None-key collision unmasked at first verify**
- **Found during:** Task 2 verify step (`python3 -c "...registry walk..."`)
- **Issue:** `spr.py wrspr_gem5/rdspr_gem5` from Plan 02 hold the None inner key at funct7=0x00/0x01; npu.custom0's None-first precedence makes MM funct3 handlers unreachable.
- **Fix:** wrspr_gem5/rdspr_gem5 rs1!=0 branch re-dispatches to per-funct3 MM/MMC handler via `npu._custom0.get(funct7, {}).get(funct3)`. Verified end-to-end via in-line python script before commit.
- **Files modified:** `src/main/python/riscv/gtx/ops/spr.py`
- **Commit:** `edae639`

**2. [Rule 1 - Bug] test_spr.py _fake_npu shim missing _custom0**
- **Found during:** Task 2 verify (post-spr.py edit)
- **Issue:** `tests/gtx/test_spr.py::test_wrspr_gem5_collision_rs1_nonzero_returns_0_no_write` and `test_rdspr_gem5_rs1_nonzero_returns_0_stub` failed with `AttributeError: 'types.SimpleNamespace' object has no attribute '_custom0'` after spr.py started reading `npu._custom0`.
- **Fix:** Added `_custom0={}` to the `_fake_npu` SimpleNamespace shim. Empty dict means re-dispatch falls back to `return 0` (the original stub behavior the tests were validating).
- **Files modified:** `tests/gtx/test_spr.py`
- **Commit:** `edae639`

**3. [Rule 1 - Bug] test_dispatch.py P2-era test hung on huge synthesized dims**
- **Found during:** Task 2 verify (full suite run)
- **Issue:** `test_custom0_funct7_collision_rs1_nonzero_returns_zero` passed `XPR[3]=0x900` as packed-rs1 to a funct7=0x00 + rs1!=0 dispatch. Pre-Plan-04 this returned 0 immediately (P2 stub); Post-Plan-04 it routes to mm_s, which decodes 0x900 as `row_A=0x900, col_A=col_B=0` (promoted to 0x10000) and tried to read a 2304 x 65536 FP16 matrix in pure Python loops -- effectively hung the test runner.
- **Fix:** Test now uses `rs1_packed = (1 << 48) | (1 << 16) | 1` (1x1x1 dims). Original assertions (rc==0, gspr unchanged, lspr[0][0] unchanged) preserved verbatim -- they were testing the contract "MM does not mutate SPRs", which still holds.
- **Files modified:** `tests/gtx/test_dispatch.py`
- **Commit:** `edae639`

## Issues Encountered

- **Pre-existing pytest hang on test_dispatch.py.** Initially manifested as the full-suite pytest run hanging at `test_dispatch.py::test_custom0_funct7_collision_rs1_nonzero_returns_zero`. Spent ~10 min isolating with backgrounded runs before realizing the cause was MY change (Plan 04's WRSPR re-dispatch routed to slow Python MM loops). Resolved by Deviation #3 above. Lesson: when a test file appears to hang post-edit, run that single test with a small timeout first instead of polling the full suite.
- **disasm_insn_t name canonicalization mismatch.** First run of `test_handler_registry_has_all_10_mm_variants` failed because the test expected `'mm_s'` but the registry had `'mm.s'`. Investigation showed the C++ `disasm_insn_t` constructor rewrites `_` -> `.` for canonical Spike disasm conventions. Fixed by branching the expected set on `_RISCV_DISASM_AVAILABLE`.

## Pitfall F Per-handler NOP Audit

| Handler     | rs1==0 NOP guard? | Source line |
| ----------- | ----------------- | ----------- |
| `_exec_mm_s`     | YES | mm.py:46-47 |
| `_exec_mm_o`     | YES | mm.py:55-56 |
| `_exec_mm`       | YES | mm.py:64-65 |
| `_exec_mm_v`     | YES | mm.py:73-74 |
| `_exec_mm_t`     | YES | mm.py:82-82 |
| `_exec_mmc_s`    | YES | mm.py:97-98 |
| `_exec_mmc_o`    | YES | mm.py:106-107 |
| `_exec_mmc`      | YES | mm.py:115-116 |
| `_exec_mmc_v`    | YES | mm.py:124-125 |
| `_exec_mmc_t`    | YES | mm.py:133-134 |

`grep -c "if insn.rs1 == 0:" ops/mm.py` -> 10. Audit clean.

## Pitfall 1 / Pitfall D / L0 BE/LE Verification Results

| Test                               | Pitfall   | Verified Behavior |
| ---------------------------------- | --------- | ----------------- |
| `test_verify_minimal_be_fp16_pairs`      | Pitfall 1 | BE bit-pair compare; FP16 0x3C00 strict-pass on identical files; ULP-1 mismatch fails strict mode |
| `test_exec_mm_o_writes_scalar_to_l0_be`  | gtx_npu_mm.cc:217-218 | sum(A)=10.0 -> FP16 0x4900 -> L0 BE bytes [0x49, 0x00] |
| `test_exec_mm_v_writes_dot_to_l0_le`     | gtx_npu_mm.cc:274-275 | dot(A,B)=70.0 -> FP16 0x5460 -> L0 LE bytes [0x60, 0x54] (asymmetry vs mm_o!) |
| `test_exec_mm_t_writes_transposed`       | Pitfall D | mm_t writes C^T flat to ADDRR; reading as (N, M) shape returns C^T |

## Mode 4 Routing Verification (Note per Plan output)

ROADMAP P4 success #5 ("Mode 4 routes a synthesized firmware_mm_op to the (tmu_id, curr_id) SPU only") is verified end-to-end via the **firmware_mm_op path** (test_mode4_firmware_mm_op_routes_to_tmu_curr). Body asserts: (a) `mxe_accum[1, 5] == 10.0` after mm_o, (b) all other 63 cells unchanged via `np.delete(flat, idx)` snapshot diff.

The `dispatch_4mode -> dispatch_iss_opcode` body extension for `funct7=GTX_OP_MM` (gem5-simplified DISPATCH_MM) is intentionally **DEFERRED to P5/P6** per RESEARCH finding #4 (firmware_mm_op and dispatch_iss_opcode are separate paths). The original test_mode4_routes_to_tmu_curr is preserved as a documented-NOP regression to pin the dispatch_4mode entry-point shape until P5 promotion.

## Final Test Counts

- **Baseline (pre-Plan-04):** 184 passed, 15 skipped
- **After Plan 04:** 195 passed, 5 skipped
- **Delta:** +11 passed, -10 skipped (1 new test added: test_mode4_firmware_mm_op_routes_to_tmu_curr)
- **Remaining 5 skips:** all owned by Plan 05 (4 in test_mm_chain.py + 1 in test_regression_fw_mm.py + 1 -- wait the math doesn't add: counted manually below)
- Actually breakdown after Plan 04:
  - `test_mm_chain.py`: 4 scaffolds, all skipped (Plan 05 territory)
  - `test_regression_fw_mm.py`: 1 scaffold, skipped (Plan 05 territory)
  - = 5 total skipped. Confirmed against `pytest -q` "5 skipped" output.

## Known Stubs

None introduced by this plan. The 5 remaining `pytest.skip` calls in `test_mm_chain.py` and `test_regression_fw_mm.py` are Plan 05 territory (mm chain integration + .elf regression). All Plan 04 acceptance tests pass without skip-bypass.

## Self-Check: PASSED

**Created files exist:**
- `src/main/python/riscv/gtx/ops/mm.py` ✓ (FOUND, 148 LOC, 10 @handler entries)

**Modified files exist + contain expected content:**
- `src/main/python/riscv/gtx/encoding.py` ✓ (`grep -c "GTX_F3_MM_T" encoding.py` -> 1)
- `src/main/python/riscv/gtx/ops/__init__.py` ✓ (`grep -c "from . import mm" __init__.py` -> 1)
- `src/main/python/riscv/gtx/ops/spr.py` ✓ (re-dispatch lines present in both wrspr_gem5 and rdspr_gem5)
- `tests/gtx/test_op_mm.py` ✓ (7 scaffolds populated; 4 sibling-owned untouched)
- `tests/gtx/test_funct7_routing.py` ✓ (4 tests populated)
- `tests/gtx/test_spr.py` ✓ (`_custom0={}` shim addition)
- `tests/gtx/test_dispatch.py` ✓ (1x1x1 dims fix)

**Commits exist (verified via `git log --oneline`):**
- `a5bcc25` ✓ (FOUND -- Task 1: encoding constants)
- `edae639` ✓ (FOUND -- Task 2: ops/mm.py + spr.py re-dispatch + collateral fixes)
- `7ee15d6` ✓ (FOUND -- Task 3: 11 scaffold fills)

**Verification commands all pass:**
- `python3 -c "from riscv.gtx.encoding import GTX_F3_MM_S, GTX_F3_MM_O, GTX_F3_MM, GTX_F3_MM_V, GTX_F3_MM_T; assert (GTX_F3_MM_S, GTX_F3_MM_O, GTX_F3_MM, GTX_F3_MM_V, GTX_F3_MM_T) == (0, 1, 2, 3, 7)"` -> succeeds
- Registry walk: funct7=0x00 funct3 keys [0, 1, 2, 3, 7]; funct7=0x01 funct3 keys [0, 1, 2, 3, 7] (verified)
- `grep -c "if insn.rs1 == 0:" ops/mm.py` -> 10
- `grep -c "is_accumulate=False" ops/mm.py` -> 5 (handler bodies; doc comment shows 5 mm family)
- `grep -c "is_accumulate=True" ops/mm.py` -> 6 (5 handler bodies + 1 in MMC family doc comment)
- `pytest tests/gtx/test_op_mm.py tests/gtx/test_funct7_routing.py --noconftest -o "addopts=" -v` -> 15 passed
- `pytest tests/gtx/ --noconftest -o "addopts=" -q` -> 195 passed, 5 skipped (vs baseline 184/15)

## Next Wave Readiness

Plan 05 (Wave 2) can now wire the strict-mode .elf regression:

- The MM dispatch path is end-to-end live: spike's RoCC trampoline -> npu.custom0 -> wrspr_gem5 re-dispatch (or direct funct3-keyed if a future P5 caller bypasses) -> mm_engine.firmware_mm -> gemm_core kernel -> L1/L0 writeback.
- All 10 MM/MMC variants reachable via funct7=0x00/0x01 + funct3 in {0, 1, 2, 3, 7}.
- Pitfall F NOP safety verified by test_funct7_zero_collision_routing (rs1==0 path NOPs cleanly).
- Mode 4 routing to (tmu_id, curr_id) verified for the firmware_mm_op path used by Phase 4.
- The 4 `test_mm_chain.py` scaffolds + 1 `test_regression_fw_mm.py` scaffold are Plan 05's GREEN-fill targets. The mm_engine.firmware_mm + ops/mm.py @handler interfaces are locked, so Plan 05 can write its tests against this surface immediately.

No blockers. All Pitfall F / D / 1 / B (from Plan 03) contracts enforced and tested.

---
*Phase: 04-mm-subsystem*
*Completed: 2026-05-06*
