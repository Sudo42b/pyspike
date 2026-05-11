# Handoff: Phase 8 NEG/EXP/EXPM1/CUMSUM content fix — instruction trace to localize root cause

## Session Metadata
- Created: 2026-05-11 13:57:55
- Project: /mnt/e/14_NIGHTLY/pyspike
- Branch: main
- Session duration: ~6-8 hours (Phase 8 execution + 5 surgical fixes + OP-by-OP analysis)

### Recent Commits (for context)
- 47fba1b fix(08-NEG): vendor parity for credit counters + OPERAND3/4 stage clearing
- 11cbc66 fix(08-NEG): register credit_ld (0x50) + credit_st (0x51) handlers — unblocks hang
- e7a3e06 fix(08-RELU): register OPSET handler (funct7=0x4A) — flips RELU/GELU_ERF/GELU_QUICK
- 029e2ab fix(08-05): relax strict-mode gate to ULP=1 (B1, CLAUDE.md alignment)
- b2738ae fix(08-05): VTW-03 baseline rerecording + 5x gate timeout baseline-aware

## Handoff Chain

- **Continues from**: None (fresh start)
- **Supersedes**: None

> This is the first handoff for this task.

## Current State Summary

Phase 8 (`/gsd:execute-phase 8`) Wave 2 in progress. The original 6 plans executed: 08-01..08-06 all landed with summary commits. Mid-execution the user expanded P8 scope (rejecting Plan 04's P9 deferral for non-multi-tile fails) and requested OP-by-OP analysis to drive SMOKE_SET_12 from `M=2` (ABS + GELU only) toward the ROADMAP target `M ≥ 12`. After 4 surgical fixes M reached 8 (ABS, GELU, SIGMOID, HARDSIGMOID, LEAKY_RELU, RELU, GELU_ERF, GELU_QUICK). Remaining 4 ops (NEG/EXP/EXPM1/CUMSUM) had a 2-stage problem: stage 1 = hang (resolved by 0x50/0x51 handler registration), stage 2 = content wrong (`exact_matches=0`, `actual_dump` is mostly zero). Stage 2 root cause is **NOT** in dispatch table coverage (vec_size correctly extracted via xs1=0 quirk workaround, NEG sub_op=1 routes to `_apply_unary(0x1D, 1)` which is the correct NumPy negate). Next step: emit per-instruction trace through pyspike to locate where NEG's BANK_R output diverges from expected.

## Codebase Understanding

## Architecture Overview

- **pyspike** = pure-Python RoCC port of GTX NPU C++ functional model. Phase 8 is the v1.1 milestone closing the multi-tile DMA orchestration gap so vendor 84-op `n1s16` regression sweep passes strict-mode against `_ref.txt` golden.
- **Two vendor reference trees** matter:
  - `vendor/gtx_cpp_reference/gtx/` (pyspike's pinned reference — older, flat structure)
  - `~/NIGHTLY/gtx_spike/gtx/` (newer reference — `inc/` + `src/` split, has `gtx_npu_simd.h`, MCAST/TPOSE/FILL/DEBUG opcodes added). The vendor `.elf` test fixtures appear to have been built against the newer tree. pyspike's missing handlers correspond to opcodes added between old → new vendor.
- **SMOKE_SET_12** = ROADMAP P8 success #1 list: {ABS, ADD_VV, MUL_VV, RELU, SIGMOID, GELU, TANH, LEAKY_RELU, SUM, NEG, DIV, EXP} (latter 3 chosen by plan-stage; DIV has no vendor `.elf` so always skipped → max possible M = 11 out of 12 against this filter, but full 84-op sweep target is 12).
- **strict-mode redefinition (P8 B1)**: `compare_hex(strict=True)` was relaxed from `exact_matches == total_fp16` (0-ULP) to `failures == 0` (ULP ≤ 1 + atol ≤ 0.001), matching CLAUDE.md's stated "Bit-exact ULP 허용오차 내" constraint. The prior 0-ULP gate was unreachable for transcendentals due to vendor `std::exp(float)` libm/SIMD build-environment quirks.

## Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/main/python/riscv/gtx/npu.py` | GtxNpu RoCC class — custom0 outer wrapper with OPSET-aware OPERAND3/4 clearing (added this session) | NEG trace point: dispatch entry |
| `src/main/python/riscv/gtx/ops/dma.py` | DMA + credit handlers. Newly added 0x50/0x51 (credit_ld/credit_st) full counter logic | NEG trace point: DMA load/store routing |
| `src/main/python/riscv/gtx/ops/vec.py` | vec dispatch handlers. funct7=0x1D entry routes to `vec_engine.firmware_vec_op` | NEG trace point: 0x1D entry |
| `src/main/python/riscv/gtx/vec_engine.py` | `firmware_vec_op` body — vec_size resolution (`rs1 & 0xFFFF`) + `_apply_unary` for 0x1C/0x1D/0x1E | NEG trace point: BANK_A view + NumPy negate + BANK_R store |
| `src/main/python/riscv/gtx/ops/spr.py` | OPSET handler (funct7=0x4A) — slot 0 → OPERAND3, slot 1 → 0x005 | OPSET stage check |
| `src/main/python/riscv/gtx/_verify.py` | `compare_hex` — strict gate relaxed to ULP-1 in this session | Verify oracle |
| `tests/gtx/test_regression_fw_full_sweep.py` | Vendor 84-op sweep harness. Removed external `within_tolerance==0` assertion | Test harness |
| `tests/gtx/data/baseline_walltime.txt` | VTW-03 baseline — 5104.646s (HAS_NUMBA=False, comment-header format) | VTW-03 closure pending |
| `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc` | Old vendor — GTX_VEC_VNEG = 10 enum, dispatch in `exec_vector_op` | Reference (lines 295-308 = VNEG body) |
| `~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_vec.cc` | NEW vendor — same enum but +280 lines (SIMD, more ops) | Reference (lines 290-308) |
| `~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:1042-1058` | Vendor outer wrapper that clears OPERAND3 / 0x005 after non-OPSET dispatch | Already ported |
| `/mnt/e/14_NIGHTLY/pyspike/test/NEG/n1s16/n1s16_neg.elf` | Vendor pre-built NEG binary (M=0 multi-tile firmware) | Test fixture |
| `/mnt/e/14_NIGHTLY/pyspike/test/NEG/n1s16/data/n1s16_neg_{input,ref}.txt` | Input + golden BE FP16 hex text | Test data |
| `tests/gtx/data/golden_full/neg.hex` | Imported full-region golden (via `scripts/import_vendor_golden.py --full`) | Verify target |
| `.planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md` | Plan 03 investigation verdict: Outcome B (NPU code fix) | Background |
| `.planning/phases/08-multi-tile-dma-parity/08-04-SUMMARY.md` | Multi-tile DMA fix landing (credit.ld.chk handler) | Background |

### Key Patterns Discovered

1. **RoCC funct3 = `{xd, xs1, xs2}`**: assembler emits these bits, but for non-standard mnemonics (`neg.v`, `abs.v`) the actual reg-use semantics diverge from spec. ABS = funct3=0 (all three bits 0, reg used by convention only), NEG = funct3=1 (xs2=1). Both still expect `rs1` to carry vec_size — pyspike correctly reads via direct `proc.state.XPR[insn.rs1]` (xs1=0 quirk workaround per CORE-04).
2. **OPSET special-case**: funct7=0x4A is the ONLY instruction that leaves OPERAND3/OPERAND4 staging slots populated for the next consumer. Vendor's outer `custom0()` wrapper clears both slots after every other dispatch. Ported this session.
3. **Vendor dispatch routing**: For funct7 ≥ 0x08 with no explicit case in custom0.cc switch (e.g., 0x50, 0x51, 0x52, 0x53), vendor's default branch falls through to `dispatch(p, val_rs1, val_rs2, funct7)` after setting `gspr[GSPR_GTX_OPCODE] = f3` AND staging `iss_rs1_shadow / iss_rs2_shadow`. pyspike currently uses explicit per-funct7 handlers — does not need the default-branch routing for the ops covered, but missing handlers fall through to silent NOP (line 156 in npu.py:custom0).
4. **`__set_spm_addr(R, C, B, A)` arg order**: positional `(spm_addr_R, spm_addr_C, spm_addr_B, spm_addr_A)` but firmware body writes to SPRs in **A→B→C→R** order via 4 WRSPRs. Pyspike LSPR_SPM_ADDRA/B/C/R = 0x900/0x901/0x902/0x903 (matches vendor `gtx_csr_v1_0_7.h:232-235`).
5. **deferred DDR stores**: S-loop `__store_cr(L2 → DDR)` enqueues to `npu.deferred_ddr_stores`. Trigger to flush: `end_p` (if `!wsplit_seen`) OR `credit_ld_chk`/`credit_st_chk` when `is_sloop` OR atexit hook. NEG/EXP/EXPM1/CUMSUM emit `__split()` so `wsplit_seen=True` → end_p does NOT flush. Flush relies on credit_ld_chk inside the S-loop block.

## Work Completed

### Tasks Finished

- [x] Plan 08-01: Tile-2 RED-state proof + state-reset audit (committed `6e1bdad`)
- [x] Plan 08-02: Vendor asset wire-up (`_find_elf` 3-tier, `import_vendor_golden.py --all`, firmware/ wheel exclusion) — commits `759cfa7..f859d31`
- [x] Plan 08-03: Dump-size investigation — verdict Outcome B (NPU code fix needed) — commits `25c54a5..4c93c2d`
- [x] Plan 08-04: Multi-tile DMA surgical fix — credit.ld.chk (0x52) handler — commits `8660c89..da62177`. Effect: ABS 196609 lines byte-exact (was 389124 differing)
- [x] Plan 08-06: VTW-04 docs closure — README 4-contract, ARCHITECTURE BE/LE, STATE/ROADMAP sync — commits `d714121..8113295`
- [x] VTW-03 baseline rerecording (Plan 08-05 stage 1) — 5104.646s captured in HAS_NUMBA=False venv (commit `b2738ae`)
- [x] Plan 08-05 conftest parser patch — multi-line/comment-tolerant baseline_walltime.txt (commit `59d7078`)
- [x] test_njit_perf.py timeout fix — hardcoded 600s → `max(int(baseline*1.5), 600)` baseline-aware
- [x] B1 strict-mode relaxation (compare_hex + test_regression_fw_full_sweep within_tolerance assertion) — commit `029e2ab`
- [x] OPSET handler (funct7=0x4A) — commit `e7a3e06` (M=5 → M=8)
- [x] credit_ld/credit_st handlers (0x50/0x51) — hang resolved (commit `11cbc66`)
- [x] vendor parity: counter logic + OPERAND3/4 clearing (commit `47fba1b`)

## Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `tests/gtx/data/baseline_walltime.txt` | `4.5` → multi-line comment header + `5104.646` | VTW-03 HAS_NUMBA=False baseline |
| `tests/gtx/test_njit_perf.py:108` | hardcoded `timeout=600` → baseline-aware `max(int(baseline*1.5), 600)` | 600s cap unreachable with real 5104s baseline |
| `src/main/python/riscv/gtx/_verify.py:38-103,150-160,180-194` | `strict=True` gate exact==total → failures==0; docstring + CLI epilog updated | B1 — CLAUDE.md ULP-1 alignment |
| `tests/gtx/test_regression_fw_full_sweep.py:425+` | Removed external `assert within_tolerance == 0` | compare_hex's `passed` is now authoritative ULP-1 gate |
| `src/main/python/riscv/gtx/ops/spr.py:43-67` | Added OPSET (funct7=0x4A) handler — slot 0 → OPERAND3, slot 1 → 0x005 | RELU 76% fail root cause |
| `src/main/python/riscv/gtx/encoding.py:76-77` | Added GTX_ISS_F7_CREDIT_LD (0x50), GTX_ISS_F7_CREDIT_ST (0x51) | NEG/EXP/EXPM1/CUMSUM hang resolution |
| `src/main/python/riscv/gtx/ops/dma.py:309-346` | Added _credit_ld / _credit_st with full S-loop/T-loop counter logic | vendor parity (1:1 port of dispatch.cc:950-974) |
| `src/main/python/riscv/gtx/npu.py:145-176` | custom0 outer wrapper clears OPERAND3 + 0x005 after non-OPSET dispatch | vendor parity (gtx_npu_custom0.cc:1042-1058) |
| `src/main/python/riscv/gtx/npu.py:65-77,114-117` | GtxNpu._credit_ld / ._credit_st int32 [NEST][SPU] state + reset zeroing | counter state holder |

## Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| **B1 strict-mode relaxation** | A) per-op strict override, B) ROADMAP wording change, **C) global strict→failures==0** | C wins: aligns with CLAUDE.md "ULP 1 + atol 0.001" original constraint. SIGMOID/HARDSIGMOID/LEAKY_RELU PASS in 1 commit. No per-op exception machinery needed. |
| **Reject P9 deferral, expand P8 scope** | A) close P8 at M=2, B) **expand scope to smoke 12** | User chose B for milestone completeness. Trade-off accepted: longer P8 vs cleaner v1.1 ship. |
| **credit_ld/credit_st: full vendor counter logic (NOT NOP)** | A) keep as NOP, **B) port full counter logic** | User feedback ("제대로 구현 안된 것 같은데"). B preserves vendor 1:1 parity for any future check-path coupling. Functional behavior unchanged (counters unobserved in sequential model) but code is no longer a lie. |
| **OPERAND3/4 clearing in custom0 wrapper** | A) per-handler clear, **B) outer wrapper clear** | B matches vendor `gtx_npu_t::custom0` outer wrapper exactly. Single point of clearing, OPSET-aware exception. |
| **VTW-03 timeout baseline-aware** | A) bump to fixed 7200s, **B) compute `max(baseline*1.5, 600)`** | B scales with real measurement; future-proof. |

## Pending Work

## Immediate Next Steps

1. **Run instruction trace on NEG** to localize where BANK_R output goes wrong (the next agent's PRIMARY task). Specifically:
   - Add temporary instrumentation in `vec_engine.py:firmware_vec_op` (just before `_l1_view_addr(npu, nest, spu, addr_r, vec_size)[:] = result`) to log: nest/spu, addr_a, addr_r, vec_size, first 4 bytes of `view` input, first 4 bytes of `result`. Use `print(...)` to stderr — D-03 prohibits `GTX_DEBUG_*` env vars but a SHORT-LIVED test-only print is acceptable per "instrumentation = test-side snapshots only" (08-CONTEXT D-03 wording).
   - Run NEG single-op: `GTX_VENDOR_TEST_DIR=/mnt/e/14_NIGHTLY/pyspike/test/ python -m pytest tests/gtx/test_regression_fw_full_sweep.py -k 'NEG and not LEAKY_RELU and not GELU' -v --no-cov --timeout=180 -s` (`-s` to see prints).
   - Compare: does BANK_A view contain input row bytes? Is `result` actually the negate? Where does it diverge — at the L1 view, the unary op, or the BANK_R writeback?
   - **REMOVE the prints before commit** (or wrap in `if os.environ.get("PYSPIKE_NEG_TRACE")` only for one-shot, do not commit if env-var route is taken).
2. **Verify deferred-store flush ordering for single-tile firmware**: NEG calls `__store_cr` in S-loop (deferred) BEFORE the T-loop fills L2_RESULT. Check if the deferred-store snapshot policy (`plan_has_tloop` snapshot-vs-ref decision per 03-RESEARCH "Deferred-Store Flush Trigger") is correctly capturing the post-T-loop state. Read `src/main/python/riscv/gtx/dma_engine.py:DeferredDdrStore` + the snapshot/ref selection logic.
3. **If trace doesn't localize within 30 min**: consider applying the **D1) vendor reference upgrade** option (pyspike's pinned `vendor/gtx_cpp_reference/gtx/` → `~/NIGHTLY/gtx_spike/gtx/` newer tree sync). gtx_npu_act.cc + 527 lines, gtx_npu_vec.cc + 280 lines — could include semantic fixes that explain the 4 remaining fails.
4. **Once NEG passes**: replay EXP/EXPM1/CUMSUM. They may share the same root cause (all single-tile firmware), or each may need separate analysis. Commit per-op as discovered.
5. **Verify no regression**: re-run SMOKE_SET_12 to confirm M=8 stable (ABS, GELU, SIGMOID, HARDSIGMOID, LEAKY_RELU, RELU, GELU_ERF, GELU_QUICK).
6. **Final closure** (once M ≥ 12 reached): re-run 5x gate `test_vendor_sweep_walltime_5x` with the patched 7200s timeout; if PASS → VTW-03 closure → Plan 08-05 final commit → spawn gsd-verifier → `/gsd:complete-milestone v1.1`.

### Blockers/Open Questions

- [ ] Why does NEG dump zero output while ABS (same dispatch path, same 0x1D family) passes? Stage 1 (hang) is fixed; stage 2 root cause unknown.
- [ ] Are EXP/EXPM1/CUMSUM blocked by the same root cause as NEG, or do they need separate analysis?
- [ ] Should pyspike's vendor reference be upgraded to `~/NIGHTLY/gtx_spike/gtx/` (option D1)? It's a larger but more thorough fix path.
- [ ] DIV vendor `.elf` doesn't exist → graceful skip. Max achievable M against SMOKE_SET_12 filter is 11 (out of 12). For ROADMAP P8 #1's `M ≥ 12`, we need full 84-op sweep with M ≥ 12 (the count is across the full sweep, not just smoke-12).

### Deferred Items

- [ ] Plan 08-05 final SUMMARY.md + STATE/ROADMAP sync (depends on VTW-03 5x gate PASS)
- [ ] gsd-verifier invocation for Phase 8 closure
- [ ] `/gsd:complete-milestone v1.1` (depends on Phase 8 closure)
- [ ] If Outcome A path applies for some ops (no NPU code change needed), revisit `.planning/seeds/p9-vendor-sweep-non-multi-tile-bugs.md` to formally cancel that seed.

## Context for Resuming Agent

## Important Context

**The user is operating in `/gsd:execute-phase 8` flow** (see `~/.claude/get-shit-done/workflows/execute-phase.md`). They reject premature P9 deferral and want OP-by-OP analysis to drive M from 8 → 12+. Do NOT silently fall back to "close P8 at current M and defer" — the user has already considered and rejected that.

**Two vendor reference trees coexist**:
1. `vendor/gtx_cpp_reference/gtx/` — pyspike's pinned reference (flat, old)
2. `~/NIGHTLY/gtx_spike/gtx/` — newer (inc/src split, +1100 lines across .cc files, has SIMD header, MCAST/TPOSE/FILL/DEBUG opcodes added between old → new)

The vendor `.elf` fixtures (`/mnt/e/14_NIGHTLY/pyspike/test/<OP>/n1s16/`) appear to have been built against the newer tree. So far all surfaced gaps (CREDIT_LD_CHK, OPSET, CREDIT_LD, CREDIT_ST, OPERAND3/4 clearing) were already in BOTH old and new trees — they were just missing from the pyspike PORT. Don't assume the gap is always between old vs new vendor; it's usually pyspike ← vendor lag.

**SMOKE_SET_12 current status (after this session)**:
```
M = 8 PASS:  ABS, GELU, SIGMOID, HARDSIGMOID, LEAKY_RELU, RELU, GELU_ERF, GELU_QUICK
M = 4 FAIL:  NEG, EXP, EXPM1, CUMSUM (all stage 1 hang resolved, stage 2 content wrong)
M = 1 SKIP:  DIV (vendor .elf missing)
```

**The 4 failing ops share a pattern**:
- All emit identical custom0 funct7 inventory (0x40 DMA, 0x49 WRSPR, 0x4A OPSET, 0x50/0x51/0x52 credits, + 0x1D/0x1C for the compute kernel).
- All have single-tile firmware (no `MAX_SHARED_DMA_BYTES` loop), unlike ABS which uses multi-tile pattern.
- All currently produce `actual_dump` mostly zero (e.g., NEG: `exact_matches=0, within_tolerance=22, failures=17386, total_fp16=17408`).
- Identical pattern → likely shared single-tile firmware data-path bug in pyspike.

**xs1=0 quirk is already handled correctly** in `vec_engine.py:firmware_vec_op` line 100 (`rs1 = int(proc.state.XPR[insn.rs1])` bypasses spike's -1 marshalling). Don't waste time on this.

**Strict mode is RELAXED**: `compare_hex(strict=True)` now means `failures == 0` (ULP ≤ 1). If you re-tighten this you'll break SIGMOID/HARDSIGMOID/LEAKY_RELU. Don't.

## Assumptions Made

- Numba is installed in the active venv after VTW-03 baseline recording (`pip install numba` re-installed 0.65.1; HAS_NUMBA=True). If you uninstall again, restore with `python -m pip install numba`.
- `GTX_VENDOR_TEST_DIR=/mnt/e/14_NIGHTLY/pyspike/test/` must be set in env or test invocation for vendor `.elf` resolution (3rd-tier `_find_elf` candidate).
- Running on WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2) — most timeouts/cap values are tuned for this host's speed (~85 min full HAS_NUMBA=False sweep).
- All commits use `--no-verify` per the Phase 8 parallel-executor discipline. Pre-commit hooks are validated once at wave completion (not after each commit). Continue this pattern.

## Potential Gotchas

- **DON'T** introduce production env vars like `GTX_DEBUG_TILE_TRACE` or `_debug.py` modules — D-03 forbids this. Test-side prints in pytest fixtures or temporary `print()` in `vec_engine.py` are OK but must be removed before commit.
- **DON'T** modify `tests/gtx/data/firmware/` (wheel-excluded by D-07 + sentinel test `test_wheel_excludes_firmware_dir` in `tests/gtx/test_wheel_data_present.py`). Place new test fixtures elsewhere.
- **DON'T** re-tighten `compare_hex` strict mode (see B1 decision above).
- **DON'T** assume vendor's `~/NIGHTLY/gtx_spike/gtx/` is automatically authoritative — pyspike pins `vendor/gtx_cpp_reference/gtx/`. Reference both when diffing, but commits to pyspike code must align with the pinned old vendor unless explicitly upgrading.
- **Spike subprocess timeout = 600s per op** (in `tests/gtx/test_regression_fw_full_sweep.py`). If your trace adds heavy logging that slows pyspike, individual ops can hit the cap. Print sparingly.
- **`MockProcessor` (in `tests/gtx/_mocks.py`) does NOT trigger the real custom0 dispatch** — it directly calls handler functions. If you write a unit-test reproduction of NEG, you'll bypass the outer wrapper's OPERAND3/4 clearing. Use the full subprocess `.elf` flow OR call `npu.custom0(proc, insn, xs1, xs2)` explicitly (not the handler).
- The user previously had a hang-state spike process (PID 2211797, MUL_MAT_ID, 11hr+ R-state) and a separate GTX_ISS reference simulator (PID 772037, 45hr+). The user's manual kills cleared the orphans. If running multi-op sweep and seeing system slowness, check `ps -auxf | grep -E 'pytest|spike|n1s16_' | grep -v grep`.

## Environment State

### Tools/Services Used

- Python 3.10.12 (system + WSL2)
- pytest 9.0.1 + pytest-timeout 2.4.0 + pytest-benchmark 5.1.0
- numba 0.65.1 / llvmlite 0.47.0 (re-installed mid-session — see Assumptions)
- riscv64-unknown-elf-objdump at `/opt/riscv/bin/` (used for instruction inventory analysis)
- spike binary at `/home/sw.lee/.local/bin/spike` (pyspike-bundled via pip)
- `pyspike` CLI at `/home/sw.lee/.local/bin/pyspike`

### Active Processes

- None expected. If `ps -auxf | grep -E 'pytest|spike|n1s16_'` shows lingering processes, kill them before tracing.

### Environment Variables

- `GTX_VENDOR_TEST_DIR` (must be set to `/mnt/e/14_NIGHTLY/pyspike/test/` for vendor `.elf` resolution)
- `GTX_DDR_INIT`, `GTX_DDR_DUMP`, `GTX_DDR_DUMP_ADDR`, `GTX_DDR_DUMP_SIZE`, `GTX_DDR_REVERSED`, `GTX_NO_EXIT` — set INLINE by `test_regression_fw_full_sweep.py:_run_subprocess` per `is_vendor_elf` branch (don't export globally)
- `PYSPIKE_NEG_TRACE` — proposed for instrumentation gating; not yet committed

## Related Resources

- `.planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md` — 13 locked decisions D-01..D-13 (especially D-03 instrumentation policy, D-04 surgical scope, D-09 tile-2 test design)
- `.planning/phases/08-multi-tile-dma-parity/08-RESEARCH.md` — vendor C++ ↔ pyspike Python 1:1 diff matrix, 7-hypothesis ranking (Hypothesis 5 NEW = harness-side, now superseded by direct credit_ld_chk / OPSET fixes)
- `.planning/phases/08-multi-tile-dma-parity/08-03-INVESTIGATION.md` — Plan 03 verdict + Outcome A/B decision tree
- `.planning/phases/08-multi-tile-dma-parity/08-04-SUMMARY.md` — Multi-tile DMA fix landing report (credit.ld.chk root cause)
- `.planning/phases/08-multi-tile-dma-parity/08-06-SUMMARY.md` — VTW-04 docs closure report
- `.planning/seeds/p9-vendor-sweep-non-multi-tile-bugs.md` — P9 deferral seed (now revoked by user; keep file but don't action)
- `tests/gtx/data/firmware/README.md` — D-08 4-contract documentation (BE/LE FP16, GTX_DDR_REVERSED auto-application, vendor .elf import, _find_elf priority)
- `~/NIGHTLY/gtx_spike/gtx-firmware/include/gtx/intrinsics/intrin_level1.h:556` — `__neg_v` macro definition (`neg.v %[r1]` asm)
- `~/NIGHTLY/gtx_spike/gtx-firmware/gtx/intrinsics/intrin_level2.c:76` — `__set_spm_addr` definition (4-WRSPR sequence)
- `~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_vec.cc:290-308` — vendor GTX_VEC_VNEG body (newer tree, 1:1 with old)
- `~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:1042-1058` — vendor outer wrapper (already ported)
- `~/NIGHTLY/gtx_spike/gtx/src/gtx_npu_dispatch.cc:950-974` — vendor credit_ld/credit_st full counter logic (already ported)

---

**Security Reminder**: No secrets in this file. All `GTX_*` env vars are project-local config, not credentials.
