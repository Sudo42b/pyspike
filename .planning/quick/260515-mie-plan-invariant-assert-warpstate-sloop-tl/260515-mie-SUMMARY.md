---
phase: quick-260515-mie
plan: 01
subsystem: gtx/unit/context (warp_state + control)
status: stopped-at-gate-1
tags:
  - gtx
  - warp-state
  - plan-invariant
  - assert-cleanup
  - cleanup-arc-step-3
dependency_graph:
  requires:
    - WarpState dataclass (warp_state.py)
    - _do_* helpers (control.py)
  provides:
    - PLAN-lifetime invariant enforcement (one shared + one thread section per plan)
    - fail-fast assertions on 6 plan-structure transitions
  affects:
    - tests/gtx/test_custom_dispatch_chain.py::test_end_to_end_custom0_and_custom1_return_int  # STALE — see Findings
tech_stack:
  added: []
  patterns:
    - "fail-fast assert with state-rich error message"
    - "PLAN-lifetime sentinel cleared at start_p; process-lifetime sentinel (wsplit_seen) preserved"
key_files:
  created: []
  modified:
    - src/main/python/riscv/gtx/unit/context/warp_state.py
    - src/main/python/riscv/gtx/unit/context/control.py
decisions:
  - "PLAN-lifetime sentinels reset at start_p AND defensively at end_p (lifecycle clarity)"
  - "end_p assert fires BEFORE clearing is_ploop and BEFORE flush_deferred_ddr_stores() — preserves existing P3 wsplit_seen branch ordering"
  - "end_t assert fires BEFORE tloop_buffer flush — stray end_t must not drain a buffer"
  - "STOPPED at Gate 1 instead of reverting unilaterally — smoke regression is a STALE-TEST signal, not a production-correctness signal; user owns the resolution"
metrics:
  duration_seconds: 234
  commits: 2
  files_modified: 2
  invariant_cases_passing: 9  # inline 9-case test in Task 2 verify block
  smoke_tests_passing: 25
  smoke_tests_failing: 1
  abs_elf_gate: not-run-blocked-by-gate-1
completed: 2026-05-15
---

# Quick 260515-mie: Plan Invariant Assert (WarpState sloop_seen_in_plan / tloop_seen_in_plan) Summary

One-liner: Added PLAN-lifetime sentinels + converted 6 `_do_*` helpers from silent-overwrite to assert-guarded; stopped at Gate 1 due to a stale test that synthesizes `end_p` without matching `start_p`.

## What Changed

**2 files, 7 modification sites:**

1. `src/main/python/riscv/gtx/unit/context/warp_state.py` (commit `45d72f1`)
   - +2 fields: `sloop_seen_in_plan: bool = False`, `tloop_seen_in_plan: bool = False`
   - `reset()` now also clears both new sentinels; `wsplit_seen` preserved as process-lifetime
   - Module docstring updated

2. `src/main/python/riscv/gtx/unit/context/control.py` (commit `ed92898`)
   - `_do_startp`: assert `not is_ploop` (no nested plans) + clear PLAN-lifetime sentinels on plan entry
   - `_do_endp`: assert `is_ploop` (balanced); load-bearing order preserved: `is_ploop=False → flush_deferred_ddr_stores()` (when `!wsplit_seen`) → sentinel cleanup
   - `_do_startt`: assert `not is_tloop` (no nesting) + assert `not tloop_seen_in_plan` (one thread section per plan); GTX_TLOOP_DISABLE env override preserved
   - `_do_endt`: assert `is_tloop` (balanced); assert fires BEFORE tloop_buffer flush
   - `_do_starts`: assert `not is_sloop` (no nesting) + assert `not sloop_seen_in_plan` (one shared section per plan)
   - `_do_ends`: assert `is_sloop` (balanced)

All asserts include the relevant flag state + `tmu_id`/`spu_id`/`gdmac_id` so vendor-firmware bugs surface fast in regression logs.

## Verification Status

### Task 1 inline verify — PASS
`uv run python -c "..."` confirmed: new sentinels default False; `reset()` clears them; `wsplit_seen` survives `reset()` (process-lifetime).

### Task 2 inline 9-case invariant test — PASS

All 9 scenarios pass:
1. Happy path: `start_p → start_s → end_s → start_t → end_t → end_p`
2. Nested `start_p` → AssertionError
3. Two `start_s` in same plan → AssertionError
4. Two `start_t` in same plan → AssertionError
5. Nested `start_s` → AssertionError
6. Nested `start_t` → AssertionError
7. Stray `end_p` → AssertionError
8. Stray `end_s` → AssertionError
9. Stray `end_t` → AssertionError
10. Cross-plan sentinel reset: plan N+1 must accept a fresh `start_s`/`start_t` → PASS

### Gate 1 — Smoke (regression observed, STOPPED per plan)

```
tests/gtx/test_custom_dispatch_chain.py  — 1 FAILED, 8 PASSED
tests/gtx/test_custom0_smoke.py          — PASSED
tests/gtx/test_fsm_smoke.py              — PASSED
tests/gtx/test_csr_registry_chain.py     — PASSED
                                  TOTAL: 25 passed, 1 failed in 12.86s
```

**The one failure:** `test_end_to_end_custom0_and_custom1_return_int` at `tests/gtx/test_custom_dispatch_chain.py:154`.

### Gate 2 — ABS .elf regression — NOT RUN (blocked by Gate 1)

Per plan instructions, Gate 2 is gated on Gate 1 PASS. Halted before invoking ABS.

## Findings

### Finding 1 — Stale smoke test exercises invariant violation as test setup

**Test:** `tests/gtx/test_custom_dispatch_chain.py:141-155 :: test_end_to_end_custom0_and_custom1_return_int`

**Pattern (intent):** dispatch chain returns `int` (RoCC `reg_t` contract). Test synthesizes two isolated insns:
- `insn0`: `custom0` with `funct=GTX_ISS_F7_OPSET, rs1=2, rs2=3` → routes through `run_pipeline`
- `insn1`: `custom1` with reconstructed `funct3 = (xd<<2)|(xs1<<1)|xs2 = 7 = END_P`

**Failure trace (verbatim):**

```
src/main/python/riscv/gtx/unit/context/control.py:81: AssertionError
E       AssertionError: end_p without matching start_p (is_ploop=False, tmu_id=0)
```

Call stack: `gtx_npu.custom1(...)` → `run_pipeline(custom1)` → `_STATE_EXECUTE` → `endp` handler → `_do_endp` → new assert (`assert npu.warp.is_ploop`) trips because the test never called `start_p` first.

**Diagnosis:** The test was authored under the silent-overwrite semantics. It synthesizes an `end_p` insn in isolation purely to verify the dispatch chain returns `int` — it does not exercise the plan invariant intentionally. With the new asserts, this synthesis is now invalid because it violates the firmware invariant the asserts encode.

**Audit miss:** The planner's audit note at Task 3 covered direct flag manipulation:
> "test_custom_dispatch_chain.py:166 — sets `gtx_npu.warp.is_tloop = True` directly, does NOT call `_do_startt` → SAFE (assert is in `_do_startt` only)."

But the audit MISSED the `endp`-via-dispatch synthesis at line 154, which DOES go through `_do_endp` and so DOES trip the new assert.

**This is NOT a real-firmware invariant violation** — it's a test-only artifact. The test is asking the dispatch chain "given an `end_p` insn, do you return `int`?" without bothering to set up the FSM state that real firmware would have. Pre-this-plan, the silent-overwrite let the test get away with that lazy setup.

**NOT REVERTED** — awaiting user decision per plan rollback rule. Production code (warp_state.py + control.py) is correct; the test is stale and should either:
1. Be updated to drive `start_p` first, then `end_p` (canonical plan structure).
2. Be replaced with a direct `_do_endp` call after manually setting `npu.warp.is_ploop = True` (mirrors the line-166 direct-flag pattern that the audit already accepted).
3. Be split into two tests: one for `custom0` return-int (already works), one for full `start_p → end_p` round-trip via dispatch.

### Finding 2 — Plan rollback rule has slight internal tension

The plan says both:
- "If smoke tests (Task 3 gate 1) regress, that IS a revert signal" (rollback_rule block)
- "Do not modify the production code to make them pass — instead capture the trace and stop." (Gate 1 action body)

Followed the latter (procedural directive): captured trace, stopped, escalated to user. Did NOT revert unilaterally because:
1. The 9-case invariant test (`uv run python -c ...`) confirms the production code is semantically correct.
2. The smoke "regression" is a stale-test artifact (test invokes the exact pattern the assert is designed to catch), not a production-behavior regression.
3. Reverting would lose the work; user can trivially update the test in a follow-up commit.

## Memory Updates

No new memory note yet — the failure is a stale test pattern, not a vendor-firmware invariant violation. If user opts for relaxation (e.g., make `end_p` without `start_p` a warning instead of an assert), a memory note will be added at that point.

## Commits

- `45d72f1` — refactor(gtx): WarpState plan-lifetime sentinels (sloop/tloop_seen_in_plan)
- `ed92898` — refactor(gtx): silent-overwrite → assert on _do_startp/s/t + _do_endp/s/t

## Followups (deferred to user decision)

1. **Update or remove the stale test** at `tests/gtx/test_custom_dispatch_chain.py:141-155`. Suggested rewrite: drive `start_p` then `end_p` via the dispatch chain, or mirror the line-166 direct-flag pattern (set `gtx_npu.warp.is_ploop = True` before dispatching the `end_p` insn). Once that test is green, re-run Gate 1 (must pass clean) → then run Gate 2 (ABS .elf byte-exact).
2. **Run Gate 2 (ABS .elf strict)** once Gate 1 is clean: `PYTEST_ELF_REGRESSION=1 uv run pytest tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=600 -v 2>&1 | tail -30`. If ABS passes byte-exact, the plan invariant holds in real vendor firmware and the cleanup arc step closes. If ABS trips a new assert, capture `(plan_no, section_pattern, firmware_op_id)` per the original rollback rule — DO NOT revert.
3. **Extend regression to other vendor ops** (GELU, ADD_VV, MUL_VV, SIGMOID etc.) once ABS clean — they currently pass under silent-overwrite; the asserts add coverage with zero happy-path cost.
4. **Stale-test sibling audit**: `tests/gtx/test_deferred_store.py` already broken per STATE.md last_activity (5 breakages, separately deferred). Once both files are rewritten, audit for any other `_do_*` callers in tests that assume silent semantics.

## Self-Check: PASSED

Files verified to exist:
- `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/context/warp_state.py` — FOUND
- `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/context/control.py` — FOUND

Commits verified:
- `45d72f1` — FOUND
- `ed92898` — FOUND
