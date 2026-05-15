---
phase: quick-260515-mie
plan: 01
subsystem: gtx/unit/context (warp_state + control)
status: reverted-abs-pre-existing-broken
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

---

## Continuation (post-Gate 1 resume, 2026-05-15)

### Gate 1 stale test fix

`tests/gtx/test_custom_dispatch_chain.py:152` `test_end_to_end_custom0_and_custom1_return_int` — option 2 (Followups #1) 적용:
end_p insn dispatch 전에 `gtx_npu.warp.is_ploop = True` 한 줄 추가 (line-166 직접 flag 패턴 차용). dispatch chain의 int 반환만 검증하는 test이므로 plan invariant 부분은 우회.

**Gate 1 재실행: 26/26 PASS** (was 25 passed / 1 failed).

### Gate 2 첫 시도 — invariant strict 가 ABS .elf 트립

`PYTEST_ELF_REGRESSION=1 uv run pytest tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=600`:

```
AssertionError: second thread section in same plan — invariant violation
  (tmu_id=0, new spu_id=1)
At: _do_startt → startt handler → dispatch.wrapped → state_execute → run_pipeline → custom1
```

ABS firmware가 한 plan(`start_p..end_p`) 안에 thread section을 **여러 번**(SPU 수만큼) emit. 사용자 초기 invariant "thread 1번 per plan"이 vendor 실제 firmware보다 over-strict로 판명.

**해석 — 실제 vendor 구조**: 한 plan = 한 NEST의 full cycle. shared section은 plan당 1번(NEST 단위 load), thread section은 SPU별로 NEST 안 16개까지 반복(GTX_SPU_NUM).

메모리에 기록: `project_plan_invariant_discovery.md`.

### Invariant 완화 (option B 적용)

사용자 결정: "sloop 1번 강제 유지, tloop은 GTX_SPU_NUM까지 허용".

**변경 (uncommitted 상태로 Gate 2 검증 중):**

- `warp_state.py`: `tloop_seen_in_plan: bool` → `tloop_count_in_plan: int = 0`. `reset()`도 counter 0으로.
- `control.py`:
  - `_do_startp`: `tloop_seen_in_plan = False` → `tloop_count_in_plan = 0`
  - `_do_startt`: `assert not tloop_seen_in_plan` → `assert tloop_count_in_plan < GTX_SPU_NUM` + `tloop_count_in_plan += 1`
  - `_do_endp`: defensive cleanup도 `tloop_count_in_plan = 0`
- `sloop_seen_in_plan`은 그대로 유지 (shared section은 plan당 1번 강제).

### Gate 2 재실행 (1200s timeout)

`PYTEST_ELF_REGRESSION=1 PYTEST_ELF_TIMEOUT=1200 uv run pytest tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=1200`:

- 첫 시도(600s timeout): AssertionError 사라짐 (완화 성공), 다만 70 tiles까지 진행 후 timeout (8s/tile × 96 tiles ≈ 13min 필요).
- 재실행 진행 중.

### Gate 2 재실행 결과 — byte-mismatch

`PYTEST_ELF_REGRESSION=1 PYTEST_ELF_TIMEOUT=1200 ...` 13분 39초 완주, 97 tiles 진행 후 **byte-mismatch**:
```
Failed: abs: fp16 mismatch beyond ULP=1 / atol=0.001 vs n1s16_abs_ref.txt
  line 1 fp16[0]: ref=0x5837 (134.875) dump=0x0000 (0.0) ulp=22583
```

stderr에 AssertionError **0건** — 새 assert는 발화 안 함. logic 변경도 없음(추가는 모두 assert + sentinel 추가, 기존 path 변경 없음). 그런데 ABS first byte부터 mismatch (`dump=0.0`).

### Bisect로 원인 격리

옵션 A 따라 plan invariant 변경 모두 revert해서 baseline 확인:

```
725b2aa Revert "refactor(gtx): silent-overwrite → assert on _do_startp/s/t + _do_endp/s/t"
15a9d19 Revert "refactor(gtx): WarpState plan-lifetime sentinels (sloop/tloop_seen_in_plan)"
```

**Revert 후 ABS 재실행 결과 — 정확히 같은 byte-mismatch**:
```
line 1 fp16[0]: ref=0x5837 (134.875) dump=0x0000 (0.0) ulp=22583
```

→ **결론: plan invariant 변경은 ABS와 완전히 무관**. ABS는 이 quick task 시작 전부터 broken 상태였음. cleanup arc commits (b464bb4, 765d7fb 등) 또는 그 이전 어디선가 ABS byte-exact가 깨졌고 측정 누락.

### 최종 상태

- Plan invariant 변경 모두 main에서 revert됨 (`725b2aa`, `15a9d19`)
- production code는 quick task 시작 전 상태(2ec3fab 시점)와 동등
- Plan invariant 작업은 **abandoned가 아니라 보류** — ABS broken root cause 식별 후 별도 quick task로 재시도 가능 (assert/sentinel 자체는 ABS에 영향 없음을 측정으로 확인)

### Followups (재정리)

1. **ABS regression root cause debug** (최우선) — `/gsd:debug`로 cleanup arc commits bisect. 가장 유력 candidate: `b464bb4` (single-source SPR addresses). `dump=0.0` first byte부터 = store path 자체 broken 의미. SPR 변경이 dispatch/store path에 영향 가능성.
2. **Plan invariant 재시도** — ABS broken fix 후 별도 quick task. 옵션 B 완화(sloop strict + tloop counter ≤ GTX_SPU_NUM) 그대로 land 가능 — 이미 ABS와 독립임을 측정으로 확인.
3. **Extend regression** (Followups #3 그대로 보류): GELU/ADD_VV/MUL_VV/SIGMOID.
4. **Stale-test sibling audit** (Followups #4 그대로 보류): test_deferred_store.py 재작성.
