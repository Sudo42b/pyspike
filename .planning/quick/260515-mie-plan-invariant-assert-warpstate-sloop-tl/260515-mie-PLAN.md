---
phase: quick-260515-mie
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/context/warp_state.py
  - src/main/python/riscv/gtx/unit/context/control.py
autonomous: true
requirements:
  - INV-PLAN-01  # one shared section per plan
  - INV-PLAN-02  # one thread section per plan
  - INV-NEST-01  # no nested plan/section
  - INV-BAL-01   # no end_* without matching start_*

must_haves:
  truths:
    - "Firmware emitting `start_s ... start_s` within one plan trips an AssertionError"
    - "Firmware emitting `start_t ... start_t` within one plan trips an AssertionError"
    - "Firmware emitting `start_p ... start_p` (nested plan) trips an AssertionError"
    - "Firmware emitting `end_s` without preceding `start_s` trips an AssertionError"
    - "Firmware emitting `end_t` without preceding `start_t` trips an AssertionError"
    - "Firmware emitting `end_p` without preceding `start_p` trips an AssertionError"
    - "Existing smoke tests (custom_dispatch_chain, custom0_smoke, fsm_smoke, csr_registry_chain) PASS unchanged"
    - "ABS .elf single-case regression (PYTEST_ELF_REGRESSION=1) PASSes byte-exact OR fails loudly with documented (plan_no, section_pattern) context"
    - "PLAN-lifetime sentinels (sloop_seen_in_plan, tloop_seen_in_plan) reset at every start_p, NOT at SPLIT/JOIN"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/context/warp_state.py"
      provides: "WarpState dataclass with 2 new PLAN-lifetime sentinel fields"
      contains: "sloop_seen_in_plan"
    - path: "src/main/python/riscv/gtx/unit/context/warp_state.py"
      contains: "tloop_seen_in_plan"
    - path: "src/main/python/riscv/gtx/unit/context/control.py"
      provides: "6 assert-guarded _do_* helpers (startp/endp/starts/ends/startt/endt)"
  key_links:
    - from: "_do_startp"
      to: "WarpState.sloop_seen_in_plan / tloop_seen_in_plan"
      via: "reset to False on plan entry"
      pattern: "sloop_seen_in_plan = False"
    - from: "_do_starts"
      to: "WarpState.sloop_seen_in_plan"
      via: "assert-then-set"
      pattern: "assert not.*sloop_seen_in_plan"
    - from: "_do_startt"
      to: "WarpState.tloop_seen_in_plan"
      via: "assert-then-set"
      pattern: "assert not.*tloop_seen_in_plan"
    - from: "WarpState.reset()"
      to: "process-lifetime (npu.reset/__init__) sentinel cleanup"
      via: "wsplit_seen preserved (process-lifetime), new sentinels cleared (plan-lifetime, but defensively also cleared on full reset)"
      pattern: "is_sloop = False"
---

<objective>
Enforce vendor firmware plan invariant ("one shared section + one thread section per plan; no nesting; no unbalanced end_*") as fail-fast assertions in `_do_startp/_do_endp/_do_starts/_do_ends/_do_startt/_do_endt`. Extension of the "silent-clamp → assert" cleanup arc (commits b464bb4 / 765d7fb).

Purpose: Current `_do_*` helpers silently overwrite `is_ploop/is_tloop/is_sloop` flags. A firmware that emits `start_s ... start_s ... end_s ... end_s` (nested or duplicated) executes silently and may produce subtly wrong DDR. The user wants the next step in the "no silent invariant violations" posture: any plan-structure violation must fail loudly so vendor firmware bugs surface in regression, not in production.

Output: 2 modified files (warp_state.py + control.py), zero new files, ABS .elf regression PASS (or documented assertion-fail with `(plan_no, section_pattern)` context — DO NOT REVERT if assert fires).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@src/main/python/riscv/gtx/unit/context/warp_state.py
@src/main/python/riscv/gtx/unit/context/control.py
@src/main/python/riscv/gtx/npu.py

<interfaces>
<!-- Current WarpState shape (warp_state.py:10-26) -->
```python
@dataclass
class WarpState:
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False
    tmu_id: int = 0
    curr_id: int = 0
    wsplit_seen: bool = False  # process-lifetime; NOT reset by reset()

    def reset(self) -> None:
        self.is_ploop = False
        self.is_tloop = False
        self.is_sloop = False
        self.tmu_id = 0
        self.curr_id = 0
        # NOTE: wsplit_seen intentionally NOT reset (process-lifetime).
```

<!-- Current _do_* helpers (control.py:56-123) -->
- `_do_startp(npu, rs1, rs2)`: sets `tmu_id`, `is_ploop = True`. (Already asserts on NEST id range.)
- `_do_endp(npu, rs1, rs2)`: clears `is_ploop`, then if `not wsplit_seen` calls `flush_deferred_ddr_stores()`.
- `_do_startt(npu, rs1, rs2)`: asserts SPU id range, sets `curr_id`, `is_tloop = True`, inits `_tloop_buf` unless `GTX_TLOOP_DISABLE`.
- `_do_endt(npu, rs1, rs2)`: drains `_tloop_buf` if present, clears `is_tloop`.
- `_do_starts(npu, rs1, rs2)`: asserts GDMAC id range, sets `curr_id`, `is_sloop = True`.
- `_do_ends(npu, rs1, rs2)`: clears `is_sloop`.

<!-- npu.py:185-216 reset() — invoked per-hart by spike. Calls self.warp.reset(). -->
<!-- npu.py:77-80 __init__ — constructs fresh WarpState() (all defaults False/0). -->

<!-- Test files that directly manipulate WarpState flags (audit candidates) -->
<!-- test_custom_dispatch_chain.py:166 — sets `gtx_npu.warp.is_tloop = True` directly, does NOT call _do_startt → SAFE (assert is in _do_startt only). -->
<!-- test_deferred_store.py — sets is_ploop/is_sloop/is_tloop directly on lines 145,146,170,171,218,219,238,239,etc. Does NOT call _do_* helpers → SAFE for new asserts. NOTE: this file is already broken per STATE.md last_activity (5 breakages, deferred to a future quick task). Don't try to fix it in this plan. -->
</interfaces>

<rollback_rule>
**CRITICAL — Read before Task 3:**

If the ABS .elf regression in Task 3 FAILS with an `AssertionError` from one of the new guards:

1. **DO NOT REVERT** the production code.
2. Capture the failing context: `(plan_no, section_pattern, firmware_op_id)` — i.e., which assert message fired, what NEST id / SPU id was active, which firmware sequence triggered it.
3. Write the failure context to the SUMMARY.md "Findings" section.
4. Document in memory that this vendor firmware violates the new invariant.
5. User will decide afterward: (a) relax the invariant, (b) tag the firmware as exception, (c) keep strict and treat firmware as buggy.

**Do not unilaterally relax the invariant.** A loud failure here is the WHOLE POINT of this plan — silent miscompute was the prior state.

If smoke tests (Task 3 gate 1) regress, that IS a revert signal — those tests don't exercise the plan invariant and should not trip the new asserts.
</rollback_rule>

</context>

<tasks>

<task type="auto">
  <name>Task 1: Add PLAN-lifetime sentinels to WarpState + audit lifecycle</name>
  <files>src/main/python/riscv/gtx/unit/context/warp_state.py</files>
  <action>
Edit `src/main/python/riscv/gtx/unit/context/warp_state.py`:

1. Add two new fields to the `WarpState` dataclass (after `wsplit_seen` on line 19, before `def reset`):
   ```python
   # PLAN-lifetime sentinels — set True inside a plan, cleared at every
   # start_p (NOT at process reset, NOT at SPLIT/JOIN). Used by _do_starts
   # / _do_startt to assert the vendor invariant "one shared section + one
   # thread section per plan".
   sloop_seen_in_plan: bool = False
   tloop_seen_in_plan: bool = False
   ```

2. Update `reset()` (line 21-26) to ALSO clear the two new sentinels (defensive — they will be re-cleared at the next start_p, but explicit cleanup makes process-reset lifecycle obvious and prevents leakage across `.elf` runs in regression tests):
   ```python
   def reset(self) -> None:
       self.is_ploop = False
       self.is_tloop = False
       self.is_sloop = False
       self.tmu_id = 0
       self.curr_id = 0
       self.sloop_seen_in_plan = False
       self.tloop_seen_in_plan = False
       # NOTE: wsplit_seen intentionally NOT reset (process-lifetime).
   ```

3. Update the module docstring (lines 1-5) to mention the new PLAN-lifetime sentinels:
   - Before: "WarpState -- P/S/T loop state machine, port of gtx_npu_t loop fields."
   - After: add a sentence "Plan invariant sentinels (sloop_seen_in_plan, tloop_seen_in_plan) are PLAN-lifetime (cleared at every start_p); wsplit_seen remains process-lifetime."

4. Verify the lifecycle audit:
   - Run `grep -n "warp.reset\|WarpState()" src/main/python/riscv/gtx/npu.py` to confirm the only construction sites are `__init__` (line 80, fresh WarpState → all defaults False) and `reset()` (line 209, calls `self.warp.reset()`).
   - Both paths now clear the new sentinels. No leakage between `.elf` runs.

DO NOT touch `wsplit_seen` — it stays process-lifetime per existing comment.
DO NOT add any other fields or methods.
  </action>
  <verify>
    <automated>uv run python -c "from riscv.gtx.unit.context.warp_state import WarpState; w = WarpState(); assert w.sloop_seen_in_plan is False; assert w.tloop_seen_in_plan is False; w.sloop_seen_in_plan = True; w.tloop_seen_in_plan = True; w.wsplit_seen = True; w.reset(); assert w.sloop_seen_in_plan is False, 'sloop sentinel not reset'; assert w.tloop_seen_in_plan is False, 'tloop sentinel not reset'; assert w.wsplit_seen is True, 'wsplit_seen MUST stay process-lifetime'; print('OK')"</automated>
  </verify>
  <done>
WarpState has 2 new bool fields (`sloop_seen_in_plan`, `tloop_seen_in_plan`), both default False. `reset()` clears them. `wsplit_seen` lifecycle preserved (process-lifetime). Docstring updated.
  </done>
</task>

<task type="auto">
  <name>Task 2: Convert 6 _do_* helpers from silent-overwrite to assert-guarded</name>
  <files>src/main/python/riscv/gtx/unit/context/control.py</files>
  <action>
Edit `src/main/python/riscv/gtx/unit/context/control.py` lines 56-123. Convert 6 helpers (`_do_startp/_do_endp/_do_starts/_do_ends/_do_startt/_do_endt`) from silent-overwrite to assert-guarded. Match the existing assert message style on line 59 (`f"Invalid NEST ID {nest_id} in startp (is_ploop={npu.warp.is_ploop})"` — include relevant state for debuggability).

**Site 1: `_do_startp` (line 56-61)** — add nested-plan assert + sentinel reset:
```python
def _do_startp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startp. Sets is_ploop, tmu_id.

    Plan invariant: no nested plans. Also resets PLAN-lifetime
    sentinels so the per-plan "one shared + one thread section"
    invariant starts fresh.
    """
    nest_id = _extract_id(rs1, rs2)
    assert 0 <= nest_id < GTX_NEST_NUM, f"Invalid NEST ID {nest_id} in startp (is_ploop={npu.warp.is_ploop})"
    assert not npu.warp.is_ploop, f"nested start_p (is_ploop=True, tmu_id={npu.warp.tmu_id} → new nest_id={nest_id})"
    npu.warp.sloop_seen_in_plan = False
    npu.warp.tloop_seen_in_plan = False
    npu.warp.tmu_id = nest_id
    npu.warp.is_ploop = True
```

**Site 2: `_do_endp` (line 64-75)** — add unbalanced-end assert; PRESERVE existing wsplit_seen branch + flush ordering; also clear sentinels defensively:
```python
def _do_endp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::endp. Clears is_ploop. P3 (Plan 05): flushes the
    deferred-store queue when !wsplit_seen.

    RESEARCH "Deferred-Store Flush Trigger" #1: simple firmware (no WSPLIT)
    flushes here at end_p. Plan-style firmware (with WSPLIT) flushes via
    credit_st_chk mid-execution instead -- see ops/dma.py:_credit_st_chk.
    The wsplit_seen sentinel chooses the path. ROADMAP P3 success #4 path.
    """
    assert npu.warp.is_ploop, f"end_p without matching start_p (is_ploop=False, tmu_id={npu.warp.tmu_id})"
    npu.warp.is_ploop = False
    if not npu.warp.wsplit_seen:
        npu.flush_deferred_ddr_stores()
    # Defensive PLAN-lifetime sentinel cleanup (will also be cleared at
    # next start_p; explicit here for lifecycle clarity).
    npu.warp.sloop_seen_in_plan = False
    npu.warp.tloop_seen_in_plan = False
```
**Load-bearing constraint:** `flush_deferred_ddr_stores()` MUST stay BEFORE the sentinel cleanup AND its position relative to `is_ploop = False` MUST NOT change. The order is `is_ploop = False` → `flush` → sentinel cleanup.

**Site 3: `_do_startt` (line 78-93)** — add nested-thread + plan-invariant asserts; PRESERVE GTX_TLOOP_DISABLE env override + _tloop_buf init:
```python
def _do_startt(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startt. Sets is_tloop, curr_id.

    Also opens the T-loop instruction buffer (see :mod:`gtx.tloop_buffer`)
    so subsequent bufferable mnemonics are captured for replay-at-endt
    instead of executing immediately.

    Plan invariant: at most one thread section per plan; no nested
    thread sections.
    """
    spu_id = _extract_id(rs1, rs2)
    assert spu_id < GTX_SPU_NUM, f"Invalid SPU ID {spu_id} in startt (is_tloop={npu.warp.is_tloop})"
    assert not npu.warp.is_tloop, f"nested start_t (is_tloop=True, curr_id={npu.warp.curr_id} → new spu_id={spu_id})"
    assert not npu.warp.tloop_seen_in_plan, f"second thread section in same plan — invariant violation (tmu_id={npu.warp.tmu_id}, new spu_id={spu_id})"
    npu.warp.curr_id = spu_id
    npu.warp.is_tloop = True
    npu.warp.tloop_seen_in_plan = True
    # Hard kill-switch: ``GTX_TLOOP_DISABLE=1`` keeps the FSM on the eager
    # path while leaving the buffer wiring in place, so we can A/B against
    # the in-order replay path without reverting the patch.
    if not os.environ.get("GTX_TLOOP_DISABLE"):
        npu._tloop_buf = []
```

**Site 4: `_do_endt` (line 96-106)** — add unbalanced-end assert; PRESERVE tloop_buffer flush ordering:
```python
def _do_endt(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::endt. Clears is_tloop.

    Drains any buffered T-loop instructions BEFORE clearing ``is_tloop``
    so replayed handlers see the warp state they were captured under.
    """
    assert npu.warp.is_tloop, f"end_t without matching start_t (is_tloop=False, curr_id={npu.warp.curr_id})"
    if npu._tloop_buf:
        from ...tloop_buffer import flush as _flush_tloop_buf
        _flush_tloop_buf(npu)
    npu._tloop_buf = None
    npu.warp.is_tloop = False
```
**Load-bearing constraint:** assert MUST be first (before the tloop_buffer flush), so a stray `end_t` does not trigger the flush. tloop_buffer flush MUST stay BEFORE `is_tloop = False`.

**Site 5: `_do_starts` (line 109-118)** — add nested-shared + plan-invariant asserts:
```python
def _do_starts(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::starts. P3 DMA: sets is_sloop, curr_id (GDMAC).

    GTX_GDMAC_NUM == GTX_NUM_NESTS == 4 in the C++ reference, so we clamp
    against GTX_NEST_NUM.

    Plan invariant: at most one shared section per plan; no nested
    shared sections.
    """
    gdmac_id = _extract_id(rs1, rs2)
    assert 0 <= gdmac_id < GTX_NEST_NUM, f"Invalid GDMAC ID {gdmac_id} in starts (is_sloop={npu.warp.is_sloop})"
    assert not npu.warp.is_sloop, f"nested start_s (is_sloop=True, curr_id={npu.warp.curr_id} → new gdmac_id={gdmac_id})"
    assert not npu.warp.sloop_seen_in_plan, f"second shared section in same plan — invariant violation (tmu_id={npu.warp.tmu_id}, new gdmac_id={gdmac_id})"
    npu.warp.curr_id = gdmac_id
    npu.warp.is_sloop = True
    npu.warp.sloop_seen_in_plan = True
```

**Site 6: `_do_ends` (line 121-123)** — add unbalanced-end assert:
```python
def _do_ends(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::ends. Clears is_sloop."""
    assert npu.warp.is_sloop, f"end_s without matching start_s (is_sloop=False, curr_id={npu.warp.curr_id})"
    npu.warp.is_sloop = False
```

**OUT OF SCOPE — DO NOT TOUCH:**
- WSPLIT/WJOIN handlers (lines 185-211) — orthogonal to plan invariant.
- `wsplit_seen` field — stays process-lifetime, separate from new plan-lifetime sentinels.
- `wsplit_custom0` / `wjoin_custom0_no_exit` (lines 218-242).
- `dispatch_*_stub` handlers (lines 245-266).
- `_credit_st_chk` / `_credit_ld_chk` in `unit/context/dma.py` — user confirmed correct as NOPs (per memory project_gtx_credit_semantics).
- The existing `wsplit_seen` branch in `_do_endp` — preserve verbatim with its flush call.
  </action>
  <verify>
    <automated>uv run python -c "
from riscv.gtx.unit.context.control import _do_startp, _do_endp, _do_starts, _do_ends, _do_startt, _do_endt
from riscv.gtx.unit.context.warp_state import WarpState

class _StubNpu:
    def __init__(self):
        self.warp = WarpState()
        self._tloop_buf = None
        self.deferred_ddr_stores = []
    def flush_deferred_ddr_stores(self): pass

# Happy path: start_p -> start_s -> end_s -> start_t -> end_t -> end_p
n = _StubNpu()
_do_startp(n, 0, 0); _do_starts(n, 0, 0); _do_ends(n, 0, 0)
_do_startt(n, 0, 0); _do_endt(n, 0, 0); _do_endp(n, 0, 0)
print('happy path OK')

# Nested start_p must fail
n = _StubNpu(); _do_startp(n, 0, 0)
try: _do_startp(n, 0, 0); raise SystemExit('FAIL: nested start_p not caught')
except AssertionError as e: assert 'nested start_p' in str(e); print('nested start_p OK')

# Two start_s in same plan must fail
n = _StubNpu(); _do_startp(n, 0, 0); _do_starts(n, 0, 0); _do_ends(n, 0, 0)
try: _do_starts(n, 0, 0); raise SystemExit('FAIL: 2nd start_s not caught')
except AssertionError as e: assert 'second shared section' in str(e); print('2nd start_s OK')

# Two start_t in same plan must fail
n = _StubNpu(); _do_startp(n, 0, 0); _do_startt(n, 0, 0); _do_endt(n, 0, 0)
try: _do_startt(n, 0, 0); raise SystemExit('FAIL: 2nd start_t not caught')
except AssertionError as e: assert 'second thread section' in str(e); print('2nd start_t OK')

# Nested start_s must fail
n = _StubNpu(); _do_startp(n, 0, 0); _do_starts(n, 0, 0)
try: _do_starts(n, 0, 0); raise SystemExit('FAIL: nested start_s not caught')
except AssertionError as e: assert 'nested start_s' in str(e); print('nested start_s OK')

# Nested start_t must fail
n = _StubNpu(); _do_startp(n, 0, 0); _do_startt(n, 0, 0)
try: _do_startt(n, 0, 0); raise SystemExit('FAIL: nested start_t not caught')
except AssertionError as e: assert 'nested start_t' in str(e); print('nested start_t OK')

# Unbalanced end_p / end_s / end_t must fail
n = _StubNpu()
try: _do_endp(n, 0, 0); raise SystemExit('FAIL: stray end_p not caught')
except AssertionError as e: assert 'end_p without' in str(e); print('stray end_p OK')
try: _do_ends(n, 0, 0); raise SystemExit('FAIL: stray end_s not caught')
except AssertionError as e: assert 'end_s without' in str(e); print('stray end_s OK')
try: _do_endt(n, 0, 0); raise SystemExit('FAIL: stray end_t not caught')
except AssertionError as e: assert 'end_t without' in str(e); print('stray end_t OK')

# Sentinel reset across plans: start_p N+1 must clear sentinels so the next plan can re-emit start_s
n = _StubNpu()
_do_startp(n, 0, 0); _do_starts(n, 0, 0); _do_ends(n, 0, 0); _do_startt(n, 0, 0); _do_endt(n, 0, 0); _do_endp(n, 0, 0)
_do_startp(n, 0, 0); _do_starts(n, 0, 0); _do_ends(n, 0, 0)  # must NOT trip
print('cross-plan sentinel reset OK')

print('ALL 9 INVARIANT CASES PASS')
"</automated>
  </verify>
  <done>
All 6 `_do_*` helpers have assert guards matching the spec. Inline 9-case invariant test (happy path + 8 violation cases) passes. `_do_endp`'s `flush_deferred_ddr_stores()` ordering preserved. `_do_endt`'s tloop_buffer flush ordering preserved. GTX_TLOOP_DISABLE env override preserved.
  </done>
</task>

<task type="auto">
  <name>Task 3: Smoke + ABS .elf regression gate (rollback rule applies)</name>
  <files></files>
  <action>
Run two regression gates in order. STOP and apply the rollback rule from `<rollback_rule>` if Gate 2 fails on an AssertionError.

**Gate 1 — Smoke (must PASS, regression = revert signal):**
```bash
cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest \
  tests/gtx/test_custom_dispatch_chain.py \
  tests/gtx/test_custom0_smoke.py \
  tests/gtx/test_fsm_smoke.py \
  tests/gtx/test_csr_registry_chain.py \
  -v 2>&1 | tail -40
```
Expected: all collected tests PASS. These tests do NOT exercise the plan invariant directly; if any of them fails, the new asserts have caught a real bug in another test path AND that's a signal to investigate before proceeding. Do not modify the production code to make them pass — instead capture the trace and stop.

(Audit note from planning: `test_custom_dispatch_chain.py:166` does `gtx_npu.warp.is_tloop = True` directly but does NOT call `_do_startt`, so the new asserts in `_do_startt` won't trip. Same for `test_deferred_store.py`'s direct flag writes — but that file is already broken per STATE.md last_activity and is not in Gate 1.)

**Gate 2 — ABS .elf single-case regression (THE invariant gate):**
```bash
cd /mnt/e/14_NIGHTLY/pyspike && PYTEST_ELF_REGRESSION=1 uv run pytest \
  tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=600 -v 2>&1 | tail -30
```

**Three outcomes:**

1. **PASS (byte-exact)** → Invariant holds in real firmware. Plan complete. Write SUMMARY.md.

2. **FAIL with AssertionError from new guard** → Real firmware violates the invariant. **DO NOT REVERT.**
   - Capture from the pytest output: the exact assert message (which `_do_*` helper, what state values, which `tmu_id`/`spu_id`/`gdmac_id`).
   - If the failure is reproducible, run with a higher verbosity / faster fail: `PYTEST_ELF_REGRESSION=1 uv run pytest tests/gtx/test_regression_elf_n1s16.py -k "abs" --timeout=600 -v -s 2>&1 | tail -100` and capture surrounding context (last few instruction trace lines if available, the failing plan number if visible).
   - Record in SUMMARY.md "Findings":
     - Which assert message fired (verbatim)
     - The `(plan_no, section_pattern, firmware_op_id)` context as best as can be inferred
     - Hypothesis: which firmware violates the invariant and how
   - Update `/home/sw.lee/.claude/projects/-mnt-e-14-NIGHTLY-pyspike/memory/` with a new memory note documenting the vendor-firmware invariant violation.
   - DO NOT EDIT `warp_state.py` OR `control.py` TO MAKE THE TEST PASS. The user has explicit final say.

3. **FAIL with non-AssertionError byte-mismatch** → Regression unrelated to the new asserts (e.g., the changes accidentally affected control flow). This IS a revert signal — review the diff for unintended side effects (most likely a mis-ordering of `flush_deferred_ddr_stores()` in `_do_endp` or `_tloop_buf` flush in `_do_endt`).

**Per project memory:** all test commands MUST use `uv run pytest` — system torch is broken. Do NOT use bare `pytest` or `python -m pytest`.

After both gates complete (or after capturing the failure context), write SUMMARY.md to `.planning/quick/260515-mie-plan-invariant-assert-warpstate-sloop-tl/260515-mie-SUMMARY.md` covering:
- What changed (2 files, 7 sites: 2 fields in warp_state.py + 6 asserts in control.py)
- Gate 1 result (smoke pass/fail)
- Gate 2 result (ABS pass/fail-with-context)
- Findings if Gate 2 failed (the (plan_no, section_pattern) data — DO NOT REVERT)
- Followups (e.g., extend to other vendor firmwares once ABS clean)
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_custom_dispatch_chain.py tests/gtx/test_custom0_smoke.py tests/gtx/test_fsm_smoke.py tests/gtx/test_csr_registry_chain.py --tb=short 2>&1 | tail -10</automated>
  </verify>
  <done>
Gate 1 smoke tests PASS. Gate 2 (ABS .elf) executed and outcome documented (either byte-exact PASS, or AssertionError with `(plan_no, section_pattern)` context captured in SUMMARY.md per rollback rule — NOT reverted). SUMMARY.md written.
  </done>
</task>

</tasks>

<verification>
- 2 files modified, no new files: warp_state.py, control.py.
- `git diff src/main/python/riscv/gtx/unit/context/warp_state.py src/main/python/riscv/gtx/unit/context/control.py` shows only the documented changes — no drive-by edits to other functions, no changes to wsplit_custom0/wjoin/dispatch_* handlers, no changes to dma.py credit handlers.
- The 9-case inline invariant test in Task 2 verify block passes.
- Gate 1 smoke tests (4 files) PASS unchanged.
- Gate 2 ABS .elf either PASSes byte-exact OR fails with a documented AssertionError context (NOT a silent revert).
- `wsplit_seen` remains process-lifetime (verified by Task 1 inline test).
- `flush_deferred_ddr_stores()` call in `_do_endp` is still present and ordered after `is_ploop = False` and before the new sentinel cleanup.
- `_tloop_buf` flush in `_do_endt` still runs before `is_tloop = False`.
</verification>

<success_criteria>
- Vendor firmware plan invariant ("one shared + one thread section per plan; no nesting; no unbalanced end_*") is now enforced by assertions in 6 `_do_*` helpers.
- ABS .elf regression either PASSes byte-exact (invariant holds in real firmware) OR fails loudly with documented context (vendor firmware bug exposed — user decides next step).
- Smoke tests do NOT regress (none of the smoke tests exercise the plan invariant).
- No production behavior changes for compliant firmware — assertions are no-ops on the happy path.
- The "silent-clamp → assert" cleanup arc advances by one step.
</success_criteria>

<output>
After completion, create `.planning/quick/260515-mie-plan-invariant-assert-warpstate-sloop-tl/260515-mie-SUMMARY.md`. Summary MUST include:

- **What changed:** 2 files, 7 modification sites (2 new fields in warp_state.py + 6 assert-guards in control.py).
- **Gate 1 result:** Smoke tests (4 files) — pass count / fail count.
- **Gate 2 result:** ABS .elf byte-exact outcome.
- **Findings (if Gate 2 failed on AssertionError):** Exact assert message + inferred `(plan_no, section_pattern, firmware_op_id)` context. Hypothesis about the firmware behavior. **Explicit note: NOT REVERTED — awaiting user decision.**
- **Memory updates:** If vendor firmware was found to violate invariant, link to the new memory note documenting it.
- **Followups:** e.g., extend regression to other vendor ops once ABS is clean; replan if user opts to relax invariant.
</output>
