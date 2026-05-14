---
phase: quick-260514-sqv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/context/control.py
autonomous: true
requirements:
  - SQV-01  # Restore _extract_id to vendor 2-arg semantics (rs2 marker bit)
  - SQV-02  # Make _do_startp / _do_endp 2-arg to match dispatch handlers
  - SQV-03  # Smoke gate: tests/gtx/test_fsm_smoke.py + test_custom0_smoke.py 5/5 PASS

must_haves:
  truths:
    - "_extract_id(rs1, rs2) takes 2 args and uses rs2 & 0x400 as the marker bit (verbatim port of gtx_npu_loop.cc:21-23)"
    - "_do_startp and _do_endp accept (npu, rs1, rs2) so dispatch handlers at lines 173/183 stop raising TypeError"
    - "tests/gtx/test_custom0_smoke.py::test_custom1_returns_int turns GREEN (currently FAILS via _do_startt -> _extract_id(rs1, rs2) TypeError)"
    - "All 5 smoke tests in test_fsm_smoke.py + test_custom0_smoke.py pass"
    - "No regression: _do_startt / _do_endt / _do_starts / _do_ends / _do_startsmu (already 2-arg in signature, already call _extract_id(rs1, rs2) in body) auto-become consistent the moment _extract_id is restored — no edits required for them"
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/context/control.py"
      provides: "Restored vendor-port loop-control helpers"
      contains: "def _extract_id(rs1: int, rs2: int) -> int"
  key_links:
    - from: "control.py::startp handler (line 168-174)"
      to: "control.py::_do_startp (line 56)"
      via: "_do_startp(npu, rs1_val, rs2_val)"
      pattern: "_do_startp\\(npu, rs1_val, rs2_val\\)"
    - from: "control.py::endp handler (line 177-184)"
      to: "control.py::_do_endp (line 64)"
      via: "_do_endp(npu, rs1_val, rs2_val)"
      pattern: "_do_endp\\(npu, rs1_val, rs2_val\\)"
    - from: "control.py::_do_startt / _do_starts / _do_startsmu (already 2-arg, untouched)"
      to: "control.py::_extract_id (restored 2-arg)"
      via: "_extract_id(rs1, rs2)"
      pattern: "_extract_id\\(rs1, rs2\\)"
---

<objective>
Restore vendor 2-arg semantics for `_extract_id` and align `_do_startp` / `_do_endp` signatures with their dispatch handler call sites, fixing the TypeError introduced by d6f73f9 "Architecture Refactoring..." that broke `tests/gtx/test_custom0_smoke.py::test_custom1_returns_int`.

Purpose: d6f73f9 partially refactored `_extract_id` from 2-arg → 1-arg but only touched `_do_startp` / `_do_endp`. The other four `_do_*` helpers (startt/endt/starts/ends/startsmu) still call `_extract_id(rs1, rs2)`, and dispatch handlers at lines 173 / 183 still call `_do_startp(npu, rs1_val, rs2_val)` / `_do_endp(npu, rs1_val, rs2_val)` with 2 ints. Result: any custom1 dispatch raises TypeError. Vendor `gtx_npu_loop.cc:21-23` is authoritative: marker bit lives on `rs2` and all six functions take `(rs1, rs2)`. This plan restores that contract.

Output: A single edited `control.py` with `_extract_id(rs1, rs2)` and `_do_startp(npu, rs1, rs2)` / `_do_endp(npu, rs1, rs2)`. Smoke tests 5/5 GREEN.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/main/python/riscv/gtx/unit/context/control.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc
@tests/gtx/test_custom0_smoke.py
@tests/gtx/test_fsm_smoke.py
@tests/gtx/conftest.py
@tests/gtx/_mocks.py
@src/main/python/riscv/gtx/unit/context/spr_router.py

<interfaces>
<!-- Vendor C++ source-of-truth (gtx_npu_loop.cc:21-23, 37-39, 74-76 — all six -->
<!-- vendor loop helpers follow the same 2-arg + (rs2 & 0x400) ternary pattern). -->

```cpp
// vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:21-23
void gtx_npu_t::startp(uint64_t rs1, uint64_t rs2)
{
    uint32_t id = (rs2 & 0x400) ? (rs2 & 0x3F) : static_cast<uint32_t>(rs1);
    // ...
}

// gtx_npu_loop.cc:37-39 — endp uses identical id-extraction
void gtx_npu_t::endp(uint64_t rs1, uint64_t rs2)
{
    uint32_t id = (rs2 & 0x400) ? (rs2 & 0x3F) : static_cast<uint32_t>(rs1);
    // ...
}
```

<!-- Existing 2-arg call sites in control.py that already presume the restored -->
<!-- signature (DO NOT touch these — they auto-fix once _extract_id is restored): -->

```python
# control.py:85   _do_startt body
spu_id = _extract_id(rs1, rs2)

# control.py:115  _do_starts body
gdmac_id = _extract_id(rs1, rs2)

# control.py:136  startt dispatch handler
_do_startt(npu, rs1_val, rs2_val)

# control.py:145  endt dispatch handler
_do_endt(npu, rs1_val, rs2_val)

# control.py:156  starts dispatch handler
_do_starts(npu, rs1_val, rs2_val)

# control.py:165  ends dispatch handler
_do_ends(npu, rs1_val, rs2_val)

# control.py:173  startp dispatch handler — currently passes 2 ints to 1-arg _do_startp (TypeError)
_do_startp(npu, rs1_val, rs2_val)

# control.py:183  endp dispatch handler — also passes rs2_val (which is even
#                 commented out at line 182, so it's a NameError too if line
#                 173 hadn't already raised TypeError first)
_do_endp(npu, rs1_val, rs2_val)
```

<!-- spr_router.py:42-65 has the future un-comment sites — they call helpers -->
<!-- with `(npu, value, 0)` i.e. 2 args. Restoring 2-arg signatures KEEPS those -->
<!-- future un-comments compatible. -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Restore _extract_id to 2-arg vendor semantics and align _do_startp / _do_endp signatures</name>
  <files>src/main/python/riscv/gtx/unit/context/control.py</files>
  <action>
Surgical edits to ONLY three functions in `src/main/python/riscv/gtx/unit/context/control.py`. Do NOT touch any other function, comment, import, or formatting in this file (per Karpathy §3 surgical-changes; the file's other helpers, dispatch handlers, custom0 stubs, WJOIN, WSPLIT, progress logic, and module-level constants are all correct already).

**Edit 1 — `_extract_id` (current lines 42-49)**

Replace the current 1-arg body:

```python
def _extract_id(rs1: int) -> int:
    """Dual-mode addressing: rs2 marker bit selects rs2 low6 vs rs1 low32.

    Verbatim port of gtx_npu_loop.cc:21-23 (the marker-bit ternary).
    """
    if rs1 & 0x400:
        return rs1 & 0x3F
    return rs1 & 0xFFFFFFFF
```

with the restored 2-arg version (per SQV-01, vendor `gtx_npu_loop.cc:21-23` — marker lives on **rs2**, not rs1):

```python
def _extract_id(rs1: int, rs2: int) -> int:
    """Dual-mode addressing: rs2 marker bit selects rs2 low6 vs rs1 low32.

    Verbatim port of gtx_npu_loop.cc:21-23 (the marker-bit ternary).
    """
    if rs2 & 0x400:
        return rs2 & 0x3F
    return rs1 & 0xFFFFFFFF
```

The docstring is already correct (it describes 2-arg semantics) — keep it verbatim.

**Edit 2 — `_do_startp` (current lines 56-61)**

Replace the current 1-arg body:

```python
def _do_startp(npu: "GtxNpu", rs1: int) -> None:
    """Port of gtx_npu_t::startp. Sets is_ploop, tmu_id."""
    nest_id = _extract_id(rs1)
    assert 0 <= nest_id < GTX_NEST_NUM, f"Invalid NEST ID {nest_id} in startp (is_ploop={npu.warp.is_ploop})"
    npu.warp.tmu_id = nest_id
    npu.warp.is_ploop = True
```

with the 2-arg version (per SQV-02 — matches the existing 2-arg dispatch call at line 173):

```python
def _do_startp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
    """Port of gtx_npu_t::startp. Sets is_ploop, tmu_id."""
    nest_id = _extract_id(rs1, rs2)
    assert 0 <= nest_id < GTX_NEST_NUM, f"Invalid NEST ID {nest_id} in startp (is_ploop={npu.warp.is_ploop})"
    npu.warp.tmu_id = nest_id
    npu.warp.is_ploop = True
```

**Edit 3 — `_do_endp` (current lines 64-75)**

Add `rs2: int` to the signature ONLY. The body does not use the id, so no internal changes — the `rs2` param exists purely for caller-compat (per SQV-02, mirrors the vendor `endp(uint64_t rs1, uint64_t rs2)` signature even though the C++ vendor uses it only for trace logging at line 44).

Replace:

```python
def _do_endp(npu: "GtxNpu", rs1: int) -> None:
```

with:

```python
def _do_endp(npu: "GtxNpu", rs1: int, rs2: int) -> None:
```

KEEP the existing docstring and body verbatim — the deferred-store flush trigger logic (P3 Plan 05 / RESEARCH §"Deferred-Store Flush Trigger" #1) is correct and unrelated to this signature surgery.

**Do NOT touch:**
- Lines 78-123: `_do_startt` / `_do_endt` / `_do_starts` / `_do_ends` already have 2-arg signatures and already call `_extract_id(rs1, rs2)`. They auto-become consistent the instant Edit 1 lands.
- Lines 131-184: All six custom1 dispatch handlers (startt/endt/starts/ends/startp/endp) already pass 2 ints. They are the **driver** of this fix, not a target.
- Lines 186-267: WSPLIT / WJOIN / custom0 funct7 stubs — orthogonal to loop-id extraction.
- The `# rs2_val = state.XPR[insn.rs2]` comment at line 182 inside `endp` dispatch handler: line 183 already references `rs2_val`. Line 182 is a stale comment but per Karpathy §3 do NOT delete it — it's "unrelated dead code" that should be mentioned in the SUMMARY, not silently removed in this surgical fix. (User can decide later; if they hit it on the next run they'll see line 183's `rs2_val` is actually unbound → NameError. That is a SEPARATE bug from the one this plan fixes, surfaced naturally by line 173's TypeError no longer masking it. Flag in SUMMARY's "Notes / Followups".)

**Karpathy alignment:**
- §1 Think Before Coding: only assumption is "the docstring at line 43-46 already describes 2-arg semantics, so restoring the body to match the docstring is the natural fix" — no interpretation latitude.
- §2 Simplicity First: zero new abstractions, three minimal edits totaling ~3 line changes.
- §3 Surgical Changes: no adjacent code touched, no unrelated dead code deleted (the stale `rs2_val` comment in `endp` handler is flagged in SUMMARY, not removed).
- §4 Goal-Driven Execution: success = `uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v` shows 5 passed.

After edits, run a syntax sanity check before declaring done:

```bash
uv run python -c "from riscv.gtx.unit.context import control; \
  import inspect; \
  assert list(inspect.signature(control._extract_id).parameters) == ['rs1', 'rs2'], 'bad _extract_id sig'; \
  assert list(inspect.signature(control._do_startp).parameters) == ['npu', 'rs1', 'rs2'], 'bad _do_startp sig'; \
  assert list(inspect.signature(control._do_endp).parameters) == ['npu', 'rs1', 'rs2'], 'bad _do_endp sig'; \
  print('signatures OK')"
```
  </action>
  <verify>
    <automated>uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v</automated>
  </verify>
  <done>
    1. `uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v` reports `5 passed` (3 from test_fsm_smoke + 2 from test_custom0_smoke).
    2. `_extract_id` has signature `(rs1: int, rs2: int) -> int` and body uses `rs2 & 0x400` as the marker check.
    3. `_do_startp` has signature `(npu, rs1: int, rs2: int) -> None` and body calls `_extract_id(rs1, rs2)`.
    4. `_do_endp` has signature `(npu, rs1: int, rs2: int) -> None`; body is otherwise unchanged.
    5. No other line in `control.py` modified (verify with `git diff --stat src/main/python/riscv/gtx/unit/context/control.py` — should show only `control.py` touched, minimal line count).
    6. No new imports added; module still loads cleanly.
  </done>
</task>

</tasks>

<verification>
**Primary gate (Nyquist-automated):**

```bash
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v
```

Expected: 5 passed, 0 failed.

Specifically:
- `test_fsm_smoke.py::test_npu_state_enum_has_five_members` — already passing pre-fix (orthogonal)
- `test_fsm_smoke.py::test_state_functions_are_importable_callables` — already passing pre-fix
- `test_fsm_smoke.py::test_state_writeback_returns_idle` — already passing pre-fix
- `test_custom0_smoke.py::test_custom0_returns_int` — already passing pre-fix
- `test_custom0_smoke.py::test_custom1_returns_int` — **currently FAILS** via `_do_startt → _extract_id(rs1, rs2)` TypeError; **must turn GREEN**

**Diff sanity check:**

```bash
git diff --stat src/main/python/riscv/gtx/unit/context/control.py
```

Expected: single file in diff, ≤ ~10 line changes (3 signatures + 1 body line in `_extract_id` + 1 body line in `_do_startp`). If the diff is larger, surgery scope was violated.

**Vendor parity check (manual eyeball):**

`_extract_id` body must read `if rs2 & 0x400: return rs2 & 0x3F` — matching `gtx_npu_loop.cc:23` `(rs2 & 0x400) ? (rs2 & 0x3F) : ...`.
</verification>

<success_criteria>
1. `tests/gtx/test_fsm_smoke.py + tests/gtx/test_custom0_smoke.py` → 5 PASS / 0 FAIL.
2. `_extract_id(rs1, rs2)` body uses `rs2 & 0x400` marker (vendor-parity).
3. `_do_startp(npu, rs1, rs2)` and `_do_endp(npu, rs1, rs2)` accept the 2 ints that the dispatch handlers at lines 173 / 183 already pass.
4. Diff is confined to `src/main/python/riscv/gtx/unit/context/control.py`; no other file touched.
5. The four other `_do_*` helpers (`startt`/`endt`/`starts`/`ends`) — already 2-arg in signature and already calling `_extract_id(rs1, rs2)` — remain untouched and operate correctly via the restored `_extract_id`.
6. `spr_router.py:42-65` future un-comment sites (which call helpers with `(npu, value, 0)` — i.e. always 2 args) remain forward-compatible with the restored signatures.
</success_criteria>

<output>
After completion, create `.planning/quick/260514-sqv-extract-id-2-arg-do-startp-do-endp-d6f73/260514-sqv-01-SUMMARY.md` documenting:

1. **What changed** — the three surgical edits, with before/after one-liners.
2. **Why** — d6f73f9 partial-refactor + vendor `gtx_npu_loop.cc:21-23` as source-of-truth.
3. **Test gate** — pre-fix: 4/5 PASS (1 TypeError on `test_custom1_returns_int`); post-fix: 5/5 PASS.
4. **Notes / Followups (do NOT auto-act on these):**
   - Line 182 of `control.py` has stale comment `# rs2_val = state.XPR[insn.rs2]` immediately above line 183's `_do_endp(npu, rs1_val, rs2_val)`. With this fix, line 173's TypeError no longer masks line 183's `rs2_val` NameError-in-waiting. User should decide separately whether to un-comment line 182 or rewrite the `endp` handler to pass `0` as `rs2`. (Both vendor-equivalent for the body, which doesn't use rs2.) Per the d6f73f9 author's intent ("rs1 data만 있음" comment at line 180), they may prefer passing `0`. Flag — do NOT fix.
   - `spr_router.py:42-65` has 6 commented-out call sites of `_do_*` that pass `(npu, value, 0)`. Restored 2-arg signatures keep these forward-compatible if/when un-commented.
</output>
