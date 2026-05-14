---
phase: quick-260514-sqv
plan: 01
type: execute
subsystem: gtx/unit/context
tags: [bugfix, vendor-parity, custom1, loop-control, smoke-gate]
requirements:
  - SQV-01  # Restore _extract_id to vendor 2-arg semantics (rs2 marker bit) — DONE
  - SQV-02  # Make _do_startp / _do_endp 2-arg to match dispatch handlers — DONE
  - SQV-03  # Smoke gate: 5/5 PASS on tests/gtx/test_fsm_smoke.py + test_custom0_smoke.py — DONE
key-files:
  modified:
    - src/main/python/riscv/gtx/unit/context/control.py
  created: []
decisions:
  - "Restored _extract_id to 2-arg (rs1, rs2) form using rs2 & 0x400 as marker, per vendor gtx_npu_loop.cc:23. d6f73f9 had partial-refactored it to 1-arg using rs1 as marker, which contradicted both the docstring and the four sibling _do_* helpers that still call _extract_id(rs1, rs2)."
  - "Added rs2 parameter to _do_endp signature even though the body does not consume it — preserves call-site symmetry with _do_startp/_do_startt/etc and matches vendor C++ `endp(uint64_t rs1, uint64_t rs2)` signature. Vendor only uses rs2 in trace logging, so the unused-param shape is faithful to the reference."
  - "Did NOT touch the stale `# rs2_val = state.XPR[insn.rs2]` comment at line 182 of the `endp` handler (Karpathy §3 surgical-changes). It is flagged under Open Notes below for the successor."
metrics:
  duration: "~3 min"
  completed: "2026-05-14T11:48:00Z"
  tests_run: 5
  tests_pre_fix: "4 PASS / 1 FAIL"
  tests_post_fix: "5 PASS / 0 FAIL"
  lines_changed: "6 insertions(+), 6 deletions(-)"
---

# Phase quick-260514-sqv: Restore `_extract_id` 2-arg vendor semantics

**One-liner:** Reverted d6f73f9's partial 1-arg refactor of `_extract_id` and aligned `_do_startp` / `_do_endp` signatures with their existing 2-arg dispatch call sites, unblocking `test_custom1_returns_int` (TypeError) without touching the four sibling helpers that already used the vendor 2-arg contract.

## What changed

Three surgical edits in `src/main/python/riscv/gtx/unit/context/control.py` (lines 42-64):

| # | Function     | Before                                           | After                                                       |
|---|--------------|--------------------------------------------------|-------------------------------------------------------------|
| 1 | `_extract_id`| `def _extract_id(rs1: int)` — body keyed on rs1  | `def _extract_id(rs1: int, rs2: int)` — body keyed on **rs2** marker (`if rs2 & 0x400: return rs2 & 0x3F`) |
| 2 | `_do_startp` | `def _do_startp(npu, rs1)` — calls `_extract_id(rs1)` | `def _do_startp(npu, rs1, rs2)` — calls `_extract_id(rs1, rs2)` |
| 3 | `_do_endp`   | `def _do_endp(npu, rs1)`                         | `def _do_endp(npu, rs1, rs2)` — body unchanged              |

Diff stat: **6 insertions(+), 6 deletions(-)**. No other lines, comments, imports, or formatting touched.

## Why

Commit d6f73f9 ("Architecture Refactoring...") partially migrated `_extract_id` from 2-arg → 1-arg and updated only `_do_startp` / `_do_endp` to match. It left:

- The four sibling `_do_*` helpers (`startt`/`endt`/`starts`/`ends`) still calling `_extract_id(rs1, rs2)` — would have raised TypeError on first dispatch.
- The six custom1 dispatch handlers (lines 131-184) still passing two `int`s to `_do_startp(npu, rs1_val, rs2_val)` / `_do_endp(npu, rs1_val, rs2_val)` — also TypeError.
- The docstring of `_extract_id` still describing the 2-arg vendor semantics ("rs2 marker bit selects rs2 low6 vs rs1 low32").

Vendor `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:21-23, 37-39, 74-76, 107-109, 127-129` is the source of truth and uniformly uses:

```cpp
uint32_t id = (rs2 & 0x400) ? (rs2 & 0x3F) : static_cast<uint32_t>(rs1);
```

across all six `startp/endp/starts/ends/startt/endt`. Restoring 2-arg semantics on `_extract_id` makes the four already-2-arg helpers correct again *automatically* — no cascading edits required (Karpathy §3 surgical-changes).

## Test gate

```bash
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v
```

**Pre-fix (per 260514-s68 smoke discovery):** 4 PASS / 1 FAIL — `test_custom1_returns_int` failed via `_do_startt → _extract_id(rs1, rs2)` TypeError.

**Post-fix:** **5 passed in 2.47s** — all green:

```
tests/gtx/test_fsm_smoke.py::test_npu_state_enum_has_five_members PASSED [ 20%]
tests/gtx/test_fsm_smoke.py::test_state_functions_are_importable_callables PASSED [ 40%]
tests/gtx/test_fsm_smoke.py::test_state_writeback_returns_idle PASSED    [ 60%]
tests/gtx/test_custom0_smoke.py::test_custom0_returns_int PASSED         [ 80%]
tests/gtx/test_custom0_smoke.py::test_custom1_returns_int PASSED         [100%]
```

**Signature sanity check** (passed before the pytest run):

```bash
uv run python -c "from riscv.gtx.unit.context import control; import inspect; \
  assert list(inspect.signature(control._extract_id).parameters) == ['rs1', 'rs2']; \
  assert list(inspect.signature(control._do_startp).parameters) == ['npu', 'rs1', 'rs2']; \
  assert list(inspect.signature(control._do_endp).parameters) == ['npu', 'rs1', 'rs2']; \
  print('signatures OK')"
```
→ `signatures OK`

## Deviations from Plan

None — plan executed exactly as written.

## Open Notes for Successor (do NOT auto-act)

1. **`endp` handler latent NameError (line 182-183 of `control.py`):**
   ```python
   # rs1 data만 있음.
   rs1_val = state.XPR[insn.rs1]
   # rs2_val = state.XPR[insn.rs2]   <- line 182, stale comment
   _do_endp(npu, rs1_val, rs2_val)   <- line 183, references unbound name
   ```
   The fix to `_extract_id` removes the TypeError that was masking this. The very next call to `endp` will now raise `NameError: name 'rs2_val' is not defined`. **This is a separate bug from SQV.** The d6f73f9 author's "rs1 data만 있음" comment suggests intent to pass `0` (or to not need rs2 at all). Two reasonable resolutions, both vendor-equivalent for the body that ignores rs2:
   - **(a)** Un-comment line 182 (`rs2_val = state.XPR[insn.rs2]`) — mirrors all other handlers.
   - **(b)** Replace line 183 with `_do_endp(npu, rs1_val, 0)` — matches the comment's intent.

   Flagged per Karpathy §3 — successor decides. **Smoke tests do not exercise this path** so they stayed GREEN even with the latent bug; only a custom1 funct3=0b111 (`end.p`) dispatch in a real .elf or in a custom1-funct3-coverage test will trip it.

2. **`spr_router.py:42-65` forward-compat:** The six commented-out call sites in `spr_router.wr_spr` for GSPR 0x100..0x105 call `_do_*` with `(npu, value, 0)` — i.e. 2 ints. The restored 2-arg signatures keep these forward-compatible when un-commented in P3+.

## Known Stubs

None introduced by this plan. (The four `custom0` dispatch_*_stub functions at lines 246-267 of `control.py` are pre-existing P3+ placeholders and untouched.)

## Self-Check: PASSED

- **File modified:** `src/main/python/riscv/gtx/unit/context/control.py` — confirmed via `git diff --stat`.
- **Diff scope:** 6 insertions / 6 deletions, single file, exactly the three planned edits.
- **Signatures:** all three verified via `inspect.signature` (see Test gate).
- **Smoke gate:** 5/5 PASS confirmed.
- **No new imports added.** Module loads cleanly under `uv run python -c "from riscv.gtx.unit.context import control"`.
- **Sibling helpers untouched:** `_do_startt`/`_do_endt`/`_do_starts`/`_do_ends` retain their pre-existing 2-arg signatures and now operate correctly via the restored `_extract_id`.
