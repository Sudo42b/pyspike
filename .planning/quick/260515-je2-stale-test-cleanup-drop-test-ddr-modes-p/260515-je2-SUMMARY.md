---
phase: quick-260515-je2
plan: 01
subsystem: tests/gtx
tags: [test-cleanup, post-refactor-debt, stale-tests, dma, deferred-store]
requires:
  - 639ddb4   # riscv.gtx.ddr module consolidated into unit/memory.py
  - 53eb670   # hybrid torch backend on L2 (CUDA)
  - a79e418   # gtx split into context/ + ins/ subpackages; dispatch_4mode.py removed
  - 2ec3fab   # last STATE record before this task
provides:
  - clean-pytest-collection   # tests/gtx --collect-only: 0 errors
affects:
  - tests/gtx/test_ddr_modes.py (deleted)
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - tests/gtx/test_ddr_modes.py   # DELETED only
decisions:
  - "Task 2 (test_deferred_store.py repair) deferred to a follow-up quick task — scope expanded beyond the originally-described fixture fix; 6 of 11 tests are stale-by-semantics or stale-by-deleted-entry-point and require contract-level rewrites, not API substitution."
metrics:
  duration: "~15min (Task 1) + rollback"
  tasks_completed: 1
  tasks_deferred: 1
  files_deleted: 1
  files_modified: 0
  completed: 2026-05-15
---

# Quick Task 260515-je2: Stale Test Cleanup — Drop test_ddr_modes.py (Partial)

Deleted `tests/gtx/test_ddr_modes.py` (consolidates the post-639ddb4 module
removal); deferred the broader `test_deferred_store.py` repair to a follow-up
task after audit revealed contract-level breakages outside this task's scope.

## What Got Done

### Task 1 (COMPLETE) — `tests/gtx/test_ddr_modes.py` deleted

- **Commit:** `36f5cc5` — `test(gtx): drop stale test_ddr_modes.py — riscv.gtx.ddr removed in 639ddb4`
- **Rationale:** The file (292 lines) imported `riscv.gtx.ddr` and exercised
  `ensure_ddr / ddr_init_from_file / ddr_dump_to_file / get_ddr_cap` — all of
  which were removed in `639ddb4` (DDR module consolidated under
  `unit/memory.py` as methods on `GtxMemory.ddr` / `DDR_MEMORY`). DDR I/O
  behavior is already covered by `tests/gtx/test_regression_elf_n1s16.py` (R3,
  opt-in via `PYTEST_ELF_REGRESSION=1`). User decision: delete, not rewrite.
- **Collection gate:** `uv run pytest tests/gtx --collect-only` now reports
  `190 tests collected in 1.60s` with **0 errors** (was: 1 ImportError on
  `riscv.gtx.ddr`).

### Task 2 (DEFERRED) — `tests/gtx/test_deferred_store.py` repair

All in-progress edits to `test_deferred_store.py` were **rolled back**
(`git checkout HEAD -- tests/gtx/test_deferred_store.py`). The file is now at
tip-of-branch state, unmodified. No commit was made for Task 2.

**Why deferred:** A full read of the file during execution revealed two
breakages beyond the three documented in the PLAN, totalling **five** distinct
post-refactor breakages — three are simple API substitutions, but two require
contract-level rewrites that exceed a "fix fixture" quick task's scope.

## Audit: Five Breakages Found in `test_deferred_store.py`

### Breakages in PLAN (A, B, C) — mechanical substitution

| ID | Site                                         | Issue                                                       | Fix Pattern                                    |
| -- | -------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| A  | Line 74 `npu_with_pattern` fixture           | `np.arange(..., dtype=np.uint8)` RHS to torch.uint8 CUDA slice → `TypeError` (53eb670) | torch tensor on `_dst.device`                  |
| B  | Line 112 `from riscv.gtx.ddr import ensure_ddr` | Module removed in 639ddb4                                   | `npu.mem.ensure_ddr(...)` method               |
| C  | Lines 118, 123, 158, 230, 277 `npu.mem._ddr_bytes` | Attribute removed; DDR is now `DDR_MEMORY` instance         | `npu.mem.ddr.read(off, n).numpy()` / `.raw()`  |

A/B/C are well-scoped and were the original target of this quick task.

### Breakages discovered during audit (D, E) — OUT OF SCOPE

**Breakage D — `dispatch_4mode.dispatch_iss_opcode` entry-point deleted (commit `a79e418`):**

- Test lines 263, 273, 284, 293 import and call
  `from riscv.gtx.dispatch_4mode import dispatch_iss_opcode`.
- `src/main/python/riscv/gtx/dispatch_4mode.py` **no longer exists** post-`a79e418`
  (the gtx package was split into `context/` + `ins/` subpackages; no file in the
  current tree exports `dispatch_iss_opcode`).
- Verification:
  ```
  find src/main/python/riscv/gtx -name "dispatch*.py" -type f
   → src/main/python/riscv/gtx/dispatch.py
   → src/main/python/riscv/gtx/dispatch_state.py
  grep -rn "dispatch_iss_opcode" src/main/python/riscv/gtx --include="*.py"
   → (no matches)
  ```
- **Affected tests (2):**
  - `test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop` (line 255)
  - `test_dispatch_iss_opcode_credit_st_chk_no_flush_when_not_sloop` (line 282)
- **Resolution required:** Either delete these two tests entirely (the
  dispatch_iss_opcode entry-point is gone — there's nothing left to test
  through it) or rewrite them against the current dispatch surface (`dispatch.py`).
  This is a test-deletion / test-rewrite decision, not an API substitution.

**Breakage E — `_credit_st_chk` semantic flipped to intentional NOP (commit `a79e418` era):**

- Test lines 215-232 (`test_credit_st_chk_flushes_when_is_sloop`) and
  236-249 (`test_credit_st_chk_no_flush_when_not_sloop`) assert that
  `custom0 funct7=0x53` (credit.st.chk) with `is_sloop=True` flushes the
  deferred-store queue, and with `is_sloop=False` retains it.
- Current implementation at `src/main/python/riscv/gtx/unit/context/dma.py:381-388`
  is an **unconditional NOP returning 0** — no flush, regardless of `is_sloop`:
  ```python
  @handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST_CHK,
           mnemonic='credit.st.chk')
  def _credit_st_chk(npu, proc, insn, xs1, xs2):
      """Direct port of gtx_npu_dispatch.cc credit.st.chk branch.

      Functional-model NOP — same rationale as :func:`_credit_ld_chk`:
      deferred-store visibility is owned by ``end_p`` / ``__join``, not by
      in-loop credit checks.
      """
      return 0
  ```
- **Vendor C++ justification** (quoted from `_credit_ld_chk` docstring at
  `src/main/python/riscv/gtx/unit/context/dma.py:369-374` — `_credit_st_chk`
  shares the rationale):
  > "Vendor C++ Spike commented out the same flush call (see
  > `gtx_npu_dispatch.cc` 'GTX ggml bring-up: do not commit deferred L2->DDR
  > stores from credit_chk; endp/launch boundary must own visibility.'). All
  > deferred stores instead drain at `end_p` (when not `wsplit_seen`) or at
  > `__join` — both of which run after the thread loop has finished writing
  > `L2_RES`."
- **Affected tests (4 assertions across 2 tests):**
  - `test_credit_st_chk_flushes_when_is_sloop` — asserts `deferred_ddr_stores == []` after the call (line 228) and that DDR bytes equal the pattern (line 229-232). Both will fail: handler is NOP, queue is untouched, DDR is still zeros.
  - `test_credit_st_chk_no_flush_when_not_sloop` — coincidentally still passes (handler NOP-s, queue retained), but for the **wrong reason** (it's testing the absence of behavior that no longer exists at this site).
  - The two `dispatch_iss_opcode_*` tests (Breakage D) duplicate the same flawed assumption that credit.st.chk owns flush.
- **Resolution required:** These 4 assertions test the *old* P8 MTDMA-01
  contract that was explicitly reverted. The new contract — "credit_chk is
  NOP; flush happens at `end_p` / `__join`" — needs new tests against the
  correct sites (`control.py end_p` handler and the WJOIN path), not patched
  assertions on the obsolete site. This is a test-rewrite, not a fixture fix.

### Summary of test fate

| Test (in `test_deferred_store.py`)                                       | Status under post-refactor contract |
| ------------------------------------------------------------------------ | ----------------------------------- |
| 5 tests not touching DDR/credit_chk (e.g., `test_wsplit_custom0_sets_wsplit_seen`, reset-clearing) | Will pass after A+C mechanical fix  |
| `test_credit_st_chk_flushes_when_is_sloop`                                | **Stale by semantics (E)** — needs rewrite or deletion |
| `test_credit_st_chk_no_flush_when_not_sloop`                              | Coincidentally passes; **logically stale (E)** |
| `test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop`            | **Stale by deleted entry-point (D)** — delete |
| `test_dispatch_iss_opcode_credit_st_chk_no_flush_when_not_sloop`          | **Stale by deleted entry-point (D)** — delete |
| `test_deferred_store_flush_diff` (line 112-region, uses ensure_ddr / `_ddr_bytes`) | Will pass after A+B+C mechanical fix |
| `test_reset_clears_deferred_queue_but_not_wsplit_seen` etc.               | Will pass after A+C mechanical fix  |

**6 of 11 tests** require contract-level intervention (4 stale-by-semantics
under E, 2 stale-by-deleted-entry-point under D). Patching A+B+C alone would
leave the file in a misleading "passes for the wrong reasons" state.

## Verification (This Task Only)

| Gate                                                                            | Status |
| ------------------------------------------------------------------------------- | ------ |
| `tests/gtx/test_ddr_modes.py` removed                                            | PASS — `[ -f ... ]` reports DELETED |
| `uv run pytest tests/gtx --collect-only` reports 0 errors                        | **PASS** — 190 collected, 0 errors |
| `git status tests/gtx/test_deferred_store.py` clean (rollback successful)        | PASS — `nothing to commit, working tree clean` |
| `uv run pytest tests/gtx/test_deferred_store.py` reports `11 passed` (PLAN gate) | **NOT MET — explicitly deferred** |

**Collection-error gate (this task's only in-scope success criterion) is MET.**
The PLAN's full success criteria (`11 passed`) is **deferred** to the
follow-up quick task per the decision recorded above.

## Deviations from Plan

### Scope reduction (architectural, Rule 4-equivalent — flagged to main session, not auto-fixed)

- **Found during:** Task 2 audit (reading the file end-to-end before patching)
- **Issue:** PLAN's Breakage A/B/C list is incomplete; breakages D and E
  require contract-level test redesign, not API substitution.
- **Action:** Halted Task 2 mid-edit, rolled back all uncommitted changes
  to `tests/gtx/test_deferred_store.py`, surfaced to main session for
  scope decision. Main session chose **Option C — defer Task 2 to a separate
  follow-up plan** so the test-rewrite work can be sized and reviewed
  independently from the test-deletion this quick task targeted.
- **Files modified:** None (rollback).
- **Commit:** None for Task 2.

## Follow-up Required (to be opened by main session)

A new quick task to address `test_deferred_store.py` must decide, per test,
between:

1. **Delete** the two `dispatch_iss_opcode_*` tests (Breakage D — entry-point
   is gone, no equivalent symbol exists).
2. **Rewrite** the two `_credit_st_chk` tests against the new contract
   ("flush owned by `end_p` / `__join`"), targeting the `end_p` handler in
   `unit/context/control.py` and the `__join` path. Cite the vendor C++
   rationale at `src/main/python/riscv/gtx/unit/context/dma.py:369-374`.
3. **Mechanical fix** for the remaining 7 tests (A+B+C substitutions per the
   original PLAN Task 2 action block — those edits are valid, just insufficient
   on their own).

## Self-Check: PASSED

- `tests/gtx/test_ddr_modes.py`: **MISSING** as expected (deletion confirmed).
- Commit `36f5cc5`: **FOUND** in `git log --oneline`.
- `tests/gtx/test_deferred_store.py`: clean working tree (rollback confirmed via `git status`).
- Collection gate: **0 errors** out of 190 tests collected.
- Vendor C++ comment cited verbatim from `unit/context/dma.py:369-374` (verified by Read).
- Breakages D and E verified by `find` / `grep` against current source tree (no `dispatch_4mode.py`, no `dispatch_iss_opcode` symbol; `_credit_st_chk` is unconditional `return 0`).
