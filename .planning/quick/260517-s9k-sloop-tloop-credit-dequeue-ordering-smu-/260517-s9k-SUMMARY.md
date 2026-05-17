---
phase: quick/260517-s9k
plan: 01
subsystem: gtx/credit-dequeue
tags: [sloop-buffer, tloop-buffer, credit-counter, smu, tmu, producer-consumer]
dependency-graph:
  requires:
    - src/main/python/riscv/gtx/tloop_buffer.py
    - src/main/python/riscv/gtx/unit/context/dma.py:_credit_ld
    - src/main/python/riscv/gtx/unit/context/dma.py:_credit_st
  provides:
    - src/main/python/riscv/gtx/sloop_buffer.py (NEW)
    - producer-consumer dequeue scaffolding at credit_*_chk
  affects:
    - src/main/python/riscv/gtx/npu.py (custom0 dispatch + _sloop_buf slot)
    - src/main/python/riscv/gtx/execute.py (sloop dispatch hook)
    - src/main/python/riscv/gtx/unit/context/control.py (_do_starts/_do_ends)
    - src/main/python/riscv/gtx/unit/context/dma.py (_credit_ld_chk/_credit_st_chk)
tech-stack:
  added: []
  patterns:
    - SLoopEntry namedtuple (mirrors TLoopEntry shape)
    - Shim-replay pattern (XPRShim/StateShim/ProcShim/InsnShim) duplicated cheaply
    - Credit clamp-at-0 (option a) for chk-handler decrement
key-files:
  created:
    - src/main/python/riscv/gtx/sloop_buffer.py
    - src/main/python/riscv/gtx/_verify.py  # restored from git history (639ddb4 deletion)
  modified:
    - src/main/python/riscv/gtx/npu.py
    - src/main/python/riscv/gtx/execute.py
    - src/main/python/riscv/gtx/unit/context/control.py
    - src/main/python/riscv/gtx/unit/context/dma.py
    - src/main/python/riscv/gtx/tloop_buffer.py
decisions:
  - chose-clamp-at-0-over-producer-decrement-removal
  - sequential-replay-no-fusion-for-sloop
  - restore-_verify-py-from-git-as-rule-3-blocker-fix
metrics:
  duration_s: 3665
  completed: 2026-05-17
  tasks: 2
  files_changed: 6
---

# Quick Plan 260517-s9k: S-loop / T-loop credit-dequeue ordering Summary

Add per-unit S-loop instruction buffer (`_sloop_buf`) symmetric to the
existing T-loop buffer, and upgrade `credit_ld_chk` / `credit_st_chk`
from documented NOPs to credit-gated dequeue triggers — scaffolding for
the SMU/TMU producer-consumer pattern documented at
`vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:41-61`
(`use_spu_queue` / `use_tmu_queue` push/pop infrastructure).

## What landed

### Task 1 (commit `d26e1c5`): sloop_buffer + _sloop_buf wiring

- **`src/main/python/riscv/gtx/sloop_buffer.py`** (NEW, 270 lines):
  Public API: `SLOOP_BUFFERABLE_MNEMONICS` (frozenset of 5 mnemonics
  — `load`/`store`/`copy`/`credit_ld`/`credit_st`),
  `SLOOP_TRANSPARENT_MNEMONICS`, `SLoopEntry` namedtuple,
  `try_buffer`, `flush`, `dequeue_one_batch` helper. No fusion path
  (SMU emits only DMA + counter ops, not the 1.18M-entry vec hot
  loop tloop_buffer optimizes for). Module docstring cites vendor
  parity and pins the non-regression invariant (no
  `flush_deferred_ddr_stores` callsite).
- **`npu.py`**: import sloop_buffer symbols; `_sloop_buf` slot in
  `__init__` + `reset()`; `is_sloop` dispatch branch in `custom0`
  hot path parallel to existing `is_tloop` branch (mutually
  exclusive per FSM — `start_t` and `start_s` set their own flag
  only).
- **`unit/context/control.py`**: `_do_starts` opens buffer
  (`GTX_SLOOP_DISABLE=1` kill-switch parity with
  `GTX_TLOOP_DISABLE`); `_do_ends` drains then clears, mirroring
  `_do_endt`.
- **`execute.py`**: S-loop buffering hook in `state_execute`,
  parallel to T-loop hook; `credit_*_chk` stays transparent (chk
  handler will own dequeue per Task 2).
- **`_verify.py`** (Rule 3 auto-fix): restored from
  `git show 639ddb4^:src/main/python/riscv/gtx/_verify.py` (the
  ext-module consolidation commit accidentally deleted the source
  but left the `.pyc` cached; `tests/gtx/test_regression_fw_full_sweep.py:420`
  imports `riscv.gtx._verify.compare_hex` and was blocked).

### Task 2 (commit `4aacd76`): credit-gated chk dequeue + tloop comment

- **`unit/context/dma.py` `_credit_ld_chk` (TMU side, C3)**:
  S-loop drains FIRST when both buffers have content (spec rule 7);
  decrement `npu._credit_ld[nest, spu]` with **clamp-at-0**
  (option a); drain `_tloop_buf`. Docstring records (a)/(b)
  decision rationale + non-regression invariant.
- **`unit/context/dma.py` `_credit_st_chk` (SMU side, C2)**:
  Decrement first non-zero SPU slot in `npu._credit_st[nest, :]`
  with clamp-at-0; drain `_sloop_buf`. `_tloop_buf` is None in C2
  context so spec rule 7 is trivially satisfied.
- **`tloop_buffer.py` TRANSPARENT_MNEMONICS comment**: refresh the
  third bullet to reflect that `credit_*_chk` is no longer fully
  transparent on the TMU side — it triggers a dequeue. **Membership
  UNCHANGED**; fusion machinery
  (`BUFFERABLE_MNEMONICS` / `_Frame` / `_execute_fused` /
  `_try_fuse_unary` / `_drain`) **UNTOUCHED**.

## Decisions Made

### 1. Credit double-decrement: chose option (a) clamp-at-0

The producer-side `_credit_ld` T-loop branch at `dma.py:325-334`
already decrements `npu._credit_ld[nest, curr_id]`. If
`_credit_ld_chk` ALSO decremented unconditionally, the counter
would go negative on real firmware (ABS emits `credit.ld` once per
SPU per tile inside the TMU thread, then `credit.ld.chk` consumes
it).

**Plan offered two resolutions:**
- (a) Keep producer-side decrement; clamp chk-handler decrement
  at 0 (`if cred > 0: cred -= 1`). Safer default.
- (b) Remove the T-loop branch decrement entirely; make
  `_credit_ld_chk` the sole consumer. Cleaner semantically.

**Chose (a) over (b)** because:
- **Safer**: clamp-at-0 is a no-op when producer-side already
  decremented, preserving existing eager-mode behavior bit-for-bit
  (verified: ABS .elf 96-tile strict byte-exact PASS).
- **Reversible**: if a future cycle-accurate path needs (b), the
  change is local to two functions.
- **(b) is riskier**: may surface regressions in non-multi-tile
  firmware that relied on the prior producer-side decrement pattern
  — defer to a separate plan if actually needed.

Same rationale applies to `_credit_st_chk` (mirror handler).

### 2. Sequential replay only for sloop_buffer (no fusion)

SMU emits a handful of DMA setup ops per NEST per section, not the
1.18M-entry inner vec loop that tloop_buffer's `_execute_fused`
optimizes for. Sequential replay is correctness-sufficient and
keeps the new module surface area minimal. If a future SMU hot path
appears, mirror `_Frame` / `_try_fuse_unary` / `_execute_fused`
from tloop_buffer at that time.

### 3. Restore _verify.py as Rule 3 (blocking) auto-fix

Commit `639ddb4` (2026-05-12 "consolidate ext modules under unit/")
accidentally deleted `src/main/python/riscv/gtx/_verify.py` from the
source tree while leaving the `.pyc` cached. The acceptance gate
test `tests/gtx/test_regression_fw_full_sweep.py:420` imports
`riscv.gtx._verify.compare_hex` and would have failed
`ModuleNotFoundError` regardless of this Plan's changes. Restored
from git history (parent of the deletion commit) as part of Task 1
since it blocks plan verification. 156 lines, no logic changes.

## Verification

### Acceptance gate (strict byte-exact)

| Test | Status | Walltime | Notes |
|------|--------|----------|-------|
| `test_vendor_op_sweep_strict[ABS]` | **PASS** | 526s | 96 tiles, 196609 lines vendor golden, byte-exact |
| `test_vendor_op_sweep_strict[GELU]` | FAIL (pre-existing) | 57s | `act.py:298 firmware_act` assert; identical line numbers before/after Task 2 — out of scope |
| `test_regression_fw_full.py` | PASS (2 skipped) | <1s | Baseline state |
| `test_deferred_store.py` | FAIL (pre-existing) | <2s | Test fixture numpy→cuda tensor mismatch; out of scope (logged in STATE.md as "재작성") |

### Walltime delta (ABS, before / after sloop-buffering)

- Task 1 baseline (sloop wiring only, chk still NOP): 524s
- Task 2 (chk dequeue active): 526s

Delta: **< 1%, within noise**. Sloop buffering adds one
snapshot+replay per S-loop op, but the SMU per-NEST cardinality is
a few ops per section (vs the TMU 1.18M-entry inner loop), so the
overhead is amortized to near-zero.

### Non-regression invariants

```
$ grep -rn "npu\.flush_deferred_ddr_stores\|self\.flush_deferred_ddr_stores" \
  src/main/python/riscv/gtx/ --include="*.py"
src/main/python/riscv/gtx/unit/context/control.py:75:        npu.flush_deferred_ddr_stores()
src/main/python/riscv/gtx/unit/context/control.py:228:    npu.flush_deferred_ddr_stores()
src/main/python/riscv/gtx/unit/context/control.py:258:    npu.flush_deferred_ddr_stores()
```

**3 actual callsites + 1 def** (`npu.py:348`) = **4 lines of legitimate
flush wiring, UNCHANGED**. Zero callsites in `dma.py` or new
`sloop_buffer.py` (only docstring/comment mentions documenting the
non-regression invariant).

```
$ grep -cn "_sloop_buf" src/main/python/riscv/gtx/npu.py
3   # __init__ slot, reset() clear, custom0 branch read
```

### tloop_buffer.py untouched (except TRANSPARENT_MNEMONICS comment)

- `BUFFERABLE_MNEMONICS`: UNCHANGED
- `TRANSPARENT_MNEMONICS` (set membership): UNCHANGED — only the
  comment block text was refreshed.
- `_Frame`, `_execute_fused`, `_try_fuse_unary`, `_drain`,
  `_parse_frame`, `_frame_signature`, `_replay_frames`,
  `_replay`, `_decode_dma`: UNCHANGED.

## Cross-reference Comments Added

| File | Where | What |
|------|-------|------|
| `sloop_buffer.py` | Module docstring | Vendor parity ref (gtx_npu_dispatch.cc:41-61); context_map.yaml C2 group; explicit non-regression note (no flush_deferred_ddr_stores callsite) |
| `npu.py` | `_sloop_buf` slot comment + custom0 branch | Pointer to `sloop_buffer.py` + mutual-exclusivity rationale with `is_tloop` |
| `control.py` | `_do_starts`/`_do_ends` docstrings | Mirror-of-startt/endt language; `GTX_SLOOP_DISABLE` env-var parity with `GTX_TLOOP_DISABLE` |
| `dma.py` | `_credit_ld_chk`/`_credit_st_chk` docstrings | Vendor parity + spec rule 7 + (a)/(b) double-decrement rationale + non-regression invariant pin |
| `tloop_buffer.py` | TRANSPARENT_MNEMONICS comment | Cross-link to `sloop_buffer.py`; clarification that `credit_*_chk` is no longer fully transparent but membership stays in set so FSM doesn't force a hard pre-flush |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] Restored `src/main/python/riscv/gtx/_verify.py`**
- **Found during:** Task 1 verification (acceptance gate first run)
- **Issue:** `tests/gtx/test_regression_fw_full_sweep.py:420` imports
  `riscv.gtx._verify.compare_hex`, but the source file was deleted by
  commit `639ddb4` (2026-05-12 ext-module consolidation) — only the
  `.pyc` remained, which is invisible to fresh interpreter imports.
- **Fix:** Restored the file from git history
  (`git show 639ddb4^:src/main/python/riscv/gtx/_verify.py`). 156 lines,
  no behavioral changes — this is the same `_verify` module the
  consolidation commit was supposed to keep.
- **Files modified:** `src/main/python/riscv/gtx/_verify.py` (NEW —
  restored)
- **Commit:** `d26e1c5` (Task 1)

### Out-of-Scope Discoveries (Deferred)

These are PRE-EXISTING baseline failures unrelated to this Plan's
changes. Per deviation-rules scope boundary, do NOT fix forward in
this Plan.

1. **`test_deferred_store.py`** — pre-existing test fixture bug:
   `npu.mem.l2_byte(0)[100:200] = np.arange(100, dtype=np.uint8)`
   raises `TypeError: can't assign a numpy.ndarray to a
   torch.cuda.ByteTensor`. Fixture needs torch tensor on the right
   side. Logged in STATE.md as "test_deferred_store.py 재작성"
   followup.
2. **`test_vendor_op_sweep_strict[GELU]`** — pre-existing crash in
   `src/main/python/riscv/gtx/unit/ins/ops/act.py:298 firmware_act`
   assertion (`is_reversed` mismatch). Identical line numbers before
   AND after Task 2 — not caused by this Plan. Needs separate debug
   investigation; the GELU baseline never passed under the current
   repo state independently of credit-dequeue wiring.

## Self-Check

### Created files exist

```
FOUND: src/main/python/riscv/gtx/sloop_buffer.py
FOUND: src/main/python/riscv/gtx/_verify.py
FOUND: .planning/quick/260517-s9k-sloop-tloop-credit-dequeue-ordering-smu-/260517-s9k-SUMMARY.md
```

### Commits exist

```
FOUND: d26e1c5  feat(quick-260517-s9k): sloop_buffer scaffolding + _sloop_buf wiring (Task 1)
FOUND: 4aacd76  feat(quick-260517-s9k): credit-gated chk dequeue + tloop comment refresh (Task 2)
```

## Self-Check: PASSED

## Candidate MEMORY.md Entries

- **`project_credit_chk_dequeue_pattern`** (new): "credit_*_chk now owns
  buffer dequeue (260517-s9k). TMU drains _sloop_buf FIRST (rule 7) +
  _tloop_buf. SMU drains _sloop_buf. Clamp-at-0 on chk-handler
  decrement to coexist with producer-side decrement (option a, not
  option b). Non-regression invariant pinned in handler docstrings:
  MUST NOT call flush_deferred_ddr_stores."
- **`project_sloop_tloop_mutual_exclusion`** (new): "`warp.is_sloop` and
  `warp.is_tloop` are mutually exclusive — start_t sets is_tloop only,
  start_s sets is_sloop only; both share `curr_id` slot meaning only
  one can be active per dispatch. npu.custom0 branches in independent
  `if` blocks (no `elif`); only one will fire per dispatch."
