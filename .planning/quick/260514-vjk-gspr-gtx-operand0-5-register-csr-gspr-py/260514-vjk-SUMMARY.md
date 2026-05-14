---
phase: quick
plan: 260514-vjk
subsystem: gtx/csr
tags: [csr, encoding, register, t-loop, regression-guard]
requirements:
  - VJK-01  # GSPR_GTX_OPERAND0..5 @csr registration (csr/gspr.py source of truth)
  - VJK-02  # encoding.py re-export of bare-int address constants
  - VJK-03  # 5 callsite import realignment (npu/tloop_buffer/act/vec) + mm.py verify
  - VJK-04  # T-loop fast-path NameError regression test
requires:
  - "src/main/python/riscv/gtx/unit/csr/register.py: @csr decorator + bits(start, end)"
  - "src/main/python/riscv/gtx/unit/csr/__init__.py: CSR_GSPR PIPE-only view (auto-built from GSPR registry)"
  - "test fixtures: gtx_npu, mock_proc, dummy_insn (tests/gtx/conftest.py)"
provides:
  - "csr/gspr.py: 6 @csr declarations GSPR_GTX_OPERAND0..5 @ 0x000..0x005 (RW, 64-bit, PIPE)"
  - "encoding.py: 6 module-level int constants GSPR_GTX_OPERAND0..5 derived from GSPR registry"
  - "npu.py: T-loop fast-path resolves _GSPR_OP3 / _GSPR_OP5 (NameError-free)"
  - "tloop_buffer.py: import path active for GSPR_GTX_OPERAND3/5 (snapshot path)"
  - "act.py: GSPR_GTX_OPERAND1/2/3 in-scope for PReLU/GeLU/Tanh/Pool/Softmax kernels"
  - "vec.py: GSPR_GTX_OPERAND2/3 in-scope for int-key gspr access at lines 223, 250"
  - "mm.py: CSR_GSPR['GSPR_GTX_OPERAND3'] dict lookup auto-resolves (no edit)"
  - "test_custom_dispatch_chain.py: test_tloop_fast_path_opset_no_nameerror regression guard"
affects:
  - "src/main/python/riscv/gtx/unit/csr/gspr.py"
  - "src/main/python/riscv/gtx/unit/ins/encoding.py"
  - "src/main/python/riscv/gtx/npu.py"
  - "src/main/python/riscv/gtx/tloop_buffer.py"
  - "src/main/python/riscv/gtx/unit/ins/ops/act.py"
  - "src/main/python/riscv/gtx/unit/ins/ops/vec.py"
  - "tests/gtx/test_custom_dispatch_chain.py"
tech-stack:
  added: []
  patterns:
    - "@csr decorator: register populates module-level GSPR dict at import time; value=bits(0,63) is minimal no-empty-fields satisfier"
    - "encoding.py re-export: bare-int aliases derived from GSPR[name].address (NOT hard-coded ints) — single-source-of-truth pattern"
    - "csr/__init__.py:CSR_GSPR auto-filters PIPE bus_type from GSPR registry — string-key dict lookup needs zero downstream change"
key-files:
  created: []
  modified:
    - "src/main/python/riscv/gtx/unit/csr/gspr.py (+34 lines: 6 @csr blocks @ 0x000..0x005)"
    - "src/main/python/riscv/gtx/unit/ins/encoding.py (+13 lines: re-export block)"
    - "src/main/python/riscv/gtx/npu.py (+2 lines: import GSPR_GTX_OPERAND3/5 as _GSPR_OP3/_GSPR_OP5)"
    - "src/main/python/riscv/gtx/tloop_buffer.py (0 net: 1 line uncomment)"
    - "src/main/python/riscv/gtx/unit/ins/ops/act.py (0 net: 1 line uncomment)"
    - "src/main/python/riscv/gtx/unit/ins/ops/vec.py (+1 line: add OPERAND2/3 to import)"
    - "tests/gtx/test_custom_dispatch_chain.py (+21 lines: 1 new test)"
decisions:
  - "csr/gspr.py is the single source of truth (vendor gtx_params.h:36-44 informational)"
  - "Python-extended mapping (OPERAND0..5) preserved over vendor's 4-slot mapping — pre-d6f73f9 behaviour"
  - "Each @csr block uses minimal value=bits(0,63) field (callsites use raw int access)"
  - "encoding.py constants DERIVED from GSPR[name].address (not hard-coded) — drift-safe"
  - "mm.py CSR_GSPR['GSPR_GTX_OPERAND3'] auto-resolves via PIPE-filter view — no edit needed"
  - "vec.py string-key vs int-key style mismatch kept as-is (out of scope per CONTEXT D-OutOfScope)"
  - "GSPR_GTX_OPCODE excluded — no callsite needs it as of 2026-05-14"
metrics:
  duration: "~25 minutes"
  completed: "2026-05-14"
  tasks: 2
  files: 7
  commits: 2
---

# Phase quick / 260514-vjk: GSPR_GTX_OPERAND0..5 register 복원 Summary

**One-liner:** 6 GSPR operand staging registers (0x000..0x005) silently dropped
by d6f73f9 are re-registered in `csr/gspr.py` as the single source of truth;
`encoding.py` re-exports them as bare-int aliases derived from the registry,
and 5 callsite imports (`npu.py`, `tloop_buffer.py`, `act.py`, `vec.py`) are
realigned so the T-loop fast-path / activation / vector handlers no longer
hold latent `NameError` / `KeyError` landmines for production firmware.

**Phase:** quick / 260514-vjk
**Plans executed:** 1/1 (2 tasks atomic)
**Status:** DONE
**Commits:** `b228422` (Task 1), `b5700da` (Task 2)

---

## What Changed

### Task 1 — CSR registration + encoding re-export (`b228422`)

- **`csr/gspr.py`** (+34 lines): Inserted 6 `@csr`-decorated blocks at the top
  of the "64-bit PIPE Registers" section (BEFORE existing `STACK_INFO@0x010`):

  | Name | Address | Width | RW | Field |
  |---|---|---|---|---|
  | `GSPR_GTX_OPERAND0` | `0x000` | 64 | RW | `value = bits(0, 63)` |
  | `GSPR_GTX_OPERAND1` | `0x001` | 64 | RW | `value = bits(0, 63)` |
  | `GSPR_GTX_OPERAND2` | `0x002` | 64 | RW | `value = bits(0, 63)` |
  | `GSPR_GTX_OPERAND3` | `0x003` | 64 | RW | `value = bits(0, 63)` |
  | `GSPR_GTX_OPERAND4` | `0x004` | 64 | RW | `value = bits(0, 63)` |
  | `GSPR_GTX_OPERAND5` | `0x005` | 64 | RW | `value = bits(0, 63)` |

  Each carries a single `value = bits(0, 63)` field — the minimal
  `@csr no-empty-fields` invariant satisfier. `bus_type` defaults to
  `BusType.PIPE` (matches surrounding STACK_INFO/STACK_SAVE style).

- **`encoding.py`** (+13 lines): Appended re-export block at end of file.
  Imports `GSPR` from `..csr.gspr` and re-exposes addresses as module-level
  `int` constants — DERIVED from registry, not hard-coded. `mm.py` continues
  using `CSR_GSPR['GSPR_GTX_OPERAND3']` dict lookup which now auto-resolves
  through `csr/__init__.py:CSR_GSPR` (PIPE-filter view of the GSPR registry).

### Task 2 — 5 callsite imports + new regression test (`b5700da`)

- **`npu.py`** (+2 lines): Extended import block at lines 27–31 with
  `GSPR_GTX_OPERAND3 as _GSPR_OP3, GSPR_GTX_OPERAND5 as _GSPR_OP5`. Resolves
  the previously-unbound `_GSPR_OP3` / `_GSPR_OP5` references in the T-loop
  fast-path at lines 238/240/264/265/269/270 (OPSET inline + bufferable
  snapshot + WRITEBACK-mirror clear).

- **`tloop_buffer.py`** (0 net): Uncommented existing line 35 — was
  `# from .unit.ins.encoding import GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND5`;
  now active.

- **`act.py`** (0 net): Uncommented existing line 32 in the import block —
  was `# GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,`; now
  active. Unblocks the 10 callsites in PReLU/GeLU/Tanh/Sigmoid/Softmax/Pool
  kernels (act.py:329–461). The other 2 commented imports
  (`ACT_OPS_REVERSED`, `GSPR_GTX_OPCODE`) were left commented per CONTEXT
  out-of-scope.

- **`vec.py`** (+1 line): Added `GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,` on a
  new line at the top of the existing `from ..encoding import (...)` block.
  Resolves the bare-int access at `vec.py:223`
  (`npu.gspr.get(GSPR_GTX_OPERAND3, ...)`) and `vec.py:250`
  (`npu.gspr[GSPR_GTX_OPERAND2] = rs2`). The two string-key sites at
  lines 198/212 (`npu.gspr.get("GSPR_GTX_OPERAND3", ...)`) were left as-is
  per CONTEXT — string vs int key style unification is a separate task.

- **`mm.py`** (UNCHANGED — verified via `git status --short`): The
  `CSR_GSPR['GSPR_GTX_OPERAND3']` dict lookups at lines 251 & 280 now
  resolve automatically through the PIPE-filter view rebuilt at module
  import.

- **`tests/gtx/test_custom_dispatch_chain.py`** (+21 lines): Appended
  `test_tloop_fast_path_opset_no_nameerror`. Drives the npu.py:236–241
  fast-path (`warp.is_tloop=True` + `_tloop_buf=[]` + `funct=GTX_ISS_F7_OPSET`
  + `xs1[insn.rs1] LSB=0`), asserts `gspr.tensor[0x003] == 0xCAFE`. Locks
  the NameError-free invariant going forward — pre-Task 2 this test would
  have raised `NameError: name '_GSPR_OP3' is not defined`.

---

## Test Gate

**Before (baseline):** 23/23 PASS (T-loop fast-path / activation / MM kernel
paths held latent `NameError` / `KeyError` because no test entered them).

**After (final):** **24/24 PASS** via the verification command:

```bash
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py \
              tests/gtx/test_csr_registry_chain.py \
              tests/gtx/test_custom_dispatch_chain.py -v
```

Output: `============================== 24 passed in 7.18s ==============================`.

The new `test_tloop_fast_path_opset_no_nameerror` is the 24th test — it
explicitly enters the previously-unreachable code path that pre-d6f73f9
firmware will trigger on every T-loop OPSET-bearing tile.

---

## Deviations from Plan

**None — plan executed exactly as written.**

LOC budget: plan estimated ~60 ± 5; actual delta was **+73 / -2** (47 from
Task 1, 26 from Task 2). The extra 13 lines came from the encoding.py
re-export block needing a section-header comment, the test docstring being
slightly longer than the spec sketch, and the multi-line `from .unit.ins.encoding
import (...)` block in npu.py expanding by 3 lines instead of 2. No
architectural change; well within sensible scope for the "single source of
truth" pattern.

---

## Open Notes for Successor

### Still pending (next user-flagged quick task)

1. **`OverflowError @ register_file.py:188`** — separate task, explicitly
   excluded from this scope. Symptom: `OverflowError: Python integer X out
   of bounds for int64` when storing 64-bit unsigned values into the
   int64-backed `RegisterFile.tensor`. Pre-existing — unaffected by this
   work. **Do NOT touch `register_file.py` until that task is queued.**

2. **`GSPR_GTX_OPCODE` re-export** — deferred. No callsite as of 2026-05-14
   needs it; vendor `gtx_params.h:36-44` has `GSPR_GTX_OPCODE@0x004` but the
   Python `encoding.py` extension previously mapped it to `0x012`. If a
   future op needs OPCODE access, decide first which mapping wins
   (vendor 0x004 conflicts with `GSPR_GTX_OPERAND4`).

3. **vec.py string-key vs int-key style mismatch** — `vec.py:198/212` use
   `npu.gspr.get("GSPR_GTX_OPERAND3", ...)` while `vec.py:223/250` use the
   int symbol `GSPR_GTX_OPERAND3` / `GSPR_GTX_OPERAND2`. Both work because
   the CSR registration covers both lookup paths. Style unification is a
   separate code-quality task.

4. **act.py 2 still-commented imports** — `# ACT_OPS_REVERSED,` and
   `# GSPR_GTX_OPCODE,` left commented. The first will be needed when the
   `act_engine.is_reversed` policy assertion is wired (mentioned in act.py
   module docstring item 7); the second waits on Open Note #2 resolution.

### Negative-invariant guarantees (verified clean by `git status`)

- `src/main/python/riscv/gtx/unit/ins/ops/mm.py` — UNCHANGED
- `src/main/python/riscv/gtx/unit/ins/ops/spr.py` — UNCHANGED
- `src/main/python/riscv/gtx/unit/register_file.py` — UNCHANGED
- No new occurrence of `GSPR_GTX_OPCODE` anywhere
- No reordering of existing imports in modified callsites

### Pattern to remember

For future GSPR additions: **decorate in `csr/gspr.py`** (single source of
truth) → append a re-export line in `encoding.py`. Never hard-code address
ints in `encoding.py` — drift-safe derivation from `GSPR[name].address`
keeps the registry authoritative.

---

## Canonical References

- `src/main/python/riscv/gtx/unit/csr/gspr.py:30-65` (new 6 @csr blocks)
- `src/main/python/riscv/gtx/unit/ins/encoding.py:297-309` (new re-export block)
- `src/main/python/riscv/gtx/npu.py:27-31` (extended import block)
- `src/main/python/riscv/gtx/npu.py:226-276` (T-loop fast-path, now NameError-free)
- `tests/gtx/test_custom_dispatch_chain.py:159-178` (new regression test)
- `vendor/gtx_cpp_reference/gtx/gtx_params.h:36-44` (vendor source — informational)
- `src/main/python/riscv/gtx/unit/csr/__init__.py:CSR_GSPR` (PIPE-only view)
- Prior context: `.planning/quick/260514-ti0-csr-custom0-1-dispatch-test-tests-gtx/260514-ti0-SUMMARY.md` (original Open-Note flag)

---

## Self-Check: PASSED

- File `src/main/python/riscv/gtx/unit/csr/gspr.py`: contains 6 `GSPR_GTX_OPERAND` classes — FOUND
- File `src/main/python/riscv/gtx/unit/ins/encoding.py`: contains re-export block ending in `GSPR_GTX_OPERAND5: int = ...` — FOUND
- File `src/main/python/riscv/gtx/npu.py`: imports `_GSPR_OP3, _GSPR_OP5` — FOUND
- File `src/main/python/riscv/gtx/tloop_buffer.py`: import line 35 uncommented — FOUND
- File `src/main/python/riscv/gtx/unit/ins/ops/act.py`: `GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3` import uncommented — FOUND
- File `src/main/python/riscv/gtx/unit/ins/ops/vec.py`: `GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3` added — FOUND
- File `tests/gtx/test_custom_dispatch_chain.py`: `test_tloop_fast_path_opset_no_nameerror` appended — FOUND
- Commit `b228422`: `fix(gtx): restore GSPR_GTX_OPERAND0..5 register declarations` — FOUND
- Commit `b5700da`: `fix(gtx): realign 5 callsite imports for restored GSPR_GTX_OPERAND0..5` — FOUND
- Test gate: 24/24 PASS — VERIFIED
- Negative invariants: mm.py / spr.py / register_file.py untouched — VERIFIED
