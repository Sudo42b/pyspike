---
phase: 260514-vwp
plan: 01
subsystem: gtx/unit/register_file
tags: [bugfix, csr, register-view, int64, regression-test]
requires:
  - "src/main/python/riscv/gtx/unit/register_file.py:RegisterView.__setattr__ (field branch)"
provides:
  - "RegisterView field setter that accepts 64-bit unsigned values without OverflowError"
  - "Two regression tests pinning the new behavior"
affects:
  - "All callsites writing 64-bit CSR fields (LSPR SGPR0..127.gpr, vec write-back, GSPR operand staging)"
tech-stack:
  added: []
  patterns:
    - "Signed-int64 wrap of unsigned Python ints before torch.as_tensor(dtype=int64)"
key-files:
  created: []
  modified:
    - "src/main/python/riscv/gtx/unit/register_file.py (lines 180-195, field branch only)"
    - "tests/gtx/test_csr_registry_chain.py (appended 2 tests, lines 102-126)"
decisions:
  - "Inline signed-int64 wrap (per CONTEXT.md locked decision) — no helper fn"
  - "Extended wrap to `value` operand (Rule 1 deviation, surfaced after first fix attempt)"
metrics:
  duration: "~6 min"
  completed: "2026-05-14"
  tests: "24 → 26 PASS (2 new regression tests)"
  commit: "b5df4a0"
---

# Quick Task 260514-vwp Summary

One-liner: Fixed `RegisterView.__setattr__` 64-bit field setter by wrapping
both `(mask << shift)` and unsigned 64-bit `value` to their signed-int64
images before any torch int64 cast — closes `260514-ti0` Open Notes #1.

## Before / After — register_file.py:180-189 → 180-195

**Before (broken on 64-bit fields):**
```python
if name in self._reg.fields:
    field = self._reg.fields[name]
    # Bit manipulation via tensor ops
    mask = field.mask
    shift = field.shift

    # (tensor & ~(mask << shift)) | ((value & mask) << shift)
    new_val = torch.as_tensor(value, dtype=torch.int64) & mask
    self._tensor.copy_((self._tensor & ~(mask << shift)) | (new_val << shift))
    return
```

**After (signed-int64 wrap on both mask and value):**
```python
if name in self._reg.fields:
    field = self._reg.fields[name]
    mask = field.mask
    shift = field.shift

    # Reinterpret the shifted mask as a signed int64 to avoid
    # Python's arbitrary-precision negative result from
    # `~(mask << shift)` — torch cannot cast that back into
    # int64 (OverflowError). See CONTEXT.md root_cause.
    u64 = (mask << shift) & ((1 << 64) - 1)
    shifted_mask = u64 - (1 << 64) if u64 >> 63 else u64

    # Same signed-int64 wrap for `value` when it is a Python int
    # with the top bit set (e.g. 0xCAFEBABEDEADBEEF) — torch
    # rejects unsigned >= 2^63 in int64 dtype.
    if isinstance(value, int):
        v64 = value & ((1 << 64) - 1)
        value = v64 - (1 << 64) if v64 >> 63 else v64

    new_val = torch.as_tensor(value, dtype=torch.int64) & mask
    self._tensor.copy_((self._tensor & ~shifted_mask) | (new_val << shift))
    return
```

Net change: +16 -4 in this file. Confined to the field branch — the
`_`-prefix branch (171-174), `value` branch (176-178), and trailing
`super().__setattr__` are byte-identical to pre-fix.

## New Regression Tests — tests/gtx/test_csr_registry_chain.py

**1. `test_register_view_64bit_field_broadcast_write_no_overflow`** (lines 102-119)
   Key assertions:
   - `gtx_npu.lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF` does NOT raise.
   - `stored = gtx_npu.lspr.tensor[..., 0x000]` has shape `(4, 16)`.
   - `(stored == signed).all().item()` where `signed = 0xCAFEBABEDEADBEEF - (1 << 64)`.
   - `int(gtx_npu.lspr[0][0].SGPR0.gpr) & 0xFFFFFFFFFFFFFFFF == 0xCAFEBABEDEADBEEF`
     (round-trip through `__getattr__` field path).

**2. `test_register_view_partial_field_high_bits_preserves_low_bits`** (lines 122-126)
   Key assertions:
   - `gtx_npu.nspr.THREAD_MASK.mask = 0xABCD`
   - `(gtx_npu.nspr.THREAD_MASK._tensor & 0xFFFF).tolist() == [0xABCD] * 4`

Net change in test file: +25 -0 (pure append, no edits to existing tests).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended signed-int64 wrap to the `value` operand**
- **Found during:** Task 1 verification (first pytest run after applying
  CONTEXT.md verbatim fix).
- **Issue:** CONTEXT.md root_cause analysis only flagged
  `~(mask << shift)`. After fixing that, `torch.as_tensor(value=0xCAFEBABEDEADBEEF, dtype=torch.int64)`
  still raised `ValueError: Overflow when unpacking long long` because
  the value itself exceeds int64 max (2^63 - 1).
- **Fix:** Added a symmetric 4-line block wrapping `value` to its
  signed-int64 image when it is a Python `int` with the top bit set.
  Skipped when `value` is non-int (already a tensor / numpy array etc.)
  so torch's own coercion path is preserved.
- **Files modified:** src/main/python/riscv/gtx/unit/register_file.py (lines 191-195)
- **Commit:** b5df4a0
- **Justification:** Required by `must_haves.truths[0]` — the plan
  explicitly demands `lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF` execute
  without OverflowError. CONTEXT.md verbatim block alone was
  insufficient.

The fix preserves every Karpathy guideline cited in CONTEXT.md (§2
simplicity — inline 4-line wrap, no helper fn; §3 surgical — confined to
the field branch; consistent with the `shifted_mask` pattern just
established).

## Verification

Command (verbatim from `<constraints>`):
```
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v
```

Result: **26 passed in 8.31s** (24 pre-existing + 2 new regression tests).
No failures, no errors, no new xfails, no warnings introduced.

Spot check:
```
uv run python -c "from riscv.gtx.npu import GtxNpu; ... n.lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF; print('ok')"
→ ok
```

## Closed Items

- **Quick task 260514-ti0 "Open Notes for Successor" #1** —
  `register_file.py:188 OverflowError on 64-bit field writes` is now
  closed. SGPR0..127.gpr, GSPR_GTX_OPERAND0..5.value, and every other
  64-bit CSR field is freely writable through the RegisterView
  attribute setter.

## Commit

```
b5df4a0  fix(gtx): RegisterView 64-bit field setter — wrap shifted mask through signed int64
```

Single atomic commit, exactly two files staged
(register_file.py + test_csr_registry_chain.py).

## Self-Check: PASSED

- src/main/python/riscv/gtx/unit/register_file.py — modified, FOUND
- tests/gtx/test_csr_registry_chain.py — modified, FOUND
- commit b5df4a0 — FOUND in git log
- 26 passed in 8.31s — verified
