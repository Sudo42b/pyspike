---
phase: quick-260514-ti0
plan: 01
subsystem: tests/gtx
tags: [test, csr, dispatch, fsm, post-refactor-invariant]
requires:
  - tests/gtx/conftest.py
  - tests/gtx/_mocks.py
  - riscv.gtx.unit.csr
  - riscv.gtx._registry
  - riscv.gtx.dispatch
  - riscv.gtx.decode
  - riscv.gtx.dispatch_state
  - riscv.gtx.execute
  - riscv.gtx.fsm
provides:
  - tests/gtx/test_csr_registry_chain.py
  - tests/gtx/test_custom_dispatch_chain.py
affects:
  - regression-protection for d6f73f9 architecture refactor
tech-stack:
  added: []
  patterns: [pytest-fixture-reuse, behaviour-level-invariant-tests]
key-files:
  created:
    - tests/gtx/test_csr_registry_chain.py
    - tests/gtx/test_custom_dispatch_chain.py
  modified: []
decisions:
  - "Each test asserts ONE invariant; multi-assert only when verifying the same atomic property (e.g. shape+dtype+zero-init)."
  - "Test 6 (RegisterView broadcast) deliberately uses 16-bit field (THREAD_MASK.mask) to avoid the register_file.py:188 OverflowError on 64-bit field writes — bug out of scope."
  - "Test 9 (state_execute None handler) uses _tloop_buf=None ctx setup to bypass the T-loop fast path and avoid the npu.py missing _GSPR_OP3/_GSPR_OP5 NameError — bug out of scope."
metrics:
  duration: ~12 min
  completed: 2026-05-14
  tasks: 2
  files: 2
  lines_added: 254
---

# Quick Task 260514-ti0: CSR Registry + custom0/1 Dispatch Chain Tests Summary

Add **18 invariant tests** under `tests/gtx/` locking the post-d6f73f9 architecture refactor: CSR `@csr` decorator → module-level `GSPR/NSPR/LSPR` dicts → `RegisterFile(int64 torch.Tensor)` chain (8 tests) and `_HANDLER_REGISTRY` → `collect_for_kind` → `build_custom0/1_table` → `resolve_for_context` → `state_decode`/`state_dispatch`/`state_execute` chain (10 tests).

## Test Counts

| File                                          | Tests | LOC | Budget |
| --------------------------------------------- | ----- | --- | ------ |
| `tests/gtx/test_csr_registry_chain.py`        | 8     | 99  | ≤ 130  |
| `tests/gtx/test_custom_dispatch_chain.py`     | 10    | 155 | ≤ 170  |
| **Total new**                                 | 18    | 254 | ≤ 300  |

## Commits

| Task | Hash      | Message                                                    |
| ---- | --------- | ---------------------------------------------------------- |
| 1    | `1fda59d` | test(quick-260514-ti0-01): add CSR registry + RegisterFile tensor chain coverage |
| 2    | `5c5bc67` | test(quick-260514-ti0-02): add custom0/1 dispatch chain coverage |

## Combined pytest Output (acceptance run)

```
collected 23 items

tests/gtx/test_csr_registry_chain.py::test_gspr_nspr_lspr_populated_at_import_time PASSED [  4%]
tests/gtx/test_csr_registry_chain.py::test_csr_decorator_produces_register_schema PASSED [  8%]
tests/gtx/test_csr_registry_chain.py::test_register_file_gspr_is_1024_int64_tensor PASSED [ 13%]
tests/gtx/test_csr_registry_chain.py::test_register_file_nspr_lspr_multidim_shapes PASSED [ 17%]
tests/gtx/test_csr_registry_chain.py::test_addr_by_name_pipe_only_masked_to_10_bits PASSED [ 21%]
tests/gtx/test_csr_registry_chain.py::test_register_view_attribute_write_broadcasts_across_nests PASSED [ 26%]
tests/gtx/test_csr_registry_chain.py::test_find_by_address_pipe_each_scope PASSED [ 30%]
tests/gtx/test_csr_registry_chain.py::test_find_by_address_pipe_missing_raises_keyerror PASSED [ 34%]
tests/gtx/test_custom_dispatch_chain.py::test_handler_registry_populated_after_ops_import PASSED [ 39%]
tests/gtx/test_custom_dispatch_chain.py::test_all_handlers_registered_with_universal_context PASSED [ 43%]
tests/gtx/test_custom_dispatch_chain.py::test_collect_for_kind_custom0_is_3level_dict PASSED [ 47%]
tests/gtx/test_custom_dispatch_chain.py::test_collect_for_kind_custom1_is_2level_dict PASSED [ 52%]
tests/gtx/test_custom_dispatch_chain.py::test_build_custom0_table_binds_npu_and_propagates_mnemonic PASSED [ 56%]
tests/gtx/test_custom_dispatch_chain.py::test_resolve_for_context_flattens_to_per_context_table PASSED [ 60%]
tests/gtx/test_custom_dispatch_chain.py::test_state_decode_extracts_funct7_funct3 PASSED [ 65%]
tests/gtx/test_custom_dispatch_chain.py::test_state_dispatch_resolves_handler_or_none PASSED [ 69%]
tests/gtx/test_custom_dispatch_chain.py::test_state_execute_handler_none_is_silent_nop_rd_zero PASSED [ 73%]
tests/gtx/test_custom_dispatch_chain.py::test_end_to_end_custom0_and_custom1_return_int PASSED [ 78%]
tests/gtx/test_fsm_smoke.py::test_npu_state_enum_has_five_members PASSED [ 82%]
tests/gtx/test_fsm_smoke.py::test_state_functions_are_importable_callables PASSED [ 86%]
tests/gtx/test_fsm_smoke.py::test_state_writeback_returns_idle PASSED    [ 91%]
tests/gtx/test_custom0_smoke.py::test_custom0_returns_int PASSED         [ 95%]
tests/gtx/test_custom1_smoke.py::test_custom1_returns_int PASSED         [100%]

============================== 23 passed in 7.52s ==============================
```

23/23 PASS, 0 failures, 0 errors. The 5 pre-existing smoke tests (`test_fsm_smoke.py` 3 + `test_custom0_smoke.py` 2) remain green — no regression introduced.

## What's Covered

### `test_csr_registry_chain.py` (8 tests)

1. `test_gspr_nspr_lspr_populated_at_import_time` — module-level dicts non-empty + named entries present.
2. `test_csr_decorator_produces_register_schema` — `NSPR["THREAD_MASK"]` is a `Register` with correct addr/width/rw_type/bus_type and a `mask=bits(0,15)` field.
3. `test_register_file_gspr_is_1024_int64_tensor` — `RegisterFile(GSPR, shape=(1024,), device='cpu')` materializes a zero-initialized int64 tensor.
4. `test_register_file_nspr_lspr_multidim_shapes` — `gtx_npu.{gspr,nspr,lspr}.tensor.shape` are `(1024,)`, `(4,1024)`, `(4,16,1024)`.
5. `test_addr_by_name_pipe_only_masked_to_10_bits` — `_addr_by_name` covers only PIPE entries; APB excluded; SPM_ADDRA addr = `0x900 & 0x3FF = 0x100`.
6. `test_register_view_attribute_write_broadcasts_across_nests` — vendor defaults seed THREAD_MASK.mask=0xFFFF across all 4 nests (stays on 16-bit field to avoid the OverflowError bug below).
7. `test_find_by_address_pipe_each_scope` — address routing for GSPR/NSPR/LSPR scopes by range.
8. `test_find_by_address_pipe_missing_raises_keyerror` — unused 0x099 slot raises KeyError, not silent None.

### `test_custom_dispatch_chain.py` (10 tests)

1. `test_handler_registry_populated_after_ops_import` — `_HANDLER_REGISTRY` size in `[50, 200]` after ops import.
2. `test_all_handlers_registered_with_universal_context` — every entry has `context=None` (d6f73f9 invariant).
3. `test_collect_for_kind_custom0_is_3level_dict` — `Dict[funct7, Dict[ctx, Dict[f3, fn]]]`, WRSPR=0x00 entry present, universal-context key `None` present.
4. `test_collect_for_kind_custom1_is_2level_dict` — `Dict[funct3, Dict[ctx, fn]]`, END_P=0b111 entry present.
5. `test_build_custom0_table_binds_npu_and_propagates_mnemonic` — `_bind` produces callables with `.gtx_mnemonic` attribute.
6. `test_resolve_for_context_flattens_to_per_context_table` — INITIAL_CONTEXT flattening produces `r0[funct7] -> {f3-or-None: handler}` and `r1[funct3] -> handler`.
7. `test_state_decode_extracts_funct7_funct3` — `state_decode` reads `insn.funct` into `funct7` and computes `funct3 = (xd<<2)|(xs1<<1)|xs2`, returns `DISPATCH`.
8. `test_state_dispatch_resolves_handler_or_none` — known WRSPR funct7 → bound handler; unknown 0x7F → handler=None (silent NOP path).
9. `test_state_execute_handler_none_is_silent_nop_rd_zero` — None handler keeps `rd=0`, returns `WRITEBACK`; T-loop buffering branch bypassed via `_tloop_buf=None`.
10. `test_end_to_end_custom0_and_custom1_return_int` — `gtx_npu.custom0/custom1` return `int` (RoCC reg_t contract); custom0 funct=0x4A (OPSET) routes through `run_pipeline`, custom1 funct3=END_P picks up the registered handler.

## Deviations from Plan

None - plan executed exactly as written. Both tasks committed atomically with the prescribed `test(quick-260514-ti0-*)` message prefix. LOC budgets respected (99/130, 155/170, 254/300). All 18 new + 5 existing tests pass on CUDA host.

## Open Notes for Successor

Two pre-existing bugs were observed during the writing of these tests but are **explicitly out of scope** per the plan constraints. They remain in the codebase untouched and are tracked here for the next maintenance pass:

### 1. `RegisterView.__setattr__` OverflowError on 64-bit field writes

**Location:** `src/main/python/riscv/gtx/unit/register_file.py:188`

**Reproducer:**
```python
gtx_npu.lspr.SPM_ADDRA.value = 0xDEADBEEFCAFEBABE   # raises OverflowError
# or any 64-bit-wide field write where field.mask == 0xFFFFFFFFFFFFFFFF
```

**Root cause:** Line 188 does
```python
self._tensor.copy_((self._tensor & ~(mask << shift)) | (new_val << shift))
```
With `mask=0xFFFFFFFFFFFFFFFF` and `shift=0`, `~(mask << shift)` evaluates as a Python `int` to `-0x10000000000000000` (i.e. `-(2**64)`). Casting that to int64 for the tensor op overflows the signed int64 range `[-2**63, 2**63-1]`.

**Fix candidates (pick one, verify with a regression test added under the same harness as Test 6):**
- Mask the complement explicitly to 64 bits: `~(mask << shift) & 0xFFFFFFFFFFFFFFFF`, then pass through `torch.as_tensor(..., dtype=torch.int64)`.
- Decompose into two tensor ops: clear via `self._tensor &= ...` (with the masked complement), then `self._tensor |= new_val << shift`.
- Promote the temporary path to torch ops end-to-end (`torch.bitwise_not(torch.tensor(mask << shift, dtype=torch.int64))` already wraps correctly).

**Impact assessment:** Tests in this plan deliberately use 16-bit `THREAD_MASK.mask` writes (Test 6) which sit well within the int64 range. Any code path that needs to set `SPM_ADDRA.value`, `STACK_INFO.value`, `MCAST_FAST_MODE.mcast_fast_mode` (full 64-bit field), etc. via the `RegisterView` attribute API will hit this — production code currently bypasses it by writing the underlying tensor directly (e.g. `gspr.tensor[addr] = value`). The bug is a silent correctness/usability trap, not an active failure.

### 2. `_GSPR_OP3` / `_GSPR_OP5` missing imports in `npu.py` T-loop fast-path

**Location:** `src/main/python/riscv/gtx/npu.py:238,240,264,265,269,270`

**Reproducer:**
```python
npu._tloop_buf = []          # arm the buffer
npu.warp.is_tloop = True     # enter T-loop window
npu.custom0(proc, insn, 0, 0)  # raises NameError: name '_GSPR_OP3' is not defined
```

**Root cause:** Six references to `_GSPR_OP3` and `_GSPR_OP5` appear in the T-loop fast-path inside `custom0` (lines 238, 240, 264, 265, 269, 270), but neither name is defined as a module-level constant in `npu.py` nor imported from `unit.ins.encoding` (or anywhere else). Likely a missed transcription during the d6f73f9 "Architecture Refactoring" — the pre-refactor code most likely had a `_GSPR_OP3 = 0x003` / `_GSPR_OP5 = 0x005` block near the top of `npu.py` (matching the addresses used in the `opset` handler at `unit/ins/ops/spr.py:105,107`).

**Fix candidate:** Add at the top of `npu.py` (next to the other `_TLOOP_*` private constants), or import from `unit.ins.encoding` if those constants are declared there:
```python
_GSPR_OP3 = 0x003
_GSPR_OP5 = 0x005
```

**Impact assessment:** The fast-path is gated by `buf is not None and self.warp.is_tloop`. The default `gtx_npu` fixture has `_tloop_buf=None` and `warp.is_tloop=False`, so the fast-path never fires during unit tests — including all 10 dispatch-chain tests in this plan. The bug is currently masked but will surface the first time a regression .elf actually enters T-loop with `_tloop_buf` armed (which the perf optimizations in commits `5dc7e47` / `f50a292` enabled). Verify against the pre-refactor `npu.py` (commit `d6f73f9` parent) to recover the exact constant values.

## Self-Check: PASSED

- File `tests/gtx/test_csr_registry_chain.py` exists (99 LOC).
- File `tests/gtx/test_custom_dispatch_chain.py` exists (155 LOC).
- Commit `1fda59d` present in `git log`.
- Commit `5c5bc67` present in `git log`.
- `uv run pytest tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py -v` → 23 passed.
