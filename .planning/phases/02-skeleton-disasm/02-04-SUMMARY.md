---
phase: 02-skeleton-disasm
plan: 04
subsystem: disasm
tags: [pybind11, rocc, riscv, gtx, disasm, registry, mask-match, namedtuple-fallback]

# Dependency graph
requires:
  - phase: 02-skeleton-disasm
    plan: 01
    provides: "_registry.handler decorator + _HANDLER_REGISTRY + ops/{spr,control}.py stubs + encoding (CUSTOM0_OPCODE/CUSTOM1_OPCODE)"
  - phase: 02-skeleton-disasm
    plan: 02
    provides: "ops/spr.py 4 SPR @handler registrations (wrspr/rdspr ISS+gem5)"
  - phase: 02-skeleton-disasm
    plan: 03
    provides: "ops/control.py 14 @handler registrations (8 warp custom1 + 6 custom0 stubs)"
provides:
  - "riscv.gtx.disasm.add_r_custom0(name, funct7) -> disasm_insn_t (R-type custom0, mask funct3-agnostic)"
  - "riscv.gtx.disasm.add_rf3_custom0(name, funct7, funct3) -> disasm_insn_t (R-type custom0 with funct3 sub-variant)"
  - "riscv.gtx.disasm.add_warp(name, funct3) -> disasm_insn_t (custom1 warp control, mask funct7-agnostic)"
  - "riscv.gtx.disasm.gtx_xrd / gtx_xrs1 / gtx_xrs2 arg_t formatters (xpr_name lookup)"
  - "riscv.gtx.disasm._PyDisasmInsn NamedTuple + _SentinelArg offline fallback for _riscv.so-less unit testing"
  - "riscv.gtx._registry.collect_disasms() real implementation (replaces plan 01 stub)"
  - "tests/gtx/test_disasm.py 10 tests covering formula correctness + registry integration + ROADMAP P2 #2 sample mnemonics"
affects: [02-05-integration, phase-03-dma, phase-04-mm, phase-05-vec-act, phase-06-verify]

# Tech tracking
tech-stack:
  added: []  # No new external dependencies
  patterns:
    - "Build-time vs offline disasm fallback (D-17 hybrid extension): try/except ImportError around riscv.disasm.disasm_insn_t -> _PyDisasmInsn NamedTuple sentinel"
    - "Lazy import inside collect_disasms() to dodge dispatch.py / _registry.py / disasm.py / encoding.py load-order chain"
    - "Direct verbatim port of C++ lambdas (gtx_npu_disasm.inc:23-36 add_r/add_rf3 + add_warp inline lambda) -> Python module-level helpers"

key-files:
  created:
    - "src/main/python/riscv/gtx/disasm.py -- 3 helpers + 3 arg formatters + _PyDisasmInsn fallback (128 lines)"
    - "tests/gtx/test_disasm.py -- 10 tests (5 formula + 5 registry-integration) (143 lines)"
  modified:
    - "src/main/python/riscv/gtx/_registry.py -- collect_disasms stub replaced with real walker (25 added, 3 removed)"

key-decisions:
  - "Offline _PyDisasmInsn fallback path is the unit-test path: tests run without _riscv.so via NamedTuple sentinels exposing .name/.match/.mask/.args"
  - "disasm_insn_t binding accepts py::args (positional varargs of arg_t) per riscv_disasm.cc:29-37 -- helpers pass gtx_xrd/gtx_xrs1/gtx_xrs2 positionally, no list wrapper needed"
  - "CUSTOM0_OPCODE / CUSTOM1_OPCODE constants reused from encoding.py rather than duplicating 0x0b / 0x2b literals (single source of truth)"
  - "Sample 5 P2 mnemonics (D-12 adapted) settled on ['wrspr','rdspr','wsplit_c0','wjoin_c0','warp_start_p'] -- the unambiguous custom0 firmware variants for wsplit/wjoin (custom1 'warp_split'/'warp_join' are also registered, both paths covered by test_collect_disasms_all_8_warp_mnemonics_present)"

patterns-established:
  - "When a future op module wants a disasm entry: add `mnemonic='name'` to the @handler decorator; collect_disasms() picks it up automatically. No edits to disasm.py needed -- formula helpers handle every (kind, mask_funct3) combination."
  - "Future phases adding new funct7 ops (P3 DMA, P4 MM, P5 VEC/ACT) will increase the count beyond 18 -- the test_collect_disasms_minimum_count assertion only sets a floor, not a ceiling. To track progress toward the ~140 ROADMAP target, future plans can grep `wc -l` of names in collect_disasms() output."

requirements-completed: [DISASM-01]

# Metrics
duration: 6m0s
completed: 2026-05-04
---

# Phase 02 Plan 04: Disasm Registration Layer Summary

**Wave 1 (parallel with 02-02 SPR / 02-03 warp-control): implements the disasm registration layer (DISASM-01) -- 3 mask/match helpers verbatim-ported from `gtx_npu_disasm.inc:23-36`, replaces the plan 01 `collect_disasms()` stub with a real builder that walks `_HANDLER_REGISTRY`, and adds 10 tests covering formula correctness + 18-entry registry integration + ROADMAP P2 #2 sample mnemonics.**

## Performance

- **Duration:** 6m0s
- **Started:** 2026-05-04T08:54:14Z
- **Completed:** 2026-05-04T09:00:14Z
- **Tasks:** 3 (all `type="auto" tdd="true"`)
- **Files changed:** 3 (1 modified, 2 created)

## Accomplishments

- **`riscv.gtx.disasm`** exports `add_r_custom0(name, funct7)`, `add_rf3_custom0(name, funct7, funct3)`, `add_warp(name, funct3)` -- direct port of the C++ lambdas in `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:23-36`. Each helper builds a `disasm_insn_t` (or offline `_PyDisasmInsn` fallback) carrying the standard 3 GPR arg formatters.
- **`gtx_xrd / gtx_xrs1 / gtx_xrs2`** arg formatters wired through `@isa.arg` when `_riscv.so` is available; reduced to `_SentinelArg` markers when offline. Production wheel always exercises the real path.
- **`_registry.collect_disasms()`** now walks `_HANDLER_REGISTRY` once and dispatches each mnemonic'd entry to the appropriate helper (replaces the plan-01 stub that returned `[]`). Lazy-imports the helpers from `.disasm` to dodge load-order issues.
- **Worked-example formula values verified end-to-end** for all 4 helpers:
  | Helper | Input | Match | Mask |
  |--------|-------|-------|------|
  | `add_r_custom0` | `'wrspr', 0x49` | `0x9200000B` | `0xFE00007F` |
  | `add_r_custom0` | `'wjoin', 0x03` | `0x0600000B` | `0xFE00007F` |
  | `add_rf3_custom0` | `'mm_s', 0x00, 0` | `0x0000000B` | `0xFE00707F` |
  | `add_warp` | `'warp_start_p', 0b110` | `0x0000602B` | `0x0000707F` |
- **18 disasm entries emitted** after Wave 1 lands (4 SPR + 8 warp + 6 custom0 stubs) -- exceeds ROADMAP P2 #2 ~10 threshold. ROADMAP P2 #2 sample 5 mnemonics (`wrspr`, `rdspr`, `wsplit_c0`, `wjoin_c0`, `warp_start_p`) all present.
- **Regression cleanliness:** all 62 tests in `tests/gtx/` pass (Phase 1 + Wave 0 + Wave 1 plans 02/03/04).

## Task Commits

Each task was committed atomically (with `--no-verify` for parallel-execution safety):

1. **Task 1: disasm.py -- match/mask helpers + arg formatters** -- `e6c28bb` (feat)
2. **Task 2: _registry.collect_disasms -- real implementation (replaces plan 01 stub)** -- `3babd10` (feat)
3. **Task 3: tests/gtx/test_disasm.py -- formulas + sample mnemonics + count** -- `7d4e76f` (test)

**Plan metadata commit:** to follow (this SUMMARY + STATE/ROADMAP/REQUIREMENTS updates).

## Files Created/Modified

### Created (2)
- `src/main/python/riscv/gtx/disasm.py` -- 128 lines. Three helpers + three arg formatters + `_PyDisasmInsn` NamedTuple fallback + `_build_insn` dispatcher. `_RISCV_DISASM_AVAILABLE` flag exported for downstream introspection.
- `tests/gtx/test_disasm.py` -- 143 lines, 10 tests:
  - 5 formula tests (`test_add_r_custom0_wrspr_formula`, `test_add_r_custom0_wjoin_formula`, `test_add_rf3_custom0_mm_s_formula`, `test_add_warp_start_p_formula`, `test_add_warp_join_formula`)
  - 5 registry-integration tests (`test_collect_disasms_minimum_count`, `test_collect_disasms_contains_p2_sample_5`, `test_collect_disasms_all_8_warp_mnemonics_present`, `test_collect_disasms_all_4_spr_mnemonics_present`, `test_collect_disasms_match_mask_unique_per_funct7`)

### Modified (1)
- `src/main/python/riscv/gtx/_registry.py` -- replaced the 6-line plan-01 stub `collect_disasms()` with the 28-line real implementation. Lazy-imports `add_r_custom0`/`add_rf3_custom0`/`add_warp` from `.disasm` inside the function body. No changes to the `_HANDLER_REGISTRY` list, `handler` decorator, or `collect_for_kind`.

## Decisions Made

1. **Offline path uses a `_PyDisasmInsn` NamedTuple, not a class+__init__** -- NamedTuple gives us free `.name/.match/.mask/.args` attributes plus immutability, exactly matching the read-only surface of the real `disasm_insn_t`. No mock-class boilerplate needed.
2. **Helpers call a private `_build_insn(name, match, mask)` dispatcher** rather than open-coding the available/fallback branch in each helper. Keeps the three helpers identical in shape -- only the formula differs.
3. **Mnemonic sample 5 settled on the unambiguous custom0 names** (`wsplit_c0`/`wjoin_c0`) per the plan/CONTEXT D-12 adaptation. The custom1 warp variants (`warp_split`/`warp_join`) are independently covered by `test_collect_disasms_all_8_warp_mnemonics_present`. ROADMAP P2 #2 list of 5 mnemonics (`wrspr`, `rdspr`, `wsplit`, `wjoin`, `warp_start_p`) is satisfied with EITHER choice -- both paths green.
4. **Helper signatures pass arg formatters positionally** (`disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)`), not as a list. Verified by reading `src/main/cpp/riscv_disasm.cc:29-37` -- `py_disasm_insn_t_create(name, match, mask, py::args py_args)` consumes varargs.

## Deviations from Plan

None -- plan executed exactly as written.

The plan's `<action>` block in Task 1 included a "verify at execution time" Python snippet asking us to inspect `disasm_insn_t.__init__` signature to decide between varargs and list-style arg passing. Since `_riscv.so` is not built in this environment, we resolved the question by reading the C++ source (`riscv_disasm.cc:29-37`) directly: the binding consumes `py::args` (positional varargs). The offline `_PyDisasmInsn` NamedTuple takes `args` as a single `Tuple[Any, ...]` field. Both paths are exercised by the helper code with no run-time signature dispatch.

## `_riscv.so` and Plan Status

- **`_riscv.so` was NOT available during this run.** All verification used the offline `_PyDisasmInsn` fallback path (`_RISCV_DISASM_AVAILABLE=False`). When the wheel is built and the production binding is loaded, the same helpers will produce real `disasm_insn_t` objects -- the formulas (match/mask) are identical between paths, only the holding type differs.
- **Plans 02 and 03 (parallel Wave 1) had landed by Task 2's verify run.** The 18-entry count matches the planned `4 SPR + 8 warp + 6 custom0 stubs` post-Wave-1 sum.

## Issues Encountered

- The shell wrapper for `pytest` truncates output to a single summary line ("Pytest: 10 passed" / "Pytest: No tests collected"). The plan's `<verify>` blocks invoke `pytest tests/gtx/test_disasm.py -x -q --noconftest -o "addopts="` directly. Running via `python3 -m pytest ...` produces the full pytest report (10 collected, 10 passed, 0.42s). Both yield the same exit code; the wrapper just hides intermediate output. This is environmental, not plan-related.

## Disasm Entry List (after Wave 1)

The exact 18 mnemonics emitted by `_registry.collect_disasms()`:

```
['dispatch_act', 'dispatch_dma', 'dispatch_mm', 'dispatch_vec',
 'rdspr', 'rdspr_gem5',
 'warp_end_p', 'warp_end_s', 'warp_end_t', 'warp_join', 'warp_split',
 'warp_start_p', 'warp_start_s', 'warp_start_t',
 'wjoin_c0', 'wrspr', 'wrspr_gem5', 'wsplit_c0']
```

Breakdown:
- **4 SPR funct7** (plan 02): `wrspr` (0x49), `rdspr` (0x48), `wrspr_gem5` (0x00), `rdspr_gem5` (0x01)
- **8 warp funct3** (plan 03): `warp_start_t` (0b000), `warp_end_t` (0b001), `warp_start_s` (0b010), `warp_end_s` (0b011), `warp_split` (0b100), `warp_join` (0b101), `warp_start_p` (0b110), `warp_end_p` (0b111)
- **6 custom0 stubs** (plan 03): `wsplit_c0` (0x02), `wjoin_c0` (0x03), `dispatch_mm` (0x04), `dispatch_vec` (0x05), `dispatch_act` (0x06), `dispatch_dma` (0x07)

## ROADMAP P2 Success Criterion 2 Status: COVERED

> "Phase 2: get_disasms() structure + ~10 SPR/control entries registered; full ~140 is P5/P6 cumulative target"

- ✅ `_registry.collect_disasms()` returns 18 entries (>>10 threshold)
- ✅ All 5 sample mnemonics from D-12 adaptation present
- ✅ Formula correctness verified against C++ ground truth (4 worked examples match research §537-555)
- ✅ ROADMAP threshold criterion (~10) is satisfied with 80% headroom

## Cross-Phase Hand-off

Future phases adding new funct7 / funct3 ops just need to:
1. Define a handler with `@handler(kind=..., funct7=..., mnemonic='name')` -- `mnemonic` triggers automatic disasm entry generation.
2. For ops needing funct3 sub-variant masking (P4 MM `add_rf3` cases), add `mask_funct3=True` to the decorator.
3. No changes to `disasm.py` or `_registry.collect_disasms()` needed -- the helpers and walker handle every (kind, mask_funct3) combination.

`GtxNpu.get_disasms()` is now wired through `_registry.collect_disasms()` (per plan 01 surface) -- once `_riscv.so` is built, calling it from a real spike processor returns the same 18 entries as `disasm_insn_t` instances.

## Self-Check: PASSED

Verified files exist:
- `src/main/python/riscv/gtx/disasm.py` -- FOUND (128 lines)
- `src/main/python/riscv/gtx/_registry.py` -- FOUND (modified, real `collect_disasms`)
- `tests/gtx/test_disasm.py` -- FOUND (143 lines, 10 tests)

Verified commits exist (`git log --oneline | grep`):
- `e6c28bb` feat(02-04): add disasm.py match/mask helpers for GTX RoCC ops -- FOUND
- `3babd10` feat(02-04): replace _registry.collect_disasms stub with real builder -- FOUND
- `7d4e76f` test(02-04): add disasm formula + collect_disasms verification suite -- FOUND

Verified acceptance commands all pass:
- `python3 -m pytest tests/gtx/test_disasm.py -x --noconftest -o "addopts="` -> 10 passed in 0.42s
- `python3 -m pytest tests/gtx/ -x --noconftest -o "addopts="` -> 62 passed in 0.73s (no regressions)
- `python3 -c "from riscv.gtx.disasm import add_r_custom0; e = add_r_custom0('wrspr', 0x49); assert e.match == 0x9200000B and e.mask == 0xFE00007F"` -> exit 0
- `python3 -c "from riscv.gtx import _registry; from riscv.gtx.ops import spr, control; assert len(_registry.collect_disasms()) >= 18"` -> exit 0

---
*Phase: 02-skeleton-disasm*
*Completed: 2026-05-04*
