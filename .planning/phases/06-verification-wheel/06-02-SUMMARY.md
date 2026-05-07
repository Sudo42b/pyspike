---
phase: 06-verification-wheel
plan: 02
subsystem: gtx
tags: [atexit, ddr-dump, p5-deferred-closure, vendor-port, single-global, parallel-wave-1a]
requires:
  - "P3 D-09 args-only ddr_dump_to_file (preserved verbatim — wrapper-only addition)"
  - "P5 nop_wjoin.elf fixture (subprocess primary signal)"
  - "P5 test_regression_fw_act.py 5-tier graceful-skip pattern (transitioned)"
  - "vendor gtx_npu_core.cc:59,61-73,127 (single-global + atexit + registration ports)"
provides:
  - "riscv.gtx.npu._LAST_NPU module-level singleton (vendor g_gtx_instance direct port)"
  - "riscv.gtx.ddr._atexit_ddr_dump (env-var-aware wrapper around args-only ddr_dump_to_file)"
  - "riscv.gtx.__init__ conditional atexit.register gated on os.getenv('GTX_DDR_DUMP')"
  - "tests/gtx/test_atexit_ddr_dump.py PRIMARY atexit signal (subprocess + GTX_DDR_DUMP env vars → dump file)"
  - "tests/gtx/test_regression_fw_act.py tier #5 → hard PASS (P5 deferred infra closed)"
affects:
  - "Plan 04 (VRF-04 regression matrix) — atexit hook now wired so .elf regressions can produce dump files"
tech-stack:
  added: []
  patterns:
    - "single-global last-instance-wins NPU lookup (Option C per RESEARCH §NPU Instance Lookup; v1 single-hart)"
    - "lazy module attribute lookup (`from . import npu as _m; _m._LAST_NPU`) to read live value at hook-fire time"
    - "args-only function + env-var-aware wrapper separation (P3 D-09 lineage extended)"
    - "ensure_ddr-on-dump to preserve vendor 'always has DDR' semantics under pyspike lazy-allocation"
key-files:
  created:
    - "tests/gtx/test_atexit_ddr_dump.py (122 lines, 2 tests, D-18 ZERO-OVERLAP — Plan 02 owns creation)"
  modified:
    - "src/main/python/riscv/gtx/npu.py (+11 lines: _LAST_NPU module global + __init__ assignment)"
    - "src/main/python/riscv/gtx/ddr.py (+59 lines: _atexit_ddr_dump function with ensure_ddr + live module-attr lookup)"
    - "src/main/python/riscv/gtx/__init__.py (+15 lines: import os/atexit + conditional register block)"
    - "tests/gtx/test_regression_fw_act.py (~17 lines: tier #5 graceful-skip → hard assert with PRIMARY/SECONDARY signal comment)"
decisions:
  - "Adopted RESEARCH §NPU Instance Lookup Option C (single-global) over Option A (WeakValueDictionary) and Option B (PythonBridge.references) per CONTEXT D-05 v1 single-hart scope. Vendor parity (g_gtx_instance) primary driver."
  - "Live module-attribute lookup pattern (`from . import npu as _m; _m._LAST_NPU`) instead of `from .npu import _LAST_NPU` — the latter binds the value at IMPORT time (None) and misses GtxNpu.__init__ assignment."
  - "Added `ensure_ddr(last_npu.mem, end_offset)` inside `_atexit_ddr_dump` to materialize the requested dump range when firmware never touches DDR (e.g. nop_wjoin.elf). Vendor C++ pre-allocates DDR in constructor so has_ddr() is always true; pyspike lazy-allocates per P3 D-13 — this restores vendor parity at the dump path only (no impact on hot-path)."
metrics:
  duration: "~25 minutes"
  date: 2026-05-07
---

# Phase 6 Plan 02: GTX_DDR_DUMP atexit hook Summary

Wired the GTX_DDR_DUMP atexit hook end-to-end as a 1:1 port of vendor `gtx_npu_core.cc:59,61-73,125-127` (single-global instance pointer + atexit registration). Subprocess `pyspike --extension=gtx <elf>` with the 3 GTX_DDR_DUMP env vars now produces a DDR hex dump at WJOIN/SystemExit(0). Closes the P5 deferred infrastructure that prevented `test_regression_fw_act.py` from hard-PASSing tier #5.

## What Changed

### Source modifications (3 files, +85 lines)

1. **`src/main/python/riscv/gtx/npu.py`** — Added module-level `_LAST_NPU = None` singleton (vendor `g_gtx_instance`) immediately before the `@isa.register("gtx")` decorator, and `global _LAST_NPU; _LAST_NPU = self` as the LAST line of `GtxNpu.__init__` (NOT in `reset()` — vendor `g_gtx_instance` likewise persists across resets). Last-instance-wins semantics; v1 single-hart scope.

2. **`src/main/python/riscv/gtx/ddr.py`** — Appended `_atexit_ddr_dump()` function. Wraps the existing args-only `ddr_dump_to_file` (P3 D-09 invariant preserved). Reads `GTX_DDR_DUMP` (path), `GTX_DDR_DUMP_ADDR` (hex int, default `0x37f000000`), `GTX_DDR_DUMP_SIZE` (hex int, default `0x400`) at hook-fire time. Calls `_LAST_NPU.flush_deferred_ddr_stores()` first (P3 D-05 lineage). Calls `ensure_ddr(last_npu.mem, end_offset)` to materialize the requested range under pyspike lazy-allocation. Uses `from . import npu as _npu_mod; last_npu = _npu_mod._LAST_NPU` to capture the LIVE singleton value (not a stale None binding from `from .npu import _LAST_NPU`). Safe-failure semantics: all error paths use early `return` and `print(..., file=sys.stderr)` — never raises out of the atexit handler.

3. **`src/main/python/riscv/gtx/__init__.py`** — Added `import os; import atexit` at top. After the `try: from . import npu` block, added the conditional registration: `if os.getenv('GTX_DDR_DUMP'): from .ddr import _atexit_ddr_dump; atexit.register(_atexit_ddr_dump)`. Direct port of vendor `gtx_npu_core.cc:125-127`. The env var is read ONCE at module import time — tests that toggle `GTX_DDR_DUMP` per-call MUST use `subprocess.run(..., env=env)` (each subprocess re-evaluates the gate).

### Test changes (1 created + 1 modified)

4. **`tests/gtx/test_atexit_ddr_dump.py`** — CREATED FROM SCRATCH (D-18 zero-overlap with Wave 1a sibling Plan 01). 122 lines. PRIMARY atexit signal:
   - `test_atexit_dump_fires_on_systemexit`: subprocess pyspike + `nop_wjoin.elf` + 3 env vars (`GTX_DDR_DUMP`, `GTX_DDR_DUMP_ADDR=0x0`, `GTX_DDR_DUMP_SIZE=0x20`) → returncode 0 + dump file exists + 1 hex data line of 64 chars (32 bytes zero-padded).
   - `test_atexit_dump_does_not_register_when_env_unset`: subprocess pyspike without env vars → no dump file at suspicious path.

5. **`tests/gtx/test_regression_fw_act.py`** — Modified tier #5. Replaced the `pytest.skip("GTX_DDR_DUMP not honored ...")` block (lines 154-162) with `assert actual_dump.exists(), "P6 D-04 broken: ..."` and a PRIMARY/SECONDARY signal comment (per Plan 02 INFO #4 requirement).

## How

**Vendor port (D-04, D-05, D-06):**
- `gtx_npu_core.cc:59` `static gtx_npu_t *g_gtx_instance = nullptr;` → `_LAST_NPU = None` at module scope in `npu.py`.
- `gtx_npu_core.cc:61-73` `gtx_atexit_ddr_dump()` body → `_atexit_ddr_dump()` in `ddr.py` (lazy `last_npu` lookup, `flush_deferred_ddr_stores()`, env-var hex parse with vendor defaults `0x37f000000` / `0x400`, args-only `ddr_dump_to_file` call).
- `gtx_npu_core.cc:127` `std::atexit(gtx_atexit_ddr_dump)` → `atexit.register(_atexit_ddr_dump)` in `__init__.py`, gated on `os.getenv('GTX_DDR_DUMP')` per CONTEXT D-04.

**Single-NPU model (CONTEXT D-05, RESEARCH §NPU Instance Lookup Option C):** Vendor C++ uses a process-global `g_gtx_instance` pointer reassigned in the constructor (last-instance-wins). v1 pyspike has the same single-hart scope. Tests that need per-instance isolation use subprocess. Option A (WeakValueDictionary) and Option B (PythonBridge.references) deferred to v2 if multi-hart support lands.

## Deviations from Plan

### Auto-fixed Issues (committed in `f7140f4`)

**1. [Rule 1 - Bug] Stale `_LAST_NPU` binding in `_atexit_ddr_dump`**
- **Found during:** Task 1 verification — primary atexit test failed with "DUMP MISSING" despite atexit hook registering correctly.
- **Issue:** The original Plan 02 action used `from .npu import _LAST_NPU`. Python's `from X import Y` binds the value at import time. Since `_LAST_NPU` is initially `None` (set later by `GtxNpu.__init__`), the binding inside `_atexit_ddr_dump` permanently held `None` even after a GtxNpu was instantiated.
- **Fix:** Switched to `from . import npu as _npu_mod; last_npu = _npu_mod._LAST_NPU` to read the LIVE module attribute at hook-fire time. RESEARCH §Pitfall 1 captures this Python anti-pattern; the plan's action snippet did not flag it explicitly so it slipped through.
- **Files modified:** `src/main/python/riscv/gtx/ddr.py`
- **Commit:** `f7140f4`

**2. [Rule 2 - Critical functionality] `has_ddr()` parity under pyspike lazy DDR allocation**
- **Found during:** Task 2 verification — primary atexit test still failed: subprocess executed cleanly (rc=0), atexit hook fired, `_LAST_NPU` was correctly set, but `_LAST_NPU.mem._ddr_bytes is None` so the early-return branch (`if last_npu.mem._ddr_bytes is None: return`) blocked the dump.
- **Root cause:** Vendor C++ pre-allocates DDR in `gtx_npu_t` constructor (`gtx_npu_core.cc:167` `ensure_ddr()` in `reset()`), so `has_ddr()` is always true after instantiation. Pyspike lazy-allocates per P3 D-13 doubling-grow ergonomic — for clean firmware (`nop_wjoin.elf`) that never touches DDR, `_ddr_bytes` stays `None`.
- **Fix:** Added `ensure_ddr(last_npu.mem, end_offset)` inside `_atexit_ddr_dump` immediately before `ddr_dump_to_file`. Materializes the requested dump range as zeros, restoring vendor "always has DDR" parity at the dump path only (zero impact on hot-path execution). Removed the `_ddr_bytes is None` early-return since the wrapper now guarantees allocation.
- **Files modified:** `src/main/python/riscv/gtx/ddr.py`
- **Commit:** `f7140f4`

## Verification Results

```
$ python3 -m pytest tests/gtx/test_atexit_ddr_dump.py tests/gtx/test_regression_fw_act.py \
  tests/gtx/test_dma_roundtrip.py tests/gtx/test_ddr_modes.py \
  tests/gtx/test_op_act.py tests/gtx/test_op_vec.py tests/gtx/test_regression_fw_mm.py \
  -v --no-header -o "addopts="

============================== 53 passed in 4.43s ==============================
```

Highlights:
- `test_atexit_dump_fires_on_systemexit` — **PASS** (PRIMARY signal — subprocess pyspike + nop_wjoin.elf produces 32-byte zero-padded dump as expected)
- `test_atexit_dump_does_not_register_when_env_unset` — **PASS**
- `test_act_strict_mode_pass` — **PASS** (SECONDARY signal — was tier #5 graceful-skip in P5; now hard PASS in P6)
- `test_mm_basic_strict_mode_pass` (P4) — **PASS** (no regression)
- All 49 P3-P5 unit tests — **PASS** (no regression)

Smoke imports:
- `python3 -c "import riscv.gtx; print('OK')"` (no env var) — **OK** (no atexit registered)
- `GTX_DDR_DUMP=/tmp/x.hex GTX_DDR_DUMP_ADDR=0x0 GTX_DDR_DUMP_SIZE=0x20 python3 -c "import riscv.gtx; print('OK')"` — **OK** (atexit registered + dump produced at interpreter shutdown)

## D-18 Zero-Overlap Confirmation

Files touched by Plan 02 (per `git log -3 --name-only --format=""`):
- `src/main/python/riscv/gtx/__init__.py` (Plan 02 owns)
- `src/main/python/riscv/gtx/ddr.py` (Plan 02 owns)
- `src/main/python/riscv/gtx/npu.py` (Plan 02 owns)
- `tests/gtx/test_atexit_ddr_dump.py` (Plan 02 owns CREATION — Plan 01 did not stub this)
- `tests/gtx/test_regression_fw_act.py` (Plan 02 modifies tier #5 only)

Plan 02 did NOT touch:
- `src/main/python/riscv/gtx/_verify.py` (Plan 01)
- `pyproject.toml` (Plan 01/05)
- `tests/gtx/test_verify_*.py` (Plan 01)
- `scripts/import_vendor_golden.py` (Plan 03)
- `tests/gtx/data/*` (Plan 03)
- `tests/gtx/test_assets_present.py` (Plan 03)

Zero overlap with Wave 1a siblings — confirmed.

## Self-Check: PASSED

Verified post-write:
- FOUND: `tests/gtx/test_atexit_ddr_dump.py` (created)
- FOUND: `e6e3b93` (Task 1 commit)
- FOUND: `f7140f4` (Task 1 deviation fix commit)
- FOUND: `459ad28` (Task 2 commit)

## Commits

- `e6e3b93` `feat(06-02): wire _LAST_NPU + _atexit_ddr_dump + import-time atexit registration`
- `f7140f4` `fix(06-02): _atexit_ddr_dump lookup live _LAST_NPU + ensure_ddr for has_ddr() parity`
- `459ad28` `test(06-02): add atexit dump primary signal tests + transition test_regression_fw_act tier #5 to hard PASS`
