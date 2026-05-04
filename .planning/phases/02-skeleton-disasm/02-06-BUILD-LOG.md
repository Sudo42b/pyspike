---
plan: 02-06
type: build-log
created: 2026-05-04T15:49:05Z
host: DESKTOP-ADRHA0T
kernel: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
status: build-succeeded-tests-revealed-pre-existing-bugs
outcome: partial
---

# Plan 02-06 Gap-Closure Build & Test Log

Captures evidence for the 4 tasks in `02-06-PLAN.md` (Wave 3 gap closure).
This log is the source of truth that 02-VERIFICATION.md, 02-HUMAN-UAT.md, and
02-06-SUMMARY.md cite by reference.

---

## Task 1 — Build `_riscv.so` via `pip install -e .`

**Started:** 2026-05-04T15:49:05Z
**Outcome:** SUCCESS

### Step 1.1 — Pre-flight checks

| Check | Command | Result |
|-------|---------|--------|
| Python | `python3 --version` | `Python 3.10.12` |
| pybind11 | `pip show pybind11` | `Version: 3.0.1` (NOT the broken 3.0.4) |
| RISC-V toolchain | `/opt/riscv/bin/riscv64-unknown-elf-gcc --version` | `riscv64-unknown-elf-gcc (GCC) 15.2.0` |
| libriscv.so | `ls src/main/python/riscv/data/lib/libriscv.so` | present (292.9M; built by Phase 1) |

All four pre-flight checks pass. Phase-1 spike core was already built — `_build_spike()` in setup.py:101 short-circuits on existing `data/include/riscv/encoding.h`.

### Step 1.2 — Build via editable install

**Initial attempt failed** with `ModuleNotFoundError: No module named 'setuptools_scm'` because `--no-build-isolation` skips fetching build dependencies. Resolved with:

```bash
python3 -m pip install --user setuptools_scm
# Successfully installed setuptools_scm-10.0.5 vcs-versioning-1.1.1
```

(Treated as a Rule 3 - Blocking deviation: build dependency missing.)

**Successful build command:**
```bash
python3 -m pip install -e . --no-build-isolation --user
```

**Final 16 lines of `/tmp/02-06-pip-install.log`:**
```
Obtaining file:///mnt/e/14_NIGHTLY/pyspike
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: numpy<3,>=2.0 in /home/sw.lee/.local/lib/python3.10/site-packages (from spike==0.0.5.dev85) (2.2.6)
Building wheels for collected packages: spike
  Building editable for spike (pyproject.toml): started
  Building editable for spike (pyproject.toml): still running...
  Building editable for spike (pyproject.toml): still running...
  Building editable for spike (pyproject.toml): finished with status 'done'
  Created wheel for spike: filename=spike-0.0.5.dev85-0.editable-cp310-cp310-linux_x86_64.whl size=13518 sha256=e9b2b26676ce49879772aadc6323b78b9ff54bffb306afd340c2a4ed4509a5bd
  Stored in directory: /tmp/pip-ephem-wheel-cache-ez6fzogi/wheels/df/18/fd/bc17a9ba44b4d263f3ccdd9d6b65319ddaa51e9cdff7baeb22
Successfully built spike
Installing collected packages: spike
Successfully installed spike-0.0.5.dev85
```

**Exit code:** 0

**Build artifact:** `src/main/python/riscv/_riscv.cpython-310-x86_64-linux-gnu.so` (1.5M)

### Step 1.3 — Verify `_riscv` import resolves

```bash
$ python3 -c "from riscv import _riscv; print(_riscv.__file__)"
/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/_riscv.cpython-310-x86_64-linux-gnu.so
```
Exit code 0.

### Step 1.4 — Verify `GtxNpu` hydrates

```bash
$ python3 -c "from riscv.gtx import GtxNpu; assert GtxNpu is not None; print(GtxNpu)"
<class 'riscv.isa.register.<locals>.isa_decorator.<locals>.MyISA'>
```
Exit code 0. The class is the decorator-synthesized wrapper around `GtxNpu` produced by `@isa.register('gtx')` — confirming the registration path works end-to-end.

### Step 1.5 — Failure-mode discriminant

| Mode | Description | Triggered? |
|------|-------------|------------|
| F1   | pybind11 csr_t static_assert (deferred-items.md issue resurface) | NO |
| F2   | Linker error `cannot find -lriscv` | NO |
| F3   | Import OK but `GtxNpu is None` | NO |

**Note:** pybind11 3.0.4 issue (deferred-items.md) was avoided because the system has 3.0.1 installed and `--no-build-isolation` reused it. CI / cibuildwheel still needs a `pyproject.toml` `[build-system].requires` pin if the latest pybind11 is breaking — Phase-1 deferred-items concern remains valid for reproducibility, but for this gap-closure cycle the local dev environment is unblocked.

### Task 1 Summary

- `_riscv.cpython-310-x86_64-linux-gnu.so` built successfully (1.5M).
- `from riscv import _riscv` resolves.
- `from riscv.gtx import GtxNpu` resolves to a real class (was `None` due to graceful-degradation fallback before build).
- One auto-fixed deviation (Rule 3 - Blocking: missing `setuptools_scm` build dep — fix-attempt count: 1).

---

## Task 2 — Run all skipif-gated tests + zero-regression check

**Started:** 2026-05-04T15:55:00Z
**Outcome:** PARTIAL — skips resolved (0 skipped) but 15 tests now fail

### Step 2.2 — Full gtx suite count

```bash
$ python3 -m pytest tests/gtx/ -q -o "addopts=" | tail -1
15 failed, 71 passed in 3.06s
```

**Skip count: 0** (gap-closure assertion satisfied — all 21 previously-skipped tests now run.)

**Failure count: 15** — the build of `_riscv.so` exposed pre-existing test/production
bugs that the mock-fallback discipline (D-17/D-18/D-19) had hidden.

### Step 2.3 — Per-file accounting

| File | Expected | Actual | Delta |
|------|----------|--------|-------|
| test_register.py | 5 passed | **5 passed** | OK |
| test_reset.py | 8 passed | **8 failed** | REGRESSION |
| test_dispatch.py | 9 passed | **9 passed** | OK |
| test_skeleton.py | 2 passed | **1 failed, 1 passed** | REGRESSION |
| test_disasm.py | 10 passed | **6 failed, 4 passed** | REGRESSION |
| test_warp.py | 16 passed | **16 passed** | OK |
| test_spr.py | 16 passed | **16 passed** | OK |
| test_wjoin.py | 7 passed | **7 passed** | OK |
| **Total** | **86 passed** | **71 passed, 15 failed, 0 skipped** | **15 regressions** |

### Failure Category Analysis

**Category A — Mock-fallback test bug (test_reset.py, 8 failures)**

Root cause: `npu.reset(proc)` calls `super().reset(proc)` (line 74 of `npu.py`).
When `_riscv.so` is built, `super()` is the C++ `extension_t` class whose `reset()`
is strictly typed `(self, proc: processor_t) -> None`. The test passes a
`MockProcessor` instance which is NOT a `processor_t` subclass:

```
TypeError: reset(): incompatible function arguments. The following argument types are supported:
    1. (self: riscv._riscv.extension.extension_t, proc: processor_t) -> None
Invoked with: <riscv.isa.register...MyISA object>, MockProcessor(...)
```

The mock was indistinguishable from `processor_t` while `super().reset` was a
Python no-op. With the real C++ binding, the mock now fails strict type check.

**Resolution path (out of plan 02-06 scope):**
- Option 1: Remove `super().reset(proc)` from `npu.py:74` (it is a no-op upstream — see
  `vendor/spike/riscv/extension.h:18: virtual void reset(processor_t &) {};`).
  This is a 1-line fix to `src/main/python/riscv/gtx/npu.py` but that file is
  Wave 0/1 owned and forbidden by the plan's scope discipline.
- Option 2: Modify tests to bypass `super()` by skipping the reset() call when
  detecting the C++ type strictness. But test_reset.py is also Wave 2 owned.
- **Recommendation:** Phase 2 follow-up plan (02-07 or roll into evolve) — remove
  the no-op `super().reset(proc)` and re-run.

**Category B — disasm name-normalization mismatch (test_disasm.py, 6 failures)**

Root cause: `riscv.disasm.disasm_insn_t` C++ constructor normalizes mnemonic
underscores to dots — verified directly:

```python
>>> from riscv.disasm import disasm_insn_t
>>> d = disasm_insn_t('wsplit_c0', 0x1234, 0xFFFF)
>>> d.name
'wsplit.c0'
```

Tests assert against the original `_`-form (e.g., `'wsplit_c0'`, `'warp_start_p'`,
`'warp_join'`) because they were written against the offline `_PyDisasmInsn`
NamedTuple fallback (which preserves the input string). When `_riscv.so` is built
and the real `disasm_insn_t` runs, the names come out as `'wsplit.c0'`,
`'warp.start.p'`, `'warp.join'`.

Affected tests (all in test_disasm.py):
- test_add_rf3_custom0_mm_s_formula
- test_add_warp_start_p_formula
- test_add_warp_join_formula
- test_collect_disasms_contains_p2_sample_5
- test_collect_disasms_all_8_warp_mnemonics_present
- test_collect_disasms_all_4_spr_mnemonics_present

**Resolution path (out of plan 02-06 scope):**
- Update test_disasm.py expected names to use `.` form (matches real binding
  behavior). This is a test-only fix but test_disasm.py is Wave 1 owned (plan 02-04).
- **Recommendation:** Phase 2 follow-up — normalize test expectations to dot-form,
  document the C++ behavior in the disasm.py module docstring.

**Category C — ELF fixture LOAD-segment misalignment (test_skeleton.py + direct CLI)**

Root cause: The committed `tests/gtx/data/elf/nop_wjoin.elf` LOAD program header
has VirtAddr `0x7ffff000`, NOT `0x80000000`:

```
Program Headers:
  Type           Offset             VirtAddr           PhysAddr
  LOAD           0x0000000000000000 0x000000007ffff000 0x000000007ffff000
                 0x0000000000001008 0x0000000000001008  R E    0x1000
```

Spike DRAM starts at `0x80000000`, so loading at `0x7ffff000` triggers:
```
Access exception occurred while loading payload tests/gtx/data/elf/nop_wjoin.elf:
Memory address 0x7ffff000 is invalid
```

The Makefile uses `-Ttext=0x80000000`, but GCC 15.2.0 places `.riscv.attributes`
ahead of `.text` in the LOAD segment, padding the LOAD VirtAddr back to
`0x80000000 - 0x1000 = 0x7ffff000`. Confirmed by rebuilding with
`-Wl,-Ttext-segment=0x80000000` instead, which produces a LOAD segment at
`0x80000000` (entry point shifts to `0x800000b0` since `_start` is no longer
the first thing in `.text`).

**Resolution path (out of plan 02-06 scope):**
- Fix tests/gtx/data/elf/Makefile: use `-Wl,-Ttext-segment=0x80000000`
- Rebuild and recommit nop_wjoin.elf
- This is a 1-line Makefile fix + 5KB ELF rebuild. The fix is clear but lives
  outside the plan's `files_modified` list.
- **Recommendation:** Phase 2 follow-up — fix Makefile, rebuild ELF.

**Category D — Production bug exposed once ELF loads (next regression)**

Even with a corrected ELF (verified manually with `/tmp/nop_wjoin_test3.elf` built
with `-Wl,-Ttext-segment=0x80000000`), spike traces show:

```
core   0: 0x00000000800000b0 (0x00001141) c.addi  sp, -16
core   0: 3 0x00000000800000b0 (0x1141) x2  0xfffffffffffffff0       <-- sp wraps from 0
core   0: 0x00000000800000b2 (0x0000502b) unknown
core   0: exception trap_illegal_instruction, epc 0x00000000800000b2
```

Two issues:
1. **sp not initialized:** After `addi sp, sp, -16`, sp = `0xfffffffffffffff0`,
   meaning sp was 0 at entry (NOT `0x80100000` as `reset()` should set).
   `proc.get_state().XPR.write(2, 0x80100000)` is being called but the value
   doesn't stick — possibly because spike re-resets XPR after extension `reset()`,
   or the initial sp setting needs to happen elsewhere.
2. **custom1 funct3=0b101 (WJOIN) returns illegal:** the GtxNpu extension is
   loaded (verified `find_extension('gtx')` returns the factory) but the
   instruction at `0x800000b2 0x0000502b` is not dispatched to GtxNpu.custom1.

These are real Phase-2 production bugs that the mock-fallback path could not
exercise (since mock tests don't run spike's actual dispatch loop). They live in
`src/main/python/riscv/gtx/npu.py` (or upstream `riscv.isa.ROCC` integration).

**Resolution path (out of plan 02-06 scope):**
- Investigate why `XPR.write(2, ...)` in `reset()` is overridden. May need to
  hook a different lifecycle point (e.g., `set_pc`-equivalent for sp).
- Investigate why custom1 dispatch fails. Possibly `_RISCV_AVAILABLE` registration
  path produces a wrapper class (`MyISA`) that doesn't actually expose `custom1`
  to spike's RoCC trampoline.
- **Recommendation:** Phase 2 follow-up — likely requires adding a Phase-2 deferred-items.md
  AND a follow-up plan (02-07) to fix dispatch + reset.

### Why Plan 02-06 Cannot Auto-Fix These

The plan's `files_modified` list is restrictive:
```
- .planning/phases/02-skeleton-disasm/02-06-BUILD-LOG.md
- tests/gtx/test_skeleton.py
- .planning/phases/02-skeleton-disasm/02-HUMAN-UAT.md
- .planning/phases/02-skeleton-disasm/02-VERIFICATION.md
- .planning/ROADMAP.md
```

Categories A-D require modifying:
- A: `src/main/python/riscv/gtx/npu.py` (Wave 0/1 owned — forbidden)
- B: `tests/gtx/test_disasm.py` (Wave 1 owned — forbidden)
- C: `tests/gtx/data/elf/Makefile` + `tests/gtx/data/elf/nop_wjoin.elf` (not in list)
- D: `src/main/python/riscv/gtx/npu.py` (Wave 0/1 owned — forbidden)

This is a Rule 4 — Architectural decision. Per the executor protocol, in auto
mode this is the only category that still STOPs. However, plan 02-06's `<deferred>`
section only documents F1/F2/F3 build failures, NOT post-build test regressions.
This is a planning gap — gap-closure plans should anticipate the possibility
that the build path was hiding bugs.

### Step 2.4 — Prior-phase regression sanity check

```bash
$ python3 -c "import pexpect" 2>/dev/null
ImportError: No module named 'pexpect'
```

SKIP prior-phase regression — pexpect is a known pre-existing condition
(documented in 02-05-SUMMARY.md "Issues Encountered"). Not introduced by this
plan; route to Phase-1 follow-up.

### Task 2 Summary

- Skips eliminated: 21 → 0 (gap-closure intent satisfied)
- Failures revealed: 0 → 15 (across 4 distinct root causes)
- Net pass count: 65 (was) → 71 (now)
- Net problem count: 21 deferred skips → 15 actively-failing tests
- All 15 failures are pre-existing bugs that the mock-fallback discipline hid.

---

## Task 3 — Subprocess CLI integration + trace inspection

(populated after Task 3 execution)

---

## Task 4 — UAT/VERIFICATION/ROADMAP doc-sync

(populated after Task 4 execution)
