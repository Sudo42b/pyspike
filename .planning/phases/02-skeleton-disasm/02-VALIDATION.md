---
phase: 2
slug: skeleton-disasm
status: planned
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-04
updated: 2026-05-04 (after /gsd:plan-phase 2)
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | `pyproject.toml` (existing pytest section) + `tests/gtx/conftest.py` (Wave 0 plan 01 task 1) |
| **Quick run command** | `pytest tests/gtx/ -x -q --noconftest -o "addopts="` |
| **Full suite command** | `pytest tests/gtx/ --cov=riscv.gtx` (after `_riscv.so` is built) |
| **Estimated runtime** | ~15-30 seconds for tests/gtx/ alone |
| **Mock-fallback** | D-17 hybrid: tests run without `_riscv.so` via `tests/gtx/_mocks.py` |

---

## Sampling Rate

- **After every task commit:** `pytest tests/gtx/ -x -q --noconftest -o "addopts="` (mock-fallback path, fast)
- **After every plan wave:** `pytest tests/gtx/ -x -q --noconftest -o "addopts="` + `pytest tests/gtx/ --cov=riscv.gtx` if `_riscv.so` is built
- **Before `/gsd:verify-work`:** Full suite must be green; integration test (`test_skeleton.py`) gated on `_riscv` + `.elf` availability
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

> One row per task across all 5 plans. Every row has an automated command.
> Total: 17 task rows.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-T1 | 02-01 | 0 | infra (D-17/D-18/D-19) | scaffold | `python -c "from tests.gtx._mocks import MockProcessor; MockProcessor().get_state().XPR.write(2, 0x80100000)"` | ✅ | ✅ done |
| 02-01-T2 | 02-01 | 0 | CORE-01, CORE-02 | unit | `python -c "from riscv.gtx.encoding import GTX_F7_WJOIN; assert GTX_F7_WJOIN==0x03"` | ✅ | ✅ done |
| 02-01-T3 | 02-01 | 0 | infra (D-22) | fixture | `ls tests/gtx/data/elf/nop_wjoin.S tests/gtx/data/elf/Makefile` | ✅ | ✅ done |
| 02-02-T1 | 02-02 | 1 | SPR-01 | unit | inline python verifies routing — see plan 02-02 task 1 verify block | ✅ | ✅ done |
| 02-02-T2 | 02-02 | 1 | SPR-02 | unit | `python -c "from riscv.gtx.ops.spr import wrspr_iss"` | ✅ | ✅ done |
| 02-02-T3 | 02-02 | 1 | SPR-01, SPR-02 | unit | `pytest tests/gtx/test_spr.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-03-T1 | 02-03 | 1 | DISP-02, CORE-03 | unit | inline python verifies _do_* helpers — see plan 02-03 task 1 verify block | ✅ | ✅ done |
| 02-03-T2 | 02-03 | 1 | DISP-02 | unit | `pytest tests/gtx/test_warp.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-03-T3 | 02-03 | 1 | CORE-03 | unit | `pytest tests/gtx/test_wjoin.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-04-T1 | 02-04 | 1 | DISASM-01 | unit | inline python verifies match/mask formulas — see plan 02-04 task 1 verify block | ✅ | ✅ done |
| 02-04-T2 | 02-04 | 1 | DISASM-01 | unit | `python -c "from riscv.gtx import _registry; from riscv.gtx.ops import spr, control; assert len(_registry.collect_disasms()) >= 18"` | ✅ | ✅ done |
| 02-04-T3 | 02-04 | 1 | DISASM-01 | unit | `pytest tests/gtx/test_disasm.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-05-T1 | 02-05 | 2 | CORE-01 | unit + skipif | `pytest tests/gtx/test_register.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-05-T2 | 02-05 | 2 | CORE-02 | unit + skipif | `pytest tests/gtx/test_reset.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-05-T3 | 02-05 | 2 | DISP-01 | unit + skipif | `pytest tests/gtx/test_dispatch.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-05-T4 | 02-05 | 2 | CORE-01, CORE-03 | integration + skipif | `pytest tests/gtx/test_skeleton.py -x -q --noconftest -o "addopts="` | ✅ | ✅ done |
| 02-05-T5 | 02-05 | 2 | infra | doc | `grep -E "nyquist_compliant: true" .planning/phases/02-skeleton-disasm/02-VALIDATION.md` | ✅ | ✅ done |

---

## Requirement → Test File Map

| Requirement | Primary test file(s) | Plan |
|-------------|----------------------|------|
| CORE-01 | `tests/gtx/test_register.py`, `tests/gtx/test_skeleton.py` | 02-05 |
| CORE-02 | `tests/gtx/test_reset.py` | 02-05 |
| CORE-03 | `tests/gtx/test_wjoin.py`, `tests/gtx/test_skeleton.py` | 02-03, 02-05 |
| CORE-04 | `tests/gtx/test_spr.py::test_xs1_zero_workaround_proof` | 02-02 |
| SPR-01 | `tests/gtx/test_spr.py` (routing tests) | 02-02 |
| SPR-02 | `tests/gtx/test_spr.py` (handler tests) | 02-02 |
| DISASM-01 | `tests/gtx/test_disasm.py` | 02-04 |
| DISP-01 | `tests/gtx/test_dispatch.py` | 02-05 |
| DISP-02 | `tests/gtx/test_warp.py` | 02-03 |

---

## Wave 0 Requirements

- [x] `tests/gtx/conftest.py` — shared fixtures (plan 01 task 1)
- [x] `tests/gtx/_mocks.py` — MockProcessor / MockState / MockXPR / MockInsn (plan 01 task 1)
- [x] `tests/conftest.py` D-18 try/except guard (plan 01 task 1)
- [x] `tests/gtx/data/elf/nop_wjoin.S` — assembly source (plan 01 task 3)
- [x] `tests/gtx/data/elf/Makefile` — reproducible build (plan 01 task 3)
- [x] `tests/gtx/data/elf/nop_wjoin.elf` — pre-built binary (plan 01 task 3, may require toolchain at execution time)
- [x] `tests/gtx/test_spr.py` — full impl (plan 02 task 3)
- [x] `tests/gtx/test_warp.py` — full impl (plan 03 task 2)
- [x] `tests/gtx/test_wjoin.py` — full impl (plan 03 task 3)
- [x] `tests/gtx/test_disasm.py` — full impl (plan 04 task 3)
- [x] `tests/gtx/test_register.py` — full impl (plan 05 task 1)
- [x] `tests/gtx/test_reset.py` — full impl (plan 05 task 2)
- [x] `tests/gtx/test_dispatch.py` — full impl (plan 05 task 3)
- [x] `tests/gtx/test_skeleton.py` — full impl (plan 05 task 4)

*Note: pytest framework already installed (Phase 1); no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf` exit code 0 | CORE-01 | Subprocess invocation outside pytest worker (avoids GIL contamination) | Run command, `echo $?` must equal 0 |
| Disasm trace contains `wjoin`, `wrspr`, `rdspr` mnemonics for sampled ELF | DISASM-01 | Spike `--log` output is text-stream; pytest captures stdout differently | Run with `--log=trace.log`, `grep -E '(wjoin|wrspr|rdspr)' trace.log` returns ≥3 matches |

*Both behaviors are now also covered by automated tests:*
- *CORE-01 exit code:* `tests/gtx/test_skeleton.py::test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero` (gated on `_riscv` + `.elf`)
- *Disasm mnemonics:* `tests/gtx/test_disasm.py::test_collect_disasms_contains_p2_sample_5` (always runs)

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (test files + ELF fixture .S+Makefile committed; .elf optional in dev)
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-04
