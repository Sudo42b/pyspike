---
phase: 2
slug: skeleton-disasm
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-04
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | `pyproject.toml` (existing pytest section) + `tests/gtx/conftest.py` (Wave 0) |
| **Quick run command** | `pytest tests/gtx/ -x -q` |
| **Full suite command** | `pytest tests/gtx/ --cov=riscv.gtx` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/gtx/ -x -q`
- **After every plan wave:** Run `pytest tests/gtx/ --cov=riscv.gtx`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

> Filled by gsd-planner. Each task in PLAN.md must list automated command OR Wave 0 dependency.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-XX | TBD  | 0    | infra       | scaffold  | `ls riscv/gtx/__init__.py` | ❌ W0 | ⬜ pending |

---

## Wave 0 Requirements

- [ ] `tests/gtx/conftest.py` — shared fixtures (ELF path resolver, GtxNpu instance factory)
- [ ] `tests/gtx/data/elf/nop_wjoin.elf` — minimal NOP+WJOIN firmware (build or copy)
- [ ] `tests/gtx/test_spr.py` — stubs for SPR-01, SPR-02
- [ ] `tests/gtx/test_warp.py` — stubs for DISP-02, loop state machine
- [ ] `tests/gtx/test_dispatch.py` — stubs for DISP-01, custom0 funct7 routing
- [ ] `tests/gtx/test_disasm.py` — stubs for DISASM-01
- [ ] `tests/gtx/test_skeleton.py` — stubs for CORE-01..04 (end-to-end NOP regression)

*Note: pytest framework already installed (Phase 1); no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf` exit code 0 | CORE-01 | Subprocess invocation outside pytest worker (avoids GIL contamination) | Run command, `echo $?` must equal 0 |
| Disasm trace contains `wjoin`, `wrspr`, `rdspr` mnemonics for sampled ELF | DISASM-01 | Spike `--log` output is text-stream; pytest captures stdout differently | Run with `--log=trace.log`, `grep -E '(wjoin\|wrspr\|rdspr)' trace.log` returns ≥3 matches |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (test files + ELF fixture)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter (after planner fills task map)

**Approval:** pending
