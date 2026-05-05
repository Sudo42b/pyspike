---
phase: 3
slug: dma-ddr-i-o
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-05
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (pyspike baseline; already detected from `tests/test_extension.py` + `tests/gtx/conftest.py`) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]`; `tests/gtx/conftest.py` |
| **Quick run command** | `pytest tests/gtx/test_dma_engine.py tests/gtx/test_firmware_dma.py -x --noconftest -o "addopts="` |
| **Full suite command** | `pytest tests/gtx/ -x` |
| **Estimated runtime** | ~30 seconds (full P3 suite) |

---

## Sampling Rate

- **After every task commit:** `pytest tests/gtx/test_dma_engine.py tests/gtx/test_firmware_dma.py -x --noconftest -o "addopts="` (~5 s)
- **After every plan wave:** `pytest tests/gtx/ -x` (full P3 suite, < 30 s expected)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-XX | 01 | 1 | DMA-01 | unit | `pytest tests/gtx/test_dma_engine.py -x` | ❌ W0 | ⬜ pending |
| 03-02-XX | 02 | 2 | DMA-02 | unit | `pytest tests/gtx/test_firmware_dma.py -x` | ❌ W0 | ⬜ pending |
| 03-03-XX | 03 | 1 | DMA-04 | unit | `pytest tests/gtx/test_ddr_modes.py -x` | ❌ W0 | ⬜ pending |
| 03-04-XX | 04 | 2 | DISP-03 | unit | `pytest tests/gtx/test_dispatch_4mode.py -x` | ❌ W0 | ⬜ pending |
| 03-05-XX | 05 | 3 | DMA-03, DMA-05 | unit + integration | `pytest tests/gtx/test_deferred_store.py tests/gtx/test_dma_roundtrip.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Plan-to-task mapping above is the planner's expected decomposition; actual task IDs land at planning time.*

---

## Wave 0 Requirements

- [ ] `tests/gtx/test_dma_engine.py` — covers DMA-01 (all 6 `exec_*` helpers in `dma_engine.py`)
- [ ] `tests/gtx/test_firmware_dma.py` — covers DMA-02 (rs1/rs2/rs3 decode + funct3 LOAD/STORE/COPY branches + `is_copy` carve-out)
- [ ] `tests/gtx/test_deferred_store.py` — covers DMA-03 (queue push, flush diff, `end_p` trigger when `!wsplit_seen`, `credit_st_chk` trigger when `is_sloop && wsplit_seen`)
- [ ] `tests/gtx/test_ddr_modes.py` — covers DMA-04 (LTR + `GTX_DDR_REVERSED=1`, round-trip each mode)
- [ ] `tests/gtx/test_dma_roundtrip.py` — covers DMA-05 (full L1↔L2↔DDR chain bit-exactness)
- [ ] `tests/gtx/test_dispatch_4mode.py` — covers DISP-03 (Mode 1/2/3/4 routing parametrized)

(Existing test infrastructure — `conftest.py`, `_mocks.py`, `_RISCV_AVAILABLE` self-detect — covers framework needs. No `conftest.py` changes required.)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none — P3 is Python-only programmatic per CONTEXT D-10) | — | — | — |

*All P3 phase behaviors have automated verification. First `.elf` strict-mode regression deferred to P4 success #4 per CONTEXT D-10.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (6 new test files)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter (after planner adds task `<automated>` blocks)

**Approval:** ready

*Sign-off conditions met (Plan 05 Task 2 final step):*
- All 5 plan PLANs have `<automated>` blocks in every task `<verify>`
- All 6 Wave 0 test scaffolds exist and are populated by their owning plans
- Full P3 suite (`pytest tests/gtx/ -x --noconftest -o "addopts="`) is green
  (179 passed at sign-off)
- All 6 requirement IDs (DMA-01..05, DISP-03) closed
