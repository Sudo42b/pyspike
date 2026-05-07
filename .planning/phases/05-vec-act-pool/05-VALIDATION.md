---
phase: 5
slug: vec-act-pool
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-07
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Detailed Validation Architecture (test framework, per-req → test mapping,
> sampling rate, Wave 0 gaps) is sourced verbatim from
> `.planning/phases/05-vec-act-pool/05-RESEARCH.md` §Validation Architecture
> (lines 898–977). The planner will lift those rows into a per-task map
> below as it builds PLAN.md files.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥ 8 (existing in `pyproject.toml`) |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]`; offline isolation via `--noconftest -o "addopts="` (P2 plan-05 D-1) |
| **Quick run command** | `pytest tests/gtx/test_op_vec.py tests/gtx/test_op_act.py tests/gtx/test_op_format.py tests/gtx/test_pooling.py tests/gtx/test_vsum_precision.py -x --noconftest -o "addopts="` |
| **Full suite command** | `pytest tests/gtx/ -q` |
| **Estimated runtime** | ~45s quick / ~2–3 min full (incl. 64K FP16→FP8 LUT build + 29 oracle parametrize) |

---

## Sampling Rate

- **After every task commit:** quick run command (≤45 s; pure-Python; no `_RISCV_AVAILABLE` requirement)
- **After every plan wave:** full suite command (P3 + P4 + P5 regression)
- **Before `/gsd:verify-work`:** Full suite must be green INCLUDING `test_regression_fw_act.py::test_act_strict_mode_pass` (skips cleanly when `_RISCV_AVAILABLE=False` or `.elf` missing)
- **Max feedback latency:** 45 s (quick) / 180 s (full)

---

## Per-Task Verification Map

> Populated by the planner: each `<task>` block in a PLAN.md must reference one
> Req-ID row from RESEARCH.md §Validation Architecture (`Phase Requirements →
> Test Map`, lines 909–950) and copy its `Automated Command` into the task's
> `<automated>` verify field.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| _planner-fills_ | _NN_ | _W_ | REQ-_XX_ | _type_ | `_command_` | ⬜ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Sourced from RESEARCH.md §Validation Architecture › Wave 0 Gaps (lines 956–977):

- [ ] `tests/gtx/test_op_vec.py` — covers VEC-01..05
- [ ] `tests/gtx/test_op_act.py` — covers ACT-01, ACT-02, ACT-05
- [ ] `tests/gtx/test_op_format.py` — covers ACT-04 (7 cvt directions + scale/offset)
- [ ] `tests/gtx/test_pooling.py` — covers ACT-03
- [ ] `tests/gtx/test_vsum_precision.py` — covers VEC-02 dual-mode (D-09/D-10)
- [ ] `tests/gtx/test_oracle_parity.py` — covers VRF-02 (20 oracles parametrized)
- [ ] `tests/gtx/test_regression_fw_act.py` — covers `.elf` strict-mode regression (gated on `_RISCV_AVAILABLE`)
- [ ] `tests/gtx/_oracles.py` — VRF-02 helpers (29 functions, skip GELU_ERF per CLAUDE.md scipy ban)
- [ ] `tests/gtx/conftest.py` — add `proc_with_addra_addrr_seeded` fixture
- [ ] `tests/gtx/data/elf/{activation_relu_gelu.S, activation_relu_gelu.elf}` — D-04 fixture
- [ ] `tests/gtx/data/elf/Makefile` — extend with `activation_relu_gelu.elf` rule
- [ ] `tests/gtx/data/golden/activation_relu_gelu.hex` — synthesized golden
- [ ] `src/main/python/riscv/gtx/{vec_core,vec_engine,act_core,act_engine}.py` — module-exists checks (fail until Wave 1b)
- [ ] `src/main/python/riscv/gtx/ops/{vec,act}.py` — module-exists checks
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — `from . import vec` + `from . import act` lines

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| _none expected — `.elf` strict-mode regression covers end-to-end_ | — | — | — |

*All phase behaviors target automated verification. The `.elf` regression is automated but skip-gated on `_RISCV_AVAILABLE` (P4 04-01 pattern).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45 s (quick) / 180 s (full)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
