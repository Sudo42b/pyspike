---
phase: 4
slug: mm-subsystem
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `04-RESEARCH.md` § Validation Architecture (lines 747-801).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥ 8 (existing in `pyproject.toml`) |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` (existing); offline isolation via `--noconftest -o "addopts="` (P2 plan-05 D-1) |
| **Quick run command** | `pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py tests/gtx/test_funct7_routing.py -x --noconftest -o "addopts="` |
| **Full suite command** | `pytest tests/gtx/ -q` (includes `test_regression_fw_mm.py`; gated on `_RISCV_AVAILABLE` + `mm_basic.elf` + `pyspike` on PATH) |
| **Estimated runtime** | ~30s quick / ~1-2 min full |

---

## Sampling Rate

- **After every task commit:** Quick run command (≤30s; pure-python; no `_RISCV_AVAILABLE` requirement)
- **After every plan wave:** Full suite command (full P3 + P4 regression, ~1-2 min)
- **Before `/gsd:verify-work 4`:** Full suite green INCLUDING `test_regression_fw_mm.py::test_mm_basic_strict_mode_pass`. If `_RISCV_AVAILABLE=False` or `mm_basic.elf` missing, regression test skips cleanly (NEVER fail).
- **Max feedback latency:** ~30s per task

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-?-? | TBD | 0 | MM-01 | unit | `pytest tests/gtx/test_op_mm.py::test_gemm_core_explicit_3loop_matches_oracle -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-01 | unit | `pytest tests/gtx/test_op_mm.py::test_gemm_core_fp32_internal_not_fp16 -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_handler_registry_has_all_10_mm_variants -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_basic_bit_exact -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_s_writes_fp32_to_addrc -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_o_writes_scalar_to_l0_be -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_v_writes_dot_to_l0_le -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-02 | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_t_writes_transposed -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-03 | unit | `pytest tests/gtx/test_op_mm.py::test_decode_firmware_mm_args -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-03 | unit | `pytest tests/gtx/test_funct7_routing.py::test_funct7_zero_collision_routing -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-03 | unit | `pytest tests/gtx/test_funct7_routing.py::test_funct7_one_always_mmc -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-04 | integration | `pytest tests/gtx/test_mm_chain.py::test_mm_addrc_chain_continuity -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-04 | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_chain_continuity -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-04 | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_per_cell_isolation -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-04 | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_dtype_locked -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-05 | regression | `pytest tests/gtx/test_regression_fw_mm.py::test_mm_basic_strict_mode_pass -x` | ❌ W0 (gated) | ⬜ pending |
| 4-?-? | TBD | 0 | MM-05 | unit | `pytest tests/gtx/test_op_mm.py::test_verify_minimal_be_fp16_pairs -x` | ❌ W0 | ⬜ pending |
| 4-?-? | TBD | 0 | MM-05 (#5) | unit | `pytest tests/gtx/test_funct7_routing.py::test_mode4_routes_to_tmu_curr -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Plan IDs and Task IDs to be filled by gsd-planner during PLAN.md creation.*

---

## Wave 0 Requirements

- [ ] `tests/gtx/test_op_mm.py` — covers MM-01, MM-02, MM-03 (decode), MM-05 (verify_minimal unit)
- [ ] `tests/gtx/test_mm_chain.py` — covers MM-04
- [ ] `tests/gtx/test_funct7_routing.py` — covers MM-03 (routing matrix), MM-05 (Mode 4)
- [ ] `tests/gtx/test_regression_fw_mm.py` — covers MM-05 strict-mode .elf regression
- [ ] `tests/gtx/_verify_minimal.py` — D-13 mini port (compare_hex BE FP16 bit-pair)
- [ ] `tests/gtx/data/elf/{mm_basic.S, mm_basic.elf}` — D-09 fallback fixture (ELF committed; vendor lacks asset)
- [ ] `tests/gtx/data/elf/Makefile` — extend existing or add `mm_basic.elf` rule
- [ ] `tests/gtx/data/golden/mm_basic_n1s16.hex` — D-10 fallback synthesized golden (FP16 BE bit-pair format per `verify.py:235`)
- [ ] `src/main/python/riscv/gtx/gemm_core.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/mm_engine.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/ops/mm.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — `from . import mm` line check
- [ ] Test framework: existing pytest infra is sufficient; no new install needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cross-host BLAS drift profile (NumPy + scipy-openblas vs scalar 3-loop) | MM-01 | Single-host empirical only (4 ULP / 41 of 500 trials drift on dev box). Different BLAS backends may show different profiles. | Run validation pytest on CI manylinux2014 environment after wheel build; capture drift histogram. Document follow-up ticket if drift exceeds dev-box profile. |
| Subprocess WJOIN propagation under cibuildwheel | MM-05 | P2 verified on dev box; CI manylinux2014 may differ. Subprocess-based regression depends on stdout/stderr line buffering. | Trigger cibuildwheel locally; confirm `test_regression_fw_mm` exits 0 in cp310-manylinux2014_x86_64. v2 follow-up: in-process `sim_t.run` if subprocess proves flaky in CI. |

*All P4-locked behaviors above the manual gate have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (12 missing files listed)
- [ ] No watch-mode flags (uses `-x --noconftest -o "addopts="`)
- [ ] Feedback latency < 30s for quick suite
- [ ] `nyquist_compliant: true` set in frontmatter once Wave 0 commits land

**Approval:** pending
