---
phase: 9
slug: backend-migration-numpy-cupy
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-18
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x via `uv run pytest` (system python broken by libcusparseLt) |
| **Config file** | `tests/gtx/conftest.py` (CURRENT: hard-requires CUDA at line 18 — Wave 0 must refactor) |
| **Quick run command** | `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v` |
| **Full suite command** | `uv run pytest tests/gtx/ --no-cov -v` |
| **Estimated runtime** | Quick: ~150s (6 ops × ~25s) · Full: ~10-15 min (84 op vendor sweep + units) |

---

## Sampling Rate

- **After every task commit:** Run unit test most-relevant to changed module (~5s)
- **After every plan wave:** Run quick command (6 ops + tile-2 unit test, ~3min)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 200 seconds per wave gate

---

## Per-Task Verification Map

> Populated by planner. Below is the wave-level skeleton.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| (TBD by planner) | | | | | | | |

---

## Wave 0 Requirements

- [ ] `tests/gtx/conftest.py` — refactor: remove hard `torch.cuda.is_available()` requirement (line 18). Make backend fixture xp-aware (numpy default; cupy when `GTX_USE_CUDA=1`). REQ: BM-01
- [ ] `src/main/python/riscv/gtx/config_params.py` — add `xp` alias + `to_host()` / `to_device()` helpers + GTX_USE_CUDA fail-loud handler. REQ: BM-01
- [ ] FP8 strategy decision documented + LUT-only path verified (no torch.float8_e4m3fn) — REQ: BM-03 (act engine)
- [ ] 28-kernel scope decision A/B/C user sign-off captured in Wave 0 plan summary
- [ ] `build/` artifact cleanup if present (cibuildwheel hygiene)
- [ ] `riscv/gtx/__init__.py` torch ImportError surface line removed (`import torch  # noqa: F401`)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CuPy GPU smoke test (BM-05) | BM-05 | Requires NVIDIA GPU + cupy-cuda12x installed | On GPU machine: `GTX_USE_CUDA=1 uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v` — must print `xp.__name__ == 'cupy'` + byte-exact PASS |
| Wheel size delta (BM-06) | BM-06 | Requires `uv build` + `auditwheel` comparison vs pre-migration baseline | `du -h dist/spike-*.whl` before/after; expected ≤ 0 MB delta (PyTorch removal reduces wheel) |
| CLAUDE.md "Dependencies" section update | BM-06 | Doc edit — human review needed for consistency | Diff CLAUDE.md "Dependencies" sub-section; ensure NumPy default + CuPy opt-in story replaces torch |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (conftest CUDA gate + FP8 strategy + xp helpers)
- [ ] No watch-mode flags
- [ ] Feedback latency < 200s per wave gate
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
