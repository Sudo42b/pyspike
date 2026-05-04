---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `01-RESEARCH.md` §"Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.1 (with pytest-mypy, pytest-pylint, pytest-cov; configured in `pyproject.toml [tool.pytest.ini_options]`) |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` — `addopts = "--pylint --mypy --cov-report=lcov"` |
| **Quick run command** | `pytest tests/gtx/ -x --no-header -q -p no:pylint -p no:mypy` |
| **Full suite command** | `pytest tests/gtx/ -v` (with pylint+mypy from `addopts`) |
| **Phase gate command** | `pytest tests/ -v` (full pyspike + gtx integration; existing tests must still pass) |
| **Estimated runtime** | ~1–2 s (quick) / ~5–10 s (full) / ~30–60 s (phase gate, includes existing tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/gtx/ -x --no-header -q -p no:pylint -p no:mypy` (smoke + unit only, ~1–2 s)
- **After every plan wave:** Run `pytest tests/gtx/ -v` (full unit suite with pylint/mypy, ~5–10 s)
- **Before `/gsd:verify-work`:** `pytest tests/ -v` must be green AND wheel build verification:
  `pip wheel . -w /tmp/wheel-test/ && unzip -l /tmp/wheel-test/spike-*.whl | grep -q 'riscv/gtx/__init__.py'`
- **Max feedback latency:** 10 s (full unit suite)

---

## Per-Task Verification Map

> Task IDs are placeholders (`{plan}-{task#}`). Will be finalized once gsd-planner emits PLAN.md frontmatter and task IDs in Step 8 of plan-phase. Each row maps to a Wave 0 file that must exist before the task can be marked done.

| Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-fp | 1 | FOUND-01 (round-trip) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_all_65536_fp16_values_idempotent -x` | ❌ W0 | ⬜ pending |
| 02-fp | 1 | FOUND-01 (NaN stable bit pattern) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern -x` | ❌ W0 | ⬜ pending |
| 02-fp | 1 | FOUND-01 (subnormals) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_subnormals_roundtrip -x` | ❌ W0 | ⬜ pending |
| 02-fp | 1 | FOUND-01 (negative zero sign bit) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_negative_zero_preserved -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (LE byte write → fp16 view) | unit | `pytest tests/gtx/test_memory_layout.py::test_le_byte_order_via_byte_write -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (fp16 write → LE bytes) | unit | `pytest tests/gtx/test_memory_layout.py::test_le_byte_order_via_fp16_write -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (D-12 view invariant — `arr.base is not None`) | unit | `pytest tests/gtx/test_memory_layout.py -k view_invariant -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (slice preserves base) | unit | `pytest tests/gtx/test_memory_layout.py::test_slice_preserves_base -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (L0/L1/L2 shapes match HW params) | unit | `pytest tests/gtx/test_memory_layout.py::test_l1_shape -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (D-11 SPR unified dict) | unit | `pytest tests/gtx/test_memory_layout.py::test_spr_dict -x` | ❌ W0 | ⬜ pending |
| 03-memory | 1 | FOUND-02 (D-01 DDR lazy alloc) | unit | `pytest tests/gtx/test_memory_layout.py::test_ddr_lazy_allocation -x` | ❌ W0 | ⬜ pending |
| 01-skeleton | 1 | FOUND-03 (import path) | smoke | `python -c "from riscv.gtx import fp, memory; from riscv.gtx.params import GTX_NEST_NUM; assert GTX_NEST_NUM == 4"` | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | FOUND-03 (riscv.gtx in wheel) | integration | `pip wheel . -w /tmp/wheel-test/ && unzip -l /tmp/wheel-test/spike-*.whl \| grep -q 'riscv/gtx/__init__.py'` | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | FOUND-03 (clean cp310 venv install + import) | integration | `python -m venv /tmp/p1venv && /tmp/p1venv/bin/pip install /tmp/wheel-test/spike-*.whl && /tmp/p1venv/bin/python -c "from riscv.gtx import fp"` | ❌ W0 | ⬜ pending |
| 05-submodule | 2 | FOUND-04 (submodule registered) | manual+integration | `git submodule status \| grep -q gtx_cpp_reference` | ❌ W0 | ⬜ pending |
| 05-submodule | 2 | FOUND-04 (sdist excludes vendor) | integration | `python -m build --sdist && tar tzf dist/spike-*.tar.gz \| grep -c gtx_cpp_reference` (expect `0`) | ❌ W0 | ⬜ pending |
| 05-submodule | 2 | FOUND-04 (wheel excludes vendor) | integration | `unzip -l dist/spike-*.whl \| grep -c gtx_cpp_reference` (expect `0`) | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | PKG-02 (numpy>=2.0,<3) | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert any('numpy>=2.0' in d for d in t['project']['dependencies'])"` | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | PKG-02 (requires-python >=3.10) | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert t['project']['requires-python'] == '>=3.10'"` | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | PKG-02 (cibuildwheel cp310-cp312 only) | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); b=t['tool']['cibuildwheel']['build']; assert all('cp31' in x for x in b) and not any('cp38' in x or 'cp39' in x for x in b)"` | ❌ W0 | ⬜ pending |
| 04-packaging | 2 | PKG-02 (manylinux2014_x86_64 wheel valid) | integration | `pip wheel . -w /tmp/wheel-test/ && auditwheel show /tmp/wheel-test/spike-*.whl \| grep -q manylinux2014_x86_64` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

All Phase 1 test infrastructure is missing. Wave 0 must create:

- [ ] `tests/gtx/__init__.py` — pytest collection marker (mirror `tests/__init__.py` style)
- [ ] `tests/gtx/test_fp_roundtrip.py` — covers FOUND-01 (5 test functions)
- [ ] `tests/gtx/test_memory_layout.py` — covers FOUND-02 (LE byte order, view invariants, shapes, SPR dict, DDR lazy)
- [ ] `src/main/python/riscv/gtx/__init__.py` — re-export module (FOUND-03 import-path)
- [ ] `src/main/python/riscv/gtx/params.py` — HW constants (consumed by tests)
- [ ] `src/main/python/riscv/gtx/encoding.py` — funct7 constants (P1 stub; P2 fills full disasm)
- [ ] `src/main/python/riscv/gtx/fp.py` — FP16/FP32 helpers (D-09)
- [ ] `src/main/python/riscv/gtx/memory.py` — `GtxMemory` class (D-10/D-11/D-12)
- [ ] `src/main/python/riscv/gtx/ddr.py` — DDR lazy alloc + env handling (D-01/D-02)
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — empty package marker
- [ ] `pyproject.toml` patches — `numpy>=2.0,<3`, `requires-python = ">=3.10"`, classifiers, cibuildwheel matrix, **`packages.find.include` glob fix (`["riscv", "riscv.*"]`)**
- [ ] `MANIFEST.in` patch — `prune vendor/gtx_cpp_reference`
- [ ] Submodule registration — `git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference`
- [ ] cibuildwheel `before-all` chain — ensure `git submodule update --init --recursive` runs in manylinux container

**Framework install:** Not needed — pytest 9.0.1 already in `[project.optional-dependencies].dev`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `git submodule add` actually pins `vendor/gtx_cpp_reference/` to `https://github.com/Sudo42b/gtx_spike` | FOUND-04 | First-time submodule registration is a one-shot side effect on `.gitmodules`; the automated `git submodule status \| grep gtx_cpp_reference` only proves the entry exists, not that the URL is correct | After running `git submodule add ...`, verify: `git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url` prints `https://github.com/Sudo42b/gtx_spike(.git)?` |
| cibuildwheel build matrix produces wheels for cp310/cp311/cp312 in CI | PKG-02 | Full CI run is out-of-band from local pytest; only verifiable on GitHub Actions or by running the cibuildwheel docker pipeline locally | Optional: `pipx run cibuildwheel --platform linux` locally to dry-run; canonical verification is the next CI run on `main` after merge |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (13 files + 1 git submodule + 1 cibuildwheel hook)
- [ ] No watch-mode flags (`pytest --watch` etc.)
- [ ] Feedback latency < 10 s (full unit suite)
- [ ] `nyquist_compliant: true` set in frontmatter (after planner produces matching task IDs in PLAN.md)

**Approval:** pending
