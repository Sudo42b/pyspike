# Phase 9 Scope Decision (User Sign-Off)

Date: 2026-05-18
Decided by: lswzzang17@gmail.com

## FP8 Strategy
Selected: option-b (LUT-only) — DEFAULT
Rationale: Zero new runtime deps. Uses existing FP16_TO_FP8_LUT / FP8_TO_FP16_LUT
precomputed at import in act.py:67-117. Bit-exact via uint8 indexing on numpy and cupy.

## 28-Kernel JIT Scope
Selected: option-A (P9 numpy-only, cuda.jit deferred to P10) — DEFAULT
Rationale: Phase 9 ships narrow with numpy + cupy native vectorized ops only.
cuda.jit / guvectorize dual-impl deferred to dedicated P10 phase for v1.2 milestone.
BM-04 success criterion measures numpy path; cuda.jit out-of-scope for P9.
Estimated time vs Option-A baseline: 0 days (this IS the baseline).

## Impact on Plans
- Wave 2 act.py port follows: FP8 Option-B (LUT-only). Single deterministic code path.
- Wave 3 numba layer: deferred to P10. No pyproject changes for cuda-jit.

## Wave 0 / Wave 3 DEVICE Deferral (Option-A)

User decision (2026-05-18): defer the `DEVICE` symbol clean-cut from Wave 0
(plan 09-00-scaffold, this plan) to Wave 3 (plan 09-03-finalize).

### Rationale
- CONTEXT.md line 232 already lists `src/main/python/riscv/gtx/__init__.py`
  (lines 80, 87-88 — torch import + DEVICE re-export removal) under
  **Wave 3** ownership.
- The original PLAN.md frontmatter `must_haves` text "`from riscv.gtx import
  DEVICE` MUST raise ImportError" contradicts this Wave 3 assignment.
- Downstream files still consuming `DEVICE` directly (per CONTEXT.md
  D-05 Wave mapping):
  - `src/main/python/riscv/gtx/npu.py` (lines 12, 19, 94-106, 354) — Wave 1
  - `src/main/python/riscv/gtx/unit/memory.py` (lines 6, 16, 22, 48-56, 79-145) — Wave 1
  - `src/main/python/riscv/gtx/unit/register_file.py` (line 19) — Wave 1
  - `src/main/python/riscv/gtx/unit/context/dma_engine.py` (lines 21, 682) — Wave 2
  - `src/main/python/riscv/gtx/unit/ins/ops/vec.py` (lines 20, 67-102) — Wave 2
  - `src/main/python/riscv/gtx/unit/ins/ops/act.py` (lines 24-25, 45-181) — Wave 2
  - `src/main/python/riscv/gtx/unit/ins/ops/mm.py` (lines 28, 79) — Wave 2
  - `src/main/python/riscv/gtx/unit/ins/ops/spr.py` (line 18) — Wave 2
- Removing `DEVICE` in Wave 0 before these waves complete the port would
  break import chains and the **D-07 wave-end smoke gate** (6-op smoke +
  tile-2 baseline preserved). The smoke gate preservation per D-06 is the
  controlling constraint.

### Wave 0 Implementation (under deferral)
- `config_params.py` defines `DEVICE: str = "cpu" if xp is _np else "cuda"`
  with a multi-line deprecation comment naming Wave 3 as the owner.
- `src/main/python/riscv/gtx/__init__.py` retains the
  `from .config_params import DEVICE  # noqa: E402,F401` re-export at line ~84
  (line numbers shifted after torch try/except block removal).
- `tests/gtx/test_xp_alias.py::test_device_symbol_deprecated_alias_present`
  asserts the deprecated alias is present (string type, `"cpu"` under
  xp=numpy). The Wave 3 acceptance test will flip this to
  `pytest.raises(ImportError)`.

### Wave 3 Acceptance (deferred from Wave 0)
The following becomes a Wave 3 plan (09-03-finalize) acceptance criterion:
- `from riscv.gtx import DEVICE` raises `ImportError`
- `from riscv.gtx.config_params import DEVICE` raises `ImportError`
- Grep confirms zero `DEVICE` references in `src/main/python/riscv/gtx/`
  outside of CHANGELOG / decision-log comments.

### Impact on `must_haves`
The original Wave 0 `must_haves` line:
> "DEVICE symbol removed — `from riscv.gtx import DEVICE` raises `ImportError`
> (D-04 clean cut)."

is reinterpreted as **"deferred to Wave 3 per CONTEXT line 232; tracked in
plan 09-03-finalize acceptance criteria"** for this revision pass. All other
Wave 0 must_haves remain in scope and were satisfied.
