---
phase: 09-backend-migration-numpy-cupy
plan: 00
type: execute
wave: 1
# CONTEXT D-05 Wave 0 = this plan (wave: 1). Subsequent waves (Wave 1/2/3 in D-05)
# correspond to wave: 2/3+ here. See README at top of each plan.
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/config_params.py
  - src/main/python/riscv/gtx/__init__.py
  - tests/gtx/conftest.py
  - tests/gtx/test_xp_alias.py
  - .planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md
  - .planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt
autonomous: false
requirements:
  - BM-01
user_setup: []

must_haves:
  truths:
    - "`from riscv.gtx.config_params import xp` works in clean cp310 venv with no torch installed (xp resolves to numpy module)."
    - "`GTX_USE_CUDA=1` with cupy missing raises `RuntimeError` containing `pip install 'spike[cuda]'` recovery hint (D-03 fail-loud)."
    - "`tests/gtx/conftest.py` no longer hard-requires CUDA; collection succeeds on no-GPU box."
    - "`from riscv.gtx.config_params import to_host, to_device` returns no-op identity functions when xp=numpy."
    - "`DEVICE` symbol removed — `from riscv.gtx import DEVICE` raises `ImportError` (D-04 clean cut)."
    - "FP8 strategy DEFAULT-locked to LUT-only path (Option-B); no `ml_dtypes` dep, no `torch.float8_e4m3fn`. Selecting A/C requires separate `--revise` pass."
    - "28-kernel scope DEFAULT-locked to Option-A (P9 numpy-only, cuda.jit deferred to P10). Selecting B/C requires separate `--revise` pass."
    - "User sign-off captured in 09-SCOPE-DECISION.md (confirming defaults OR signaling revision pass needed)."
    - "Pre-migration wheel size baseline pinned at `.planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt` for BM-06 delta measurement (M-1)."
    - "Wave-end gate: ABS + GELU + RELU + SIGMOID + TANH + SOFTMAX + tile-2 unit test all GREEN (baseline preserved — no source ports yet, just scaffold + DEVICE removal traceability)."
  artifacts:
    - path: "src/main/python/riscv/gtx/config_params.py"
      provides: "xp alias resolved at import-time + to_host/to_device helpers; DEVICE removed"
      contains: "xp, to_host, to_device = _resolve_backend()"
    - path: "tests/gtx/conftest.py"
      provides: "GTX_USE_CUDA-gated CUDA check (was torch.cuda.is_available() hard-require)"
      contains: "GTX_USE_CUDA"
    - path: "tests/gtx/test_xp_alias.py"
      provides: "Unit tests for BM-01: default numpy, fail-loud cupy-missing, identity helpers"
      contains: "def test_xp_default_is_numpy"
    - path: ".planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md"
      provides: "User-signed scope confirmation (DEFAULT Option-A scope + Option-B FP8, OR revision-pass signal)"
      contains: "## FP8 Strategy\n## 28-Kernel Scope Decision"
    - path: ".planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt"
      provides: "Pre-migration wheel size baseline for BM-06 delta calculation"
      contains: "dist/spike-"
  key_links:
    - from: "src/main/python/riscv/gtx/config_params.py"
      to: "src/main/python/riscv/gtx/__init__.py"
      via: "removal of DEVICE re-export at line 88; removal of `import torch  # noqa: F401` at line 80"
      pattern: "from .config_params import DEVICE"
    - from: "tests/gtx/conftest.py"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "GTX_USE_CUDA env-var gate (no torch.cuda.is_available)"
      pattern: "torch.cuda.is_available"
---

<objective>
Wave 0 scaffold for Phase 9 backend migration (= CONTEXT D-05 "Wave 0"; this PLAN file has `wave: 1` for execute-phase wave ordering). Establish the `xp` alias single-source-of-truth in `config_params.py`, port the test infrastructure off `torch.cuda.is_available()`, **confirm DEFAULT** FP8 strategy = LUT-only AND **DEFAULT** 28-kernel scope = Option-A (P10-split), pin a pre-migration wheel size baseline, and remove the `import torch` ImportError surface from package init. No source ports yet — that is Wave 1+. Wave-end gate verifies the 6-op smoke + tile-2 baseline still passes after these scaffolding changes (dual-import allowed per D-06 between waves).

Purpose: All subsequent waves import `xp` and call `to_host()`/`to_device()` — those primitives MUST exist before Wave 1 starts. Without the conftest port, no-GPU CI/dev boxes break test collection (RESEARCH critical finding #2). The FP8 + scope defaults unblock Wave 2 unconditionally; if the user wants alternative options, they must request a separate revision pass before Wave 2 begins.

Output: `config_params.py` carries the new xp/helper exports; `__init__.py` is purged of torch ImportError surface; `tests/gtx/conftest.py` is xp-aware; `tests/gtx/test_xp_alias.py` exists with 3 RED→GREEN tests for BM-01; `09-SCOPE-DECISION.md` confirms defaults (or signals revision-pass); `09-pre-wheel-size.txt` records baseline for BM-06.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md
@.planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md
@.planning/phases/09-backend-migration-numpy-cupy/09-VALIDATION.md
@src/main/python/riscv/gtx/config_params.py
@src/main/python/riscv/gtx/__init__.py
@tests/gtx/conftest.py
@CLAUDE.md

<interfaces>
<!-- Key contracts that Wave 0 establishes for Waves 1-3 to consume. -->
<!-- Wave 1+ executors import these directly — no exploration needed. -->

From src/main/python/riscv/gtx/config_params.py (after Wave 0):
```python
# Module-level eager resolution (D-01, D-02).
xp           # numpy module (default) OR cupy module (GTX_USE_CUDA=1).
to_host      # callable: identity if xp=numpy, cp.asnumpy if xp=cupy (D-12).
to_device    # callable: identity if xp=numpy, cp.asarray if xp=cupy (D-12).

# All existing constants (GTX_NEST_NUM, DEFAULT_DDR_SIZE, GTX_L0_SIZE_BYTES, ...) preserved.
# DEVICE removed (D-04 clean cut).
```

From tests/gtx/conftest.py (after Wave 0):
```python
# Old hard-require (line 18 area):
#   if not torch.cuda.is_available(): pytest.exit(...)
# New GTX_USE_CUDA-gated optional cupy check.
# No `import torch` anywhere in conftest.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add xp alias + to_host/to_device helpers to config_params.py; remove DEVICE symbol</name>
  <files>src/main/python/riscv/gtx/config_params.py</files>
  <read_first>
    - src/main/python/riscv/gtx/config_params.py (full file — lines 1-57 currently torch-based; preserve every constant from line 28 onward unchanged)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-01..D-04, D-12)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Code Example 1, lines ~565-601 — verified xp resolution pattern; Pitfall 3 — 260518-ffr fail-loud precedent)
  </read_first>
  <behavior>
    - Test 1 `test_xp_default_is_numpy`: import config_params with `GTX_USE_CUDA` unset → `xp is numpy`.
    - Test 2 `test_to_host_to_device_identity_when_numpy`: with xp=numpy, both helpers return input unchanged (`to_host(arr) is arr`).
    - Test 3 `test_gtx_use_cuda_without_cupy_fails_loud`: with `GTX_USE_CUDA=1` env-var + cupy absent, importing config_params raises `RuntimeError` whose message contains `"pip install 'spike[cuda]'"` (case-sensitive substring check).
    - Test 4 `test_device_symbol_removed`: `from riscv.gtx.config_params import DEVICE` raises `ImportError` (D-04 clean cut).
    - Module-level constants (`GTX_NEST_NUM`, `DEFAULT_DDR_SIZE`, `GTX_L0_SIZE_BYTES`, etc.) remain importable with identical values.
  </behavior>
  <action>
    Replace lines 1-25 of `src/main/python/riscv/gtx/config_params.py` with the following block. Leave lines 26-57 (NEST/SPU/DDR/memory size constants) UNCHANGED:

    ```python
    from __future__ import annotations
    import os
    import numpy as _np


    def _identity(arr):
        """No-op for xp=numpy path (D-12)."""
        return arr


    def _resolve_backend():
        """Resolve xp + to_host + to_device at import time (D-01, D-02).

        Default: numpy + identity helpers (no-op).
        GTX_USE_CUDA=1 (or "true"/"TRUE"): require cupy importable, else fail-loud
        with `pip install 'spike[cuda]'` hint (D-03). Silent fallback FORBIDDEN
        (260518-ffr regression precedent — torch.cuda.is_available auto-true
        flipped 5x ABS slowdown).
        """
        env = os.environ.get("GTX_USE_CUDA", "").strip()
        if env not in ("1", "true", "TRUE"):
            return _np, _identity, _identity

        try:
            import cupy as _cp
        except ImportError as exc:
            raise RuntimeError(
                "GTX_USE_CUDA=1 set but cupy is not importable. "
                "Install with: pip install 'spike[cuda]'"
            ) from exc

        return _cp, _cp.asnumpy, _cp.asarray


    # Module-level eager resolution, frozen for process lifetime (D-02).
    # All gtx.* modules: `from .config_params import xp, to_host, to_device`.
    xp, to_host, to_device = _resolve_backend()

    # NOTE: `DEVICE` symbol removed per D-04 (clean cut — no backwards-compat shim).
    # External callers importing `DEVICE` will get ImportError. Update call sites
    # to use xp directly (numpy/cupy have no torch.device equivalent — device is
    # implicit in the xp module reference).
    ```

    Then verify lines 28-57 (GTX_NEST_NUM through GTX_DDR_BASE through SPR address note) are byte-identical to the pre-edit state. Do NOT touch them.

    Implementation notes (per CLAUDE.md surgical-changes rule):
    - Use `_np` private alias to keep `numpy` from leaking as a public symbol via `from .config_params import *`.
    - The `_identity` function takes one positional arg (no kwargs) — matches `cp.asnumpy`/`cp.asarray` signatures.
    - Do NOT add docstring to module top (existing file has none — match style).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_xp_alias.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "^import torch" src/main/python/riscv/gtx/config_params.py` returns 0 lines.
    - `grep -n "^DEVICE" src/main/python/riscv/gtx/config_params.py` returns 0 lines.
    - `grep -cE "^(xp|to_host|to_device) = " src/main/python/riscv/gtx/config_params.py` returns 0 — only the tuple-unpacking line `xp, to_host, to_device = _resolve_backend()` exists.
    - `grep -c "_resolve_backend" src/main/python/riscv/gtx/config_params.py` returns 2 (def + call).
    - `grep -c "pip install 'spike\[cuda\]'" src/main/python/riscv/gtx/config_params.py` returns 1.
    - `uv run python -c "from riscv.gtx.config_params import xp, to_host, to_device; assert xp.__name__ == 'numpy'; assert to_host(42) == 42"` exits 0.
    - `uv run python -c "from riscv.gtx.config_params import DEVICE" 2>&1 | grep -q "ImportError"` exits 0.
    - `uv run python -c "from riscv.gtx.config_params import GTX_NEST_NUM, DEFAULT_DDR_SIZE; assert GTX_NEST_NUM == 4; assert DEFAULT_DDR_SIZE == 4*1024**3"` exits 0.
  </acceptance_criteria>
  <done>config_params.py exports `xp`, `to_host`, `to_device`; DEVICE removed; existing constants preserved; all 4 BM-01 tests in test_xp_alias.py pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Create tests/gtx/test_xp_alias.py with BM-01 RED→GREEN coverage</name>
  <files>tests/gtx/test_xp_alias.py</files>
  <read_first>
    - tests/gtx/conftest.py (current state — note the CUDA-required block at lines 5, 15, 18 that will be ported in Task 3; tests in this file must NOT depend on cupy)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Validation table lines ~778-790 — BM-01 automated commands)
    - src/main/python/riscv/gtx/config_params.py (post-Task-1 state)
  </read_first>
  <behavior>
    - Test 1 `test_xp_default_is_numpy`: subprocess `env -u GTX_USE_CUDA uv run python -c "from riscv.gtx.config_params import xp; print(xp.__name__)"` returns `numpy`.
    - Test 2 `test_to_host_to_device_identity_when_numpy`: `to_host(np.array([1,2,3]))` returns the same ndarray (`arr is to_host(arr)` is True under xp=numpy).
    - Test 3 `test_gtx_use_cuda_without_cupy_fails_loud`: subprocess with `GTX_USE_CUDA=1` env and (presumed) no cupy installed → exit code != 0 AND stderr contains `pip install 'spike[cuda]'`. Skip with reason if cupy IS importable in the test env.
    - Test 4 `test_device_symbol_removed`: `pytest.raises(ImportError)` on `from riscv.gtx.config_params import DEVICE`.
  </behavior>
  <action>
    Create new file `tests/gtx/test_xp_alias.py` with the following content (no top-of-file docstring — match other tests in the directory):

    ```python
    """BM-01 unit tests: xp alias + helpers + fail-loud + DEVICE removal."""
    from __future__ import annotations
    import os
    import subprocess
    import sys

    import numpy as np
    import pytest


    def _spawn_python(env_overrides: dict, code: str) -> subprocess.CompletedProcess:
        env = {**os.environ, **env_overrides}
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )


    def test_xp_default_is_numpy():
        """D-01/D-02: xp resolves to numpy when GTX_USE_CUDA unset."""
        env = {k: v for k, v in os.environ.items() if k != "GTX_USE_CUDA"}
        proc = subprocess.run(
            [sys.executable, "-c",
             "from riscv.gtx.config_params import xp; print(xp.__name__)"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}"
        assert proc.stdout.strip() == "numpy"


    def test_to_host_to_device_identity_when_numpy():
        """D-12: helpers are no-ops under xp=numpy (literal identity)."""
        from riscv.gtx.config_params import to_host, to_device, xp
        assert xp.__name__ == "numpy", "test environment must be numpy"
        arr = np.array([1, 2, 3], dtype=np.int32)
        assert to_host(arr) is arr
        assert to_device(arr) is arr


    def test_gtx_use_cuda_without_cupy_fails_loud():
        """D-03: silent fallback forbidden; RuntimeError with pip-install hint."""
        try:
            import cupy  # noqa: F401
            pytest.skip("cupy IS installed; cannot test fail-loud path")
        except ImportError:
            pass
        proc = _spawn_python(
            {"GTX_USE_CUDA": "1"},
            "from riscv.gtx.config_params import xp",
        )
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "RuntimeError" in combined
        assert "pip install 'spike[cuda]'" in combined


    def test_device_symbol_removed():
        """D-04: DEVICE removed (clean cut, no backwards-compat shim)."""
        with pytest.raises(ImportError):
            from riscv.gtx.config_params import DEVICE  # noqa: F401
    ```

    No torch references. No conftest fixtures used (pure stdlib subprocess for env isolation).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_xp_alias.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - File `tests/gtx/test_xp_alias.py` exists.
    - `grep -c "^def test_" tests/gtx/test_xp_alias.py` returns 4.
    - `grep -c "import torch" tests/gtx/test_xp_alias.py` returns 0.
    - `uv run pytest tests/gtx/test_xp_alias.py -x --no-cov -v` exits 0 with 4 passed (or 3 passed + 1 skipped if cupy is installed).
    - `grep -c "pip install 'spike\[cuda\]'" tests/gtx/test_xp_alias.py` returns 1.
  </acceptance_criteria>
  <done>4 BM-01 tests exist and all pass (or 3 pass + 1 skip when cupy present). File is torch-free.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Port tests/gtx/conftest.py off torch.cuda.is_available; remove `import torch` ImportError surface from gtx/__init__.py</name>
  <files>tests/gtx/conftest.py, src/main/python/riscv/gtx/__init__.py</files>
  <read_first>
    - tests/gtx/conftest.py (full file — lines 5/15/18 are the torch-CUDA block to remove)
    - src/main/python/riscv/gtx/__init__.py (lines 54-88 — torch import block + DEVICE re-export at line 88)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pitfall 3 — atexit ordering + 260518-ffr; Pitfall 8 — silent ImportError swallow at __init__.py:54-68)
    - Memory project_gtx_extension_silent_import_failure.md (referenced in CONTEXT canonical_refs) — silent swallow audit
  </read_first>
  <behavior>
    - Test A (manual repro): `uv run pytest --collect-only tests/gtx/ 2>&1 | grep -c "no tests ran"` returns 0 in a no-cupy environment (currently breaks if torch is missing).
    - Test B: `grep -rn "import torch" tests/gtx/conftest.py` returns 0 after edit.
    - Test C: `grep -n "from .config_params import DEVICE" src/main/python/riscv/gtx/__init__.py` returns 0 after edit.
    - Test D: `grep -n "^    import torch" src/main/python/riscv/gtx/__init__.py` returns 0 after edit (line 80 surface removed).
    - Test E: Wave-end gate (Task 5) verifies the 6-op smoke set + tile-2 still GREEN with the conftest port (no regression introduced).
  </behavior>
  <action>
    **Edit A — `tests/gtx/conftest.py`:**

    Find and remove the entire CUDA-required block. Current shape (lines ~5-21 approximately):
    ```python
    """...
    Collection fails fast with pytest.exit(returncode=1) if torch.cuda is
    ...
    """
    import torch
    ...
    if not torch.cuda.is_available():
        pytest.exit(..., returncode=1)
    ```

    Replace with this GTX_USE_CUDA-gated optional check (single block at top of file, after the module docstring):
    ```python
    import os
    import pytest

    # GTX_USE_CUDA gate (was torch.cuda.is_available() — D-01/D-04 removed torch).
    # Default path = numpy. Only assert cupy presence when user explicitly opts in.
    if os.environ.get("GTX_USE_CUDA", "").strip() in ("1", "true", "TRUE"):
        try:
            import cupy  # noqa: F401
        except ImportError:
            pytest.exit(
                "GTX_USE_CUDA=1 set but cupy is not installed. "
                "Install with: pip install 'spike[cuda]'",
                returncode=1,
            )
    ```

    Also update the module docstring (line 5 area): change `"Collection fails fast with pytest.exit(returncode=1) if torch.cuda is..."` to `"Collection fails fast only if GTX_USE_CUDA=1 set without cupy; numpy default needs no GPU."`. Update line 50 area comment `"torch tensor allocation in riscv.gtx package init"` to `"xp.zeros allocation in riscv.gtx package init"`.

    Do NOT remove any fixtures unrelated to the CUDA gate. Preserve all `@pytest.fixture` definitions verbatim.

    **Edit B — `src/main/python/riscv/gtx/__init__.py`:**

    1. Remove line 80 (`    import torch  # noqa: F401 -- imported for early ImportError surface`) and any accompanying try/except wrapper specific to it. The remaining `try: from . import npu` block at lines 54-68 stays (it handles npu/riscv.gtx own import errors per Pitfall 8 — leave the silent swallow audit to a Wave 3 task).
    2. Remove line 88 (`from .config_params import DEVICE` or `DEVICE` listed in `__all__`/re-exports). If DEVICE appears in any `__all__` tuple/list, remove it from that collection.
    3. Do NOT touch the LE-byte-order assertion at line 37 (FOUND-01 invariant, RESEARCH Pitfall 6 indirect).
    4. Do NOT touch the silent ImportError swallow at lines 54-68 — that is a Wave 3 audit task.

    Surgical edits only. Each removed line should leave NO orphan comment.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_xp_alias.py --collect-only --no-cov 2>&1 | grep -E "(ERRORS|error)" | wc -l</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch" tests/gtx/conftest.py` returns 0.
    - `grep -c "torch.cuda.is_available" tests/gtx/conftest.py` returns 0.
    - `grep -c "GTX_USE_CUDA" tests/gtx/conftest.py` returns at least 1 (the new env-var gate).
    - `grep -nE "^    import torch" src/main/python/riscv/gtx/__init__.py` returns 0 lines.
    - `grep -c "from .config_params import DEVICE" src/main/python/riscv/gtx/__init__.py` returns 0.
    - `uv run python -c "import riscv.gtx"` exits 0 (no ImportError from __init__.py with torch absent).
    - `uv run pytest --collect-only tests/gtx/ --no-cov 2>&1 | tail -5 | grep -E "tests collected|tests in"` returns at least one line (collection works on no-GPU box).
    - Last line of test output shows zero collection errors related to torch.
  </acceptance_criteria>
  <done>conftest.py and __init__.py no longer import torch for CUDA detection or early ImportError surface; test collection works without torch installed.</done>
</task>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 4: USER CHECKPOINT — Confirm FP8 LUT-only + Scope Option-A defaults (or signal need for revision pass)</name>
  <decision>Confirm DEFAULT choices (FP8 = Option-B LUT-only; Scope = Option-A P10-split) OR signal need for separate `--revise` pass for alternatives.</decision>
  <context>
    **Why this checkpoint:** Phase 9 must commit to two scope choices before Wave 2. To keep this plan small and contained:
    - **DEFAULT FP8 strategy** is **Option-B (LUT-only)** — uses existing precomputed `FP16_TO_FP8_LUT` (uint8[65536]) + `FP8_TO_FP16_LUT` (float16[256]) at `act.py:67-117`. Zero new deps. No pyproject.toml changes.
    - **DEFAULT 28-kernel scope** is **Option-A** — P9 = numpy-only (numpy + cupy native vectorized ops). cuda.jit kernels deferred to a future **P10 phase** (not created in this run; orchestrator surfaces in v1.2 milestone planning). BM-04 success criterion does NOT include CUDA path.

    Both defaults preserve CLAUDE.md "No new runtime deps" and keep this revision pass surgical. Selecting alternatives (FP8 Option-A `ml_dtypes` dep, FP8 Option-C `NotImplementedError` descope, Scope Option-B all-in-P9, Scope Option-C hot-path-only) requires edits to other plans (09-02a/09-02b/09-03) — that work is out-of-band and requires the user to re-run `/gsd:plan-phase 9 --revise 02a 02b 03` after this plan resolves.

    **B-2 simplification (per checker)**: Task 4's scope is now NARROW. It only confirms defaults OR signals "need revision pass". No alternate-option code paths are authored here.

    **D-13 P10 implication (H-6)**: Default Option-A means cuda.jit kernels are deferred to a future **Phase 10** phase. The orchestrator must surface P10 in v1.2 milestone planning. BM-04 success criterion in this run measures the numpy path only.
  </context>
  <options>
    <option id="confirm-defaults">
      <name>Confirm Defaults (RECOMMENDED — single-plan-completable)</name>
      <pros>Phase 9 ships narrow + fast. No new deps. No edits to other plans needed. P10 split is the standing decision for cuda.jit (deferred to v1.2 milestone planning). FP8 LUT path uses existing precomputed tables.</pros>
      <cons>cuda.jit acceleration of 28 kernels deferred to P10. ml_dtypes FP8 native support not used.</cons>
    </option>
    <option id="need-revision-pass">
      <name>Need Revision Pass (FP8 = A or C, OR Scope = B or C)</name>
      <pros>User retains flexibility to pick alternatives.</pros>
      <cons>Requires user to run `/gsd:plan-phase 9 --revise 02a 02b 03` separately. Phase 9 entry delayed by one revision iteration. Out-of-band scope edits to downstream plans.</cons>
    </option>
  </options>
  <resume-signal>
    Reply with ONE of the following:

    **OPTION 1 — Confirm defaults (FP8 LUT-only + Scope Option-A P10-split):**
    Reply `"confirm defaults"` (or `"approved"`). Executor authors `09-SCOPE-DECISION.md` with format below:

    ```markdown
    # Phase 9 Scope Decision (User Sign-Off)

    Date: <YYYY-MM-DD>
    Decided by: <user>

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
    ```

    Verify checkpoint exits when file exists with `Selected: option-b` AND `Selected: option-A` lines present.

    **OPTION 2 — Need revision pass:**
    Reply `"revise scope: <details>"` (e.g., `"revise scope: FP8 option-A ml_dtypes; Scope option-B all-in"`). Executor authors `09-SCOPE-DECISION.md` capturing the request:
    ```markdown
    ## Revision Pass Needed
    User requested non-default options. Halt Wave 2 entry until separate revision
    pass is executed via: /gsd:plan-phase 9 --revise 00 02a 02b 03
    ```
    Then the executor must NOT proceed with Wave 2. User runs the revision pass to edit downstream plans.
  </resume-signal>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md</files>
  <action>
    Pause for user decision. After user replies, the EXECUTOR (NOT the user) authors `.planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md` per the format in `<resume-signal>` based on user's reply text:
    - If reply contains "confirm" or "approved" → author Option 1 markdown.
    - If reply starts with "revise scope:" → author Option 2 markdown + halt Wave 2.

    Single owner: the executor authors the file based on the user's selection signal. No dual ownership.
  </action>
  <verify>
    <automated>test -f .planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md && grep -cE "Selected: option-(b|A)|Revision Pass Needed" .planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md</automated>
  </verify>
  <acceptance_criteria>
    - `.planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md` exists.
    - File contains EITHER `Selected: option-b` AND `Selected: option-A` (default-confirm path), OR `Revision Pass Needed` (revision-request path).
    - File contains `Date:` field with current date.
    - If default-confirm path: file does NOT contain `ml_dtypes` or `option-c` or `option-B` or `option-C` selections (only Option-B FP8 + Option-A Scope).
  </acceptance_criteria>
  <done>09-SCOPE-DECISION.md exists with user confirmation. Either defaults locked (Wave 2 unblocks) or revision-pass signaled (Wave 2 halts until revise run).</done>
</task>

<task type="auto">
  <name>Task 5: Wave-end gate — pin pre-migration wheel size baseline + 6-op smoke + tile-2 unit test + record walltime</name>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-00-WAVE-GATE.md, .planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt</files>
  <read_first>
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-07 gate + D-08 perf budget 85-105s)
    - tests/gtx/test_regression_fw_full_sweep.py (smoke set test name + invocation)
    - tests/gtx/test_multi_tile_dma.py (tile-2 unit test entry)
  </read_first>
  <action>
    Run the wave-end gate commands and capture results to `09-00-WAVE-GATE.md`. Also pin the pre-migration wheel size baseline for BM-06 (M-1 checker fix).

    1. **Pre-migration wheel size baseline (M-1)**:
       ```bash
       uv build --wheel
       du -h dist/spike-*.whl > .planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt
       ```
       This file is the baseline against which BM-06 (Plan 09-03 Task 7) measures the post-migration delta.

    2. Smoke set:
       ```bash
       uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v
       ```

    3. Tile-2 unit test (P8 MTDMA-03):
       ```bash
       uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v
       ```

    4. ABS strict walltime measurement:
       ```bash
       /usr/bin/time -f "%e" uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov 2>&1 | tail -3
       ```

    Author `.planning/phases/09-backend-migration-numpy-cupy/09-00-WAVE-GATE.md` containing:

    ```markdown
    # Wave 0 Gate Results

    Date: <YYYY-MM-DD>
    Commit: <sha after Task 3>

    ## Pre-Migration Wheel Size Baseline (BM-06 baseline)
    Command: `uv build --wheel && du -h dist/spike-*.whl > 09-pre-wheel-size.txt`
    Baseline: <X MB or human-readable>
    Stored at: `.planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt`

    ## Smoke Set (D-07, 6 ops)
    Command: `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v`
    Result: <PASS | FAIL>
    Stats: M passed / N failed / K skipped
    Output: <captured stdout last 20 lines>

    ## Tile-2 Unit Test (P8 MTDMA-03)
    Command: `uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v`
    Result: <PASS | FAIL>

    ## ABS Strict Walltime (D-08 budget: 85-105s)
    Wall: <X.XXs>
    In-budget: <YES | NO>
    Baseline (commit 2b0c66e): 94.82s

    ## Wave 0 Sign-Off
    - [x] config_params.py xp/to_host/to_device exported
    - [x] DEVICE symbol removed
    - [x] tests/gtx/conftest.py CUDA gate refactored to GTX_USE_CUDA
    - [x] __init__.py torch ImportError surface line removed
    - [x] FP8 strategy + 28-kernel scope DEFAULTS confirmed (09-SCOPE-DECISION.md)
    - [x] Pre-migration wheel size baseline pinned (09-pre-wheel-size.txt)
    - [x] Wave-end gate GREEN: 6-op smoke + tile-2 + perf in 85-105s window
    ```

    If any gate fails, **DO NOT proceed to Wave 1**. Record failure mode in 09-00-WAVE-GATE.md `## Failures` section and signal back to planner for revision.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v && uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `.planning/phases/09-backend-migration-numpy-cupy/09-00-WAVE-GATE.md` exists.
    - `.planning/phases/09-backend-migration-numpy-cupy/09-pre-wheel-size.txt` exists and contains at least one line referencing `dist/spike-` (M-1).
    - File contains lines matching `Result: PASS` for both smoke and tile-2 sections.
    - File contains `In-budget: YES` for the ABS walltime section (or documented justification if marginal).
    - `grep -c "Wall: " 09-00-WAVE-GATE.md` returns 1 (the walltime measurement was recorded).
    - `uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v` exits 0.
    - `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov` exits 0.
  </acceptance_criteria>
  <done>Wave 0 gate document exists with PASS results for smoke + tile-2 + perf in 85-105s window. Pre-migration wheel baseline pinned. Wave 1 unblocked.</done>
</task>

</tasks>

<verification>
- Wave 0 success = all 5 tasks complete; 09-SCOPE-DECISION.md signed; 09-pre-wheel-size.txt pinned; 09-00-WAVE-GATE.md GREEN.
- Sanity grep: `grep -rn "DEVICE" src/main/python/riscv/gtx/ | grep -vE "^.*#" | wc -l` should show only legitimate post-Wave-0 references (no torch-DEVICE).
- Wave 1 entry condition: smoke set + tile-2 + ABS walltime gate all GREEN + defaults confirmed (not revision-pass requested).
</verification>

<success_criteria>
1. `from riscv.gtx.config_params import xp, to_host, to_device` works in a clean numpy-only venv.
2. `from riscv.gtx import DEVICE` raises ImportError.
3. `GTX_USE_CUDA=1` without cupy → RuntimeError with `pip install 'spike[cuda]'` hint.
4. `uv run pytest tests/gtx/test_xp_alias.py` passes 4 (or 3 passed + 1 skipped if cupy installed).
5. `uv run pytest --collect-only tests/gtx/` succeeds on no-GPU box (no torch.cuda assertion).
6. FP8 strategy + 28-kernel scope DEFAULTS captured in `09-SCOPE-DECISION.md` with user sign-off (OR revision-pass signaled).
7. Pre-migration wheel size baseline pinned at `09-pre-wheel-size.txt` (BM-06 baseline).
8. Wave-end gate document `09-00-WAVE-GATE.md` shows 6-op smoke PASS + tile-2 PASS + ABS walltime in 85-105s band.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-00-SUMMARY.md`
</output>
</content>
</invoke>