---
phase: 09-backend-migration-numpy-cupy
plan: 03
type: execute
wave: 5
depends_on:
  - 09-backend-migration-numpy-cupy/02b-engines
files_modified:
  - src/main/python/riscv/gtx/tloop_buffer.py
  - src/main/python/riscv/gtx/_verify.py
  - src/main/python/riscv/gtx/__init__.py
  - tests/gtx/test_mcast_copy_mem.py
  - pyproject.toml
  - .planning/REQUIREMENTS.md
  - CLAUDE.md
autonomous: false
requirements:
  - BM-04
  - BM-05
  - BM-06

user_setup:
  - service: cupy
    why: "Optional GPU backend (xp=cupy). User opts in via `pip install spike[cuda]` + `GTX_USE_CUDA=1`."
    env_vars:
      - name: GTX_USE_CUDA
        source: "User-set; 1 to enable cupy path."
      - name: GTX_DDR_SIZE
        source: "Recommended `1G` on consumer GPUs <12 GB VRAM (see README)."
    dashboard_config: []

must_haves:
  truths:
    - "`tloop_buffer._execute_fused` is torch-free; D-15 1:1 drop-in (`torch.abs` -> `xp.abs`, `torch.negative` -> `xp.negative`, `torch.exp` -> `xp.exp`)."
    - "`_verify.py` uses `np.frombuffer` (not `torch.frombuffer`); torch import line 9 removed."
    - "`__init__.py` lines 80-84 torch ImportError surface fully gone; silent ImportError swallow at 54-68 audited (logs warning AT LEAST with exception class)."
    - "`tests/gtx/test_mcast_copy_mem.py` is torch-free (17 refs ported per CONTEXT D-16)."
    - "`pyproject.toml` removes `torch`/`torchvision` deps + `[tool.uv.sources]` torch entries + `[[tool.uv.index]] pytorch-cu126`; adds `[project.optional-dependencies] cuda = ['cupy-cuda12x>=13,<15']`."
    - "REQUIREMENTS.md `### Milestone v1.1 Post-Ship Polish` section contains BM-01..06 entries transcribed from ROADMAP success criteria, with proper Coverage table update (50 + 14 = 64 total)."
    - "CLAUDE.md `## Dependencies` section reflects NumPy default + CuPy opt-in (PyTorch removed)."
    - "Wheel size delta measured: `du -h dist/spike-*.whl` before/after recorded in 09-03-WAVE-GATE.md; delta should be NEGATIVE (PyTorch removal expected to reduce by 50-200 MB per RESEARCH BM-06 row)."
    - "Final phase gate: full 84-op vendor sweep + tile-2 + ABS perf in 85-105s + `grep -rn 'import torch' src/main/python/riscv/gtx/` returns 0."
  artifacts:
    - path: "src/main/python/riscv/gtx/tloop_buffer.py"
      provides: "_execute_fused fast path on xp (D-15 1:1)"
      contains: "from .config_params import xp"
    - path: "src/main/python/riscv/gtx/_verify.py"
      provides: "compare_hex using np.frombuffer"
      contains: "np.frombuffer"
    - path: "pyproject.toml"
      provides: "torch removed; cuda extras added"
      contains: "[project.optional-dependencies]"
    - path: ".planning/REQUIREMENTS.md"
      provides: "BM-01..06 transcribed entries + Coverage table sync"
      contains: "BM-01"
    - path: "CLAUDE.md"
      provides: "Dependencies section reflects NumPy + CuPy opt-in"
      contains: "CuPy"
    - path: ".planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md"
      provides: "Final phase gate doc with wheel size delta + full sweep result"
      contains: "## Wheel Size Delta"
  key_links:
    - from: "pyproject.toml"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "torch removed from deps; cupy in optional [cuda] extras unlocks GTX_USE_CUDA=1 path"
      pattern: "cupy-cuda12x"
    - from: "src/main/python/riscv/gtx/_verify.py"
      to: "tests/gtx/test_regression_fw_full_sweep.py"
      via: "compare_hex consumer in vendor sweep harness"
      pattern: "np.frombuffer"
---

<objective>
Wave 3: Final cleanup. Port the last 4 source files with torch references (`tloop_buffer.py`, `_verify.py`, `__init__.py`, `tests/gtx/test_mcast_copy_mem.py`), then land the project-wide artifacts: pyproject.toml dependency surgery (remove torch, add `[cuda]` extras), REQUIREMENTS.md BM-01..06 transcription, CLAUDE.md "Dependencies" section update, and the wheel size delta measurement.

Purpose: Closes the migration. After this wave, `grep -rn 'import torch' src/main/python/riscv/gtx/` returns 0. `pip install spike` no longer pulls PyTorch. `pip install spike[cuda]` is the documented opt-in path. The full 84-op vendor sweep + tile-2 + ABS perf serves as the final phase gate.

Output: 4 source files ported; pyproject.toml clean; REQUIREMENTS.md complete; CLAUDE.md current; phase-final gate doc with wheel size delta and all-green test summary.
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
@.planning/phases/09-backend-migration-numpy-cupy/09-02b-SUMMARY.md
@src/main/python/riscv/gtx/tloop_buffer.py
@src/main/python/riscv/gtx/_verify.py
@src/main/python/riscv/gtx/__init__.py
@pyproject.toml
@CLAUDE.md

<interfaces>
<!-- All op + engine + memory + register_file modules already torch-free by end of Wave 2b. -->

Remaining torch sites (verified by RESEARCH canonical_refs):
- src/main/python/riscv/gtx/tloop_buffer.py:423 (local `import torch` in `_execute_fused`)
- src/main/python/riscv/gtx/_verify.py:9 (`import torch` + `torch.frombuffer` usage at lines 45-46)
- src/main/python/riscv/gtx/__init__.py:80-84 (torch ImportError surface)
- tests/gtx/test_mcast_copy_mem.py (17 torch refs per CONTEXT D-16)
- pyproject.toml: lines 60-61 (torch/torchvision deps) + lines 196-202 ([tool.uv.sources] torch + [[tool.uv.index]] pytorch-cu126)

REQUIREMENTS.md target section (current state — missing BM-*):
```
### Multi-tile DMA Parity (MTDMA)
- [x] MTDMA-01 ... [x] MTDMA-04
### Vendor Test Wire-up (VTW)
- [x] VTW-01 ... [x] VTW-04
```
After Wave 3 insertion: add `### Backend Migration (BM)` subsection with BM-01..06.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port tloop_buffer.py _execute_fused — D-15 1:1 drop-in</name>
  <files>src/main/python/riscv/gtx/tloop_buffer.py</files>
  <read_first>
    - src/main/python/riscv/gtx/tloop_buffer.py (full file — focus on lines 415-486; local `import torch` at line 423)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-15 — 1:1 drop-in for fusion fast path)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Code Example 4 — _execute_fused port verbatim; Pitfall 1 — `.view(n, vec_size)` reshape site at 466-470)
  </read_first>
  <behavior>
    - Test 1 `test_tloop_fusion`: ABS fusion fast path fires on `_VEC_UNARY_MNEMONICS` (verified by snapshot of `tloop_buffer.fused_replay_count` after ABS run > 0).
    - Test 2 ABS smoke PASS (fusion path is the perf-critical path for ABS).
    - Test 3 `TRANSPARENT_MNEMONICS` / `BUFFERABLE_MNEMONICS` frozensets unchanged in value.
    - Test 4 no torch reference in tloop_buffer.py.
  </behavior>
  <action>
    Open `src/main/python/riscv/gtx/tloop_buffer.py`. Locate `_execute_fused` function (around lines 415-486).

    **Line 423** — Remove the local `import torch  # local to keep module import cycle-free at top level`. Replace with module-level import at top of file:
    ```python
    # At top of file, alongside other imports:
    from .config_params import xp
    ```

    **Lines 466-470 — `.view(torch.float16).view(n, vec_size)` chain (RESEARCH Pitfall 1):**
    ```python
    # BEFORE:
    src_f16 = (
        l2[src_base:src_base + total_bytes]
        .view(torch.float16)
        .view(n, vec_size)
    )
    # AFTER:
    src_f16 = (
        l2[src_base:src_base + total_bytes]
        .view(xp.float16)
        .reshape(n, vec_size)
    )
    ```

    **Lines 471-486 area — `.copy_()` / `torch.abs/neg/exp` / activation calls:**

    Apply the mapping table:
    | torch | xp |
    |-------|----|
    | `torch.abs(x)` | `xp.abs(x)` |
    | `torch.negative(x)` | `xp.negative(x)` |
    | `torch.exp(x)` | `xp.exp(x)` (uses fp32 internal accumulate via existing pattern) |
    | `dst.copy_(src)` | `xp.copyto(dst, src)` |
    | `tensor.view(torch.uint8)` | `tensor.view(xp.uint8)` (byte-reinterpret, fine on numpy/cupy) |

    Verify the resulting `_execute_fused` matches RESEARCH Code Example 4 closely (lines 696-718 of 09-RESEARCH.md).

    **DO NOT touch:**
    - `TRANSPARENT_MNEMONICS` frozenset.
    - `BUFFERABLE_MNEMONICS` frozenset.
    - `_VEC_UNARY_MNEMONICS` frozenset (D-15 contract preserved).
    - The fusion decision logic (`_should_fuse` / `_can_buffer`).
    - The replay-only regression hook (test_tloop_fusion guards this).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "import torch\|torch\." src/main/python/riscv/gtx/tloop_buffer.py && uv run pytest tests/gtx/ -k 'tloop_fusion or fused' --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/tloop_buffer.py` returns 0.
    - `grep -c "from .config_params import xp" src/main/python/riscv/gtx/tloop_buffer.py` returns 1.
    - `grep -nE "\.view\(n, ?vec_size\)" src/main/python/riscv/gtx/tloop_buffer.py` returns 0 (uses .reshape).
    - `grep -c "xp.copyto" src/main/python/riscv/gtx/tloop_buffer.py` returns at least 1.
    - `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v` exits 0.
  </acceptance_criteria>
  <done>tloop_buffer.py torch-free; fusion fast path preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Port _verify.py + audit __init__.py silent ImportError swallow</name>
  <files>src/main/python/riscv/gtx/_verify.py, src/main/python/riscv/gtx/__init__.py</files>
  <read_first>
    - src/main/python/riscv/gtx/_verify.py (full file — torch site at line 9 + frombuffer at lines 45-46)
    - src/main/python/riscv/gtx/__init__.py (full file — lines 54-68 silent ImportError swallow; line 80-84 already removed by Wave 0)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pitfall 7 — torch.frombuffer; Pitfall 8 — silent ImportError swallow audit)
    - Memory project_gtx_extension_silent_import_failure.md (referenced in CONTEXT)
  </read_first>
  <behavior>
    - Test 1 `compare_hex` byte-exact for ABS golden vs DDR dump (no semantic change).
    - Test 2 `python -m riscv.gtx._verify result.hex golden.hex --fp16 --ulp 1 --atol 0.001 --strict` returns 0.
    - Test 3 ImportError surface tightened: if `npu` import fails, the warning includes the exception class name (not just generic ImportWarning).
    - Test 4 `uv run python -W error::ImportWarning -c "import riscv.gtx"` exits 0 if all submodules import cleanly.
  </behavior>
  <action>
    **Edit A — `src/main/python/riscv/gtx/_verify.py`:**

    **Line 9** — Remove `import torch`. Add `import numpy as np` if not already present.

    **Lines 45-46 area** — `torch.frombuffer(bytes_object, dtype=torch.float16)`:
    ```python
    # BEFORE: torch.frombuffer(bytes_object, dtype=torch.float16)
    # AFTER:  np.frombuffer(bytes_object, dtype=np.float16)
    ```

    The `_verify.py` module is host-only (file I/O). No xp needed. Use bare numpy.

    Apply standard mapping to any other torch references in the file:
    | torch | numpy |
    |-------|-------|
    | `torch.float16` | `np.float16` |
    | `torch.uint16` | `np.uint16` |
    | `tensor.numpy()` | drop (already ndarray) |
    | `tensor.view(np.uint16)` | `tensor.view(np.uint16)` (dtype-only view) |

    **Edit B — `src/main/python/riscv/gtx/__init__.py`:**

    Audit lines 54-68 silent ImportError swallow. Current pattern likely:
    ```python
    try:
        from . import npu
    except ImportError as e:
        warnings.warn(f"failed to import npu: {e}", ImportWarning)
    ```

    Tighten to:
    ```python
    try:
        from . import npu
    except ImportError as e:
        # AT LEAST surface the exception class for diagnostic.
        # See memory `project_gtx_extension_silent_import_failure.md` for prior cascade.
        warnings.warn(
            f"riscv.gtx submodule import failed ({type(e).__name__}): {e}",
            ImportWarning,
            stacklevel=2,
        )
    ```

    Same pattern for any other silent swallow blocks in the file. Add `stacklevel=2` to surface the caller's frame.

    **DO NOT touch:**
    - The LE byte-order assertion at line 37 (FOUND-01 invariant).
    - The public API exports (GtxNpu, etc.).
    - Per-platform conditional imports unrelated to torch.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "import torch\|torch\." src/main/python/riscv/gtx/_verify.py && uv run python -W error::ImportWarning -c "import riscv.gtx" && uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/_verify.py` returns 0.
    - `grep -c "np.frombuffer" src/main/python/riscv/gtx/_verify.py` returns at least 1.
    - `grep -nE "^    import torch" src/main/python/riscv/gtx/__init__.py` returns 0 (Wave 0 should have already removed this).
    - `grep -c "type(e).__name__" src/main/python/riscv/gtx/__init__.py` returns at least 1 (tightened warning).
    - `uv run python -W error::ImportWarning -c "import riscv.gtx"` exits 0.
    - `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov` exits 0.
  </acceptance_criteria>
  <done>_verify.py torch-free; __init__.py warnings carry exception class info.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Port tests/gtx/test_mcast_copy_mem.py (17 torch refs per CONTEXT D-16)</name>
  <files>tests/gtx/test_mcast_copy_mem.py</files>
  <read_first>
    - tests/gtx/test_mcast_copy_mem.py (full file — 17 torch refs per CONTEXT D-16; was added in 260518-ibf)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-16 — tests scope)
  </read_first>
  <behavior>
    - All 5 unit tests landed in 260518-ibf continue to pass.
    - File has 0 torch references.
    - Numerics preserved (mcast/copy.mem byte-exact behavior).
  </behavior>
  <action>
    Mechanical 1:1 substitution per the standard table (same as Wave 1 Task 4 / Wave 2a Tasks):
    | torch | numpy |
    |-------|-------|
    | `import torch` | `import numpy as np` |
    | `torch.tensor([...], dtype=torch.uint8)` | `np.array([...], dtype=np.uint8)` |
    | `torch.zeros(shape, dtype=torch.X)` | `np.zeros(shape, dtype=np.X)` |
    | `torch.float16` / `torch.uint8` / `torch.int64` | `np.float16` / `np.uint8` / `np.int64` |
    | `torch.equal(a, b)` | `np.array_equal(a, b)` |
    | `tensor.cpu().numpy()` | drop chain (already ndarray) |
    | `tensor.dtype is torch.X` | `arr.dtype == np.X` |

    For any seed helpers using torch random, replace with `np.random.default_rng(seed)` and `.standard_normal()` / `.integers()` calls.

    Preserve test names, parameter values, expected byte patterns verbatim.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" tests/gtx/test_mcast_copy_mem.py && uv run pytest tests/gtx/test_mcast_copy_mem.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." tests/gtx/test_mcast_copy_mem.py` returns 0.
    - `grep -c "import numpy as np" tests/gtx/test_mcast_copy_mem.py` returns at least 1.
    - `uv run pytest tests/gtx/test_mcast_copy_mem.py -x --no-cov` exits 0 (same number of tests pass as before).
  </acceptance_criteria>
  <done>test_mcast_copy_mem.py torch-free; all 5 unit tests pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: pyproject.toml dependency surgery — remove torch, add [cuda] extras</name>
  <files>pyproject.toml</files>
  <read_first>
    - pyproject.toml (full file — focus on `[project]` deps lines 60-61, `[project.optional-dependencies]` for `[fast]` extras, `[tool.uv.sources]` lines 196-201)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-17 — pyproject.toml surgery)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Open Question #1 — [cuda-jit] separation; "Special — pyproject.toml [tool.uv.sources]" lines 405-416)
  </read_first>
  <behavior>
    - `uv pip install -e .` succeeds without torch resolution.
    - `uv pip install -e ".[cuda]"` attempts to install cupy (may fail without CUDA toolkit; expected on CPU-only CI).
    - `pip wheel . -w dist/` produces a valid wheel (manylinux compliance preserved).
    - `cibuildwheel` matrix builds green for cp310-cp312.
  </behavior>
  <action>
    Four concrete edits in `pyproject.toml`:

    **1. Lines 60-61 area — `[project.dependencies]`:** Remove `torch` and `torchvision` entries.

    Before:
    ```toml
    dependencies = [
        "numpy>=2.0,<3",
        "torch",
        "torchvision",
        # ... others
    ]
    ```
    After:
    ```toml
    dependencies = [
        "numpy>=2.0,<3",
        # ... others (torch + torchvision removed per BM-01)
    ]
    ```

    **2. `[project.optional-dependencies]` — Add `[cuda]` extra:**

    Locate the existing optional-dependencies section (likely contains `fast = ["numba>=0.61.2,<0.66"]`, `dev = [...]`, etc.). Add:
    ```toml
    [project.optional-dependencies]
    fast = ["numba>=0.61.2,<0.66"]  # PRESERVE existing
    cuda = ["cupy-cuda12x>=13,<15"]  # NEW (D-17)
    # If user selected [cuda-jit] extras separation per RESEARCH Open Question #1:
    # cuda-jit = ["spike[fast,cuda]"]
    dev = [...]                       # PRESERVE existing
    ```

    Whether to add `cuda-jit` extras is plan-stage discretion per RESEARCH Open Question #1. **Default: skip `cuda-jit`** unless 09-SCOPE-DECISION.md specifies it. User can `pip install spike[fast,cuda]` directly.

    **3. Lines 196-202 — `[[tool.uv.index]]` + `[tool.uv.sources]`:** Remove the pytorch-cu126 index and the torch/torchvision source mappings entirely.

    Before:
    ```toml
    [[tool.uv.index]]
    name = "pytorch-cu126"
    url = "https://download.pytorch.org/whl/cu126"
    explicit = true

    [tool.uv.sources]
    torch = [{ index = "pytorch-cu126" }]
    torchvision = [{ index = "pytorch-cu126" }]
    ```
    After: **DELETE all 4+ entries.** If `[tool.uv.sources]` is now empty, remove the section header too.

    **4. `[tool.cibuildwheel]` — Verify `test-extras`:**

    If the matrix has `test-extras = ["fast"]`, leave it. Do NOT add `cuda` to default test extras (cloud cibuildwheel runners have no GPU). Optionally add a separate matrix entry for cuda tests that SKIPs gracefully if cupy import fails.

    **DO NOT touch:**
    - `[build-system]` (setuptools / pybind11 deps).
    - `requires-python = ">=3.10"` (P1 D-08).
    - `[tool.setuptools.package-data]` (wheel content).
    - cibuildwheel `before-all = "dtc"` (P1 platform req).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -cE '"torch"|"torchvision"|pytorch-cu126' pyproject.toml && uv pip install -e . --dry-run 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c '"torch"' pyproject.toml` returns 0.
    - `grep -c '"torchvision"' pyproject.toml` returns 0.
    - `grep -c "pytorch-cu126" pyproject.toml` returns 0.
    - `grep -c "cupy-cuda12x" pyproject.toml` returns at least 1 (in `[cuda]` extras).
    - `grep -c "cuda = " pyproject.toml` returns at least 1.
    - `uv pip install -e . --dry-run 2>&1 | grep -c "torch"` returns 0 (resolution doesn't pull torch).
  </acceptance_criteria>
  <done>pyproject.toml: torch removed, [cuda] extras added, [tool.uv.sources] cleaned.</done>
</task>

<task type="auto">
  <name>Task 5: REQUIREMENTS.md — transcribe BM-01..06 from ROADMAP success criteria; update Coverage table</name>
  <files>.planning/REQUIREMENTS.md</files>
  <read_first>
    - .planning/REQUIREMENTS.md (full file — focus on `## Milestone v1.1 Post-Ship Polish` section near line 283; current state has MTDMA + VTW but NO BM-*)
    - .planning/ROADMAP.md (lines 274-280 — Phase 9 success criteria are the BM-* origin)
  </read_first>
  <behavior>
    - REQUIREMENTS.md gains a `### Backend Migration (BM)` subsection under `## Milestone v1.1 Post-Ship Polish`.
    - 6 entries BM-01..06 with checkboxes (marked complete per Wave 3 progress).
    - Coverage table updated: v1.1 total goes 14 -> 14 (already counted as 6); ensure "Combined: 64/64" is reflected if was 58/58.
    - "Last updated" footer line bumped.
  </behavior>
  <action>
    Open `.planning/REQUIREMENTS.md`. After the `### Vendor Test Wire-up (VTW)` subsection (around line 304-316), insert:

    ```markdown
    ### Backend Migration (BM)

    - [x] **BM-01**: `xp` alias scaffold + `GTX_USE_CUDA` env contract — `import torch`
      count = 0 across `src/main/python/riscv/gtx/`; `GTX_USE_CUDA=1` activates
      cupy path with fail-loud RuntimeError when cupy missing. `DEVICE` symbol
      removed.
    - [x] **BM-02**: NumPy port of memory layer — `unit/memory.py` (DDR +
      L0/L1/L2 scratchpads) and `unit/register_file.py` (SPR int64) use
      `xp.zeros`. ABS strict byte-exact PASS preserved. DDR-on-GPU verified
      with VRAM budget documented when `xp=cupy`.
    - [x] **BM-03**: NumPy port of dispatch + ops — `unit/ins/ops/{spr,mm,vec,act}.py`
      + `unit/context/{dma,mm,vec,act}_engine.py` all use xp. FP8 strategy locked
      to LUT-only path (`FP16_TO_FP8_LUT` / `FP8_TO_FP16_LUT` precomputed at
      import). GELU + RELU + SIGMOID + TANH + SOFTMAX strict PASS.
    - [x] **BM-04**: NumPy port of tloop/sloop fusion — `tloop_buffer._execute_fused`
      and `_verify.compare_hex` torch-free. ABS perf within ±10% of 94.82s
      baseline (target 85-105s window).
    - [x] **BM-05**: CuPy opt-in extras + GPU smoke test gated on `GTX_USE_CUDA`
      — `pyproject.toml` `[project.optional-dependencies] cuda = ["cupy-cuda12x>=13,<15"]`.
      ABS smoke (`tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]`)
      byte-identical between xp=numpy and xp=cupy paths.
    - [x] **BM-06**: CLAUDE.md "Dependencies" updated + wheel size delta recorded.
      PyTorch removed from runtime; wheel size delta measured at end of Wave 3
      (expected -50 to -200 MB per RESEARCH BM-06 row).
    ```

    Mark entries `[x]` only after each is verified at gate. If Wave 3 gate is partial, leave any uncompleted BM as `[ ]` and document in 09-03-WAVE-GATE.md.

    Update the Coverage Summary table at the top of REQUIREMENTS.md (around line 270):

    ```markdown
    - v1.0 requirements: 50 total -- 50 mapped, 100% coverage
    - v1.1 requirements: 14 total -- 14 mapped, 100% coverage (8 MTDMA/VTW + 6 BM)
    - Combined: 64 requirements ↔ 64 mapped ✓
    ```

    Update the phase distribution block (near line 273):

    ```markdown
    **Phase distribution:**
    - Phase 1 ... 7 (unchanged)
    - Phase 8 (Multi-tile DMA Parity, v1.1): 8 (MTDMA-01..04, VTW-01..04)
    - Phase 9 (Backend Migration, v1.1): 6 (BM-01..06)
    ```

    Update the footer "Last updated" line to today's date with note "BM-01..06 added".
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "BM-0" .planning/REQUIREMENTS.md && grep -c "64 requirements ↔ 64 mapped" .planning/REQUIREMENTS.md</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "^- \[x\] \*\*BM-0" .planning/REQUIREMENTS.md` returns 6 (one per BM-01..06, all marked complete).
    - `grep -c "### Backend Migration (BM)" .planning/REQUIREMENTS.md` returns 1.
    - `grep -c "Combined: 64" .planning/REQUIREMENTS.md` returns at least 1.
    - `grep -c "Phase 9 (Backend Migration" .planning/REQUIREMENTS.md` returns 1.
  </acceptance_criteria>
  <done>REQUIREMENTS.md complete with all 6 BM entries + coverage table sync.</done>
</task>

<task type="auto">
  <name>Task 6: CLAUDE.md Dependencies section update — NumPy default + CuPy opt-in</name>
  <files>CLAUDE.md</files>
  <read_first>
    - CLAUDE.md (full file — focus on `## Key Dependencies` section listing torch and friends)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (BM-06 doc gate)
  </read_first>
  <behavior>
    - "Key Dependencies" section reflects current truth: NumPy ≥ 2.0 default, CuPy opt-in via `[cuda]` extras.
    - PyTorch removed from dependency description.
    - GTX_USE_CUDA env var documented under "Configuration".
  </behavior>
  <action>
    Locate `## Key Dependencies` section in `CLAUDE.md` (lists `libriscv.so`, `libdisasm.a`, etc.). Within that block, remove any torch reference and add CuPy/NumPy summary. Concrete edits:

    1. If a `- **torch**` bullet exists, remove it entirely.
    2. Ensure `- **numpy** [>=2.0,<3] - Default array backend (NumPy 2.x). All gtx.* modules use the `xp` alias (numpy by default)` is present.
    3. Add new bullet: `- **cupy-cuda12x** [>=13,<15] (opt-in `pip install spike[cuda]`) - GPU backend; activates when `GTX_USE_CUDA=1` env-var is set. Fails loud if missing under that env`.

    Locate `## Configuration` section (lists `RISCV`, `PYSPIKE_LIBS`, `PYBIND11_DETAILED_ERROR_MESSAGES`). Add:
    ```
    - `GTX_USE_CUDA` - Opt-in for cupy backend (xp=cupy). Default unset (xp=numpy). When set to `1`/`true`, requires `cupy-cuda12x` installed via `pip install spike[cuda]`.
    - `GTX_DDR_SIZE` - DDR size override (default 4 GiB). Recommended `1G` on consumer GPUs with <12 GB VRAM when `xp=cupy`.
    ```

    Do NOT touch project description, technology stack overview, or unrelated dependency entries.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "cupy-cuda12x" CLAUDE.md && grep -c "GTX_USE_CUDA" CLAUDE.md</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "cupy-cuda12x" CLAUDE.md` returns at least 1.
    - `grep -c "GTX_USE_CUDA" CLAUDE.md` returns at least 1.
    - `grep -c "^- \*\*torch\*\*" CLAUDE.md` returns 0 (or only in historical context references; not in active deps).
    - `grep -c "GTX_DDR_SIZE" CLAUDE.md` returns at least 1.
  </acceptance_criteria>
  <done>CLAUDE.md Dependencies + Configuration sections reflect numpy/cupy stack.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 7: Final phase gate — full 84-op vendor sweep + tile-2 + ABS perf + wheel size delta</name>
  <what-built>
    Wave 3 final cleanup complete:
    - All `src/main/python/riscv/gtx/**.py` files torch-free (4 final modules: tloop_buffer, _verify, __init__, mcast test).
    - pyproject.toml: torch removed, [cuda] extras added, [tool.uv.sources] cleaned.
    - REQUIREMENTS.md: BM-01..06 transcribed + coverage table synced.
    - CLAUDE.md: Dependencies + Configuration sections updated.
    - All 5 prior wave gates GREEN.
  </what-built>
  <how-to-verify>
    1. **Final torch-free assertion (BM-01 success criterion):**
       ```bash
       grep -rn "import torch\|from torch" src/main/python/riscv/gtx/
       # Expected: 0 matches.
       ```

    2. **Full 84-op vendor sweep (BM-03 success criterion):**
       ```bash
       uv run pytest tests/gtx/test_regression_fw_full_sweep.py --no-cov -v
       # Expected: M passed + N skipped where M + N == 84 and M >= 12.
       ```

    3. **Tile-2 unit test (P8 MTDMA-03 invariant):**
       ```bash
       uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v
       # Expected: PASS.
       ```

    4. **ABS strict perf (BM-04 success criterion):**
       ```bash
       /usr/bin/time -f "%e" uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov 2>&1 | tail -3
       # Expected: walltime in 85-105s window (D-08).
       ```

    5. **Clean install (BM-01 + BM-05 + BM-06 ship gate):**
       ```bash
       # In a fresh venv:
       uv venv .venv-fresh && source .venv-fresh/bin/activate
       uv pip install -e .
       python -c "from riscv.gtx import GtxNpu; from riscv.gtx.config_params import xp; print(xp.__name__)"
       # Expected: prints `numpy`. No torch in installed deps.
       uv pip list | grep -ci torch  # expected 0
       deactivate
       ```

    6. **Wheel size delta (BM-06 success criterion):**
       ```bash
       # Build wheel:
       uv build --wheel
       du -h dist/spike-*.whl
       # Compare against pre-Phase-9 wheel size. Expected: delta NEGATIVE (50-200 MB smaller).
       ```

    7. **GPU smoke test (BM-05, manual — only if GPU box available):**
       ```bash
       # On a GPU machine with cupy installed:
       uv pip install -e ".[cuda]"
       GTX_USE_CUDA=1 uv run python -c "from riscv.gtx.config_params import xp; print(xp.__name__)"
       # Expected: prints `cupy`.
       GTX_USE_CUDA=1 uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v
       # Expected: ABS strict PASS byte-identical to numpy path.
       ```
       If no GPU available, mark as SKIP-with-justification.

    8. **REQUIREMENTS.md sync (BM-06 doc gate):**
       ```bash
       grep -c "^- \[x\] \*\*BM-0" .planning/REQUIREMENTS.md
       # Expected: 6 (one per BM-01..06).
       ```

    Author `.planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md`:
    ```markdown
    # Wave 3 / Phase 9 Final Gate Results

    Date: <YYYY-MM-DD>
    Commit: <sha after Task 6>

    ## Final Phase Gate

    ### Torch-Free Assertion
    `grep -rn "import torch" src/main/python/riscv/gtx/`: <0 matches | N matches — fail>

    ### Full 84-op Vendor Sweep
    M passed: <N>
    N skipped: <N>
    M + N: <should == 84>
    Result: <PASS | FAIL>

    ### Tile-2 (P8 MTDMA-03)
    Result: <PASS | FAIL>

    ### ABS Strict Perf
    Wall: <X.XXs>
    In-budget (85-105s): <YES | NO>

    ### Clean Install
    `uv pip install -e .` succeeds: <YES | NO>
    `pip list | grep -ci torch`: <0 (expected) | N>

    ### Wheel Size Delta
    Pre-migration: <X MB> (from git history or pre-Phase-9 commit)
    Post-migration: <Y MB>
    Delta: <Y-X MB> (expected NEGATIVE: -50 to -200 MB)

    ### GPU Smoke Test (BM-05)
    Result: <SKIP — no GPU | PASS | FAIL>
    xp.__name__ under GTX_USE_CUDA=1: <cupy | N/A>

    ## Phase 9 Sign-Off
    - [x] BM-01 — xp alias + DEVICE removed
    - [x] BM-02 — memory layer port
    - [x] BM-03 — dispatch + ops port
    - [x] BM-04 — tloop + verify port + perf budget
    - [x] BM-05 — cupy extras (manual GPU verify if available)
    - [x] BM-06 — CLAUDE.md + REQUIREMENTS.md + wheel size delta
    - [x] All 6 wave gates GREEN
    ```
  </how-to-verify>
  <resume-signal>
    Reply "approved" after verifying:
    1. All 7 checks pass.
    2. `09-03-WAVE-GATE.md` exists with all sections filled.
    3. Wheel size delta is NEGATIVE (or documented if positive — would be unexpected; could indicate cupy bringing transitive cost in optional extras manifest).
    4. GPU smoke test either passed or SKIPped with justification.

    If any check fails, describe failure mode and signal to planner for revision.
  </resume-signal>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md</files>
  <action>
    Pause for user verification. The EXECUTOR runs all 8 verification commands in `<how-to-verify>` above and authors `.planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md` capturing results. User reviews the doc + reply "approved" or describes failures.
  </action>
  <verify>
    <automated>test -f .planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md && grep -c "Final Phase Gate" .planning/phases/09-backend-migration-numpy-cupy/09-03-WAVE-GATE.md</automated>
  </verify>
  <done>09-03-WAVE-GATE.md captures all 8 verification check results. User approval signals Phase 9 ship-ready.</done>
</task>

</tasks>

<verification>
- Full migration done: `grep -rn "import torch\|from torch" src/main/python/riscv/gtx/ | wc -l` returns 0.
- pyproject.toml clean: no torch deps, no pytorch-cu126 index.
- Full 84-op vendor sweep passes.
- ABS perf in 85-105s.
- Wheel size delta recorded.
- REQUIREMENTS.md + CLAUDE.md current.
</verification>

<success_criteria>
1. `grep -rn "import torch\|from torch" src/main/python/riscv/gtx/` returns 0 matches (BM-01).
2. ABS strict byte-exact PASS preserved (BM-02..04 invariant).
3. GELU + 5 ACT-family ops strict PASS (BM-03).
4. CuPy opt-in works on a GPU machine, byte-identical to numpy (BM-05 — manual verify).
5. Wheel base size delta ≤ 0 MB vs pre-migration (BM-06).
6. CLAUDE.md "Dependencies" + REQUIREMENTS.md "BM-01..06" updated (BM-06).
7. Full Phase 9 gate doc `09-03-WAVE-GATE.md` exists with all sections.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-03-SUMMARY.md`
</output>
