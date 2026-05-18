---
phase: 09-backend-migration-numpy-cupy
plan: 01b
type: execute
wave: 3
# CONTEXT D-05 Wave 1 = plans 09-01a + 09-01b. This is part 2 of 2 (register_file + npu.py + test_csr_registry_chain + gate).
# B-4 split: register_file + npu state arrays + Wave gate.
depends_on:
  - "01a"
files_modified:
  - src/main/python/riscv/gtx/unit/register_file.py
  - src/main/python/riscv/gtx/npu.py
  - tests/gtx/test_csr_registry_chain.py
autonomous: false
requirements:
  - BM-02
user_setup: []

must_haves:
  truths:
    - "`unit/register_file.py` SPR int64 storage uses `xp.zeros(shape, dtype=xp.int64)`."
    - "`npu.py` `_mxe_accum`, `_credit_ld`, `_credit_st` allocations use xp."
    - "`npu.py` line 354 `.cpu()` chain replaced with `to_host()`."
    - "Wave-end perf gate: ABS strict walltime within 85-105s band (D-08). If xp=cupy + SPR-on-device exceeds 105s, RegisterFile reverts to host-pinned numpy exception (documented)."
    - "Wave-end correctness gate: 6-op smoke + tile-2 unit test all PASS."
    - "`tests/gtx/test_csr_registry_chain.py` is torch-free with numpy-based dtype assertions."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/register_file.py"
      provides: "RegisterFile int64 SPR storage on xp backend"
      contains: "from ..config_params import xp"
    - path: "src/main/python/riscv/gtx/npu.py"
      provides: "GtxNpu state arrays (mxe_accum, credit_ld/st, lspr RegisterFile) using xp"
      contains: "from .config_params import xp"
    - path: ".planning/phases/09-backend-migration-numpy-cupy/09-01-WAVE-GATE.md"
      provides: "Wave 1 gate document with smoke/tile-2/perf measurements + VRAM/SPR exception decisions"
      contains: "## ABS Strict Walltime"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/register_file.py"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "`xp.zeros(..., dtype=xp.int64)` for SPR storage"
      pattern: "xp\\.zeros"
    - from: "src/main/python/riscv/gtx/npu.py"
      to: "src/main/python/riscv/gtx/unit/register_file.py"
      via: "RegisterFile instantiation without `device=` kwarg (post-Wave-1a interface)"
      pattern: "RegisterFile\\("
---

<objective>
Wave 1 (part b): Port `unit/register_file.py` (SPR int64) and `npu.py` allocation sites (`_mxe_accum`, `_credit_ld`, `_credit_st`, RegisterFile instantiation) to xp. Port `tests/gtx/test_csr_registry_chain.py` off torch.int64 dtype assertions. Apply D-11 (RegisterFile follows scratchpad device) and add SPR-perf exception path per CONTEXT verification requirements. Final task = Wave 1 gate document.

Purpose: Completes Wave 1 storage-layer port started in 09-01a. Engines + ops (Wave 2) need RegisterFile + GtxNpu state arrays on xp before they can dispatch.

Output: 2 source files + 1 test file ported to xp; Wave 1 gate doc with ABS perf number + VRAM/SPR exception decisions for cupy.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md
@.planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md
@.planning/phases/09-backend-migration-numpy-cupy/09-00-SUMMARY.md
@.planning/phases/09-backend-migration-numpy-cupy/09-01a-SUMMARY.md
@src/main/python/riscv/gtx/unit/register_file.py
@src/main/python/riscv/gtx/npu.py
@tests/gtx/test_csr_registry_chain.py
@CLAUDE.md

<interfaces>
<!-- xp/helpers + memory.py interface (established by Wave 0 + 1a). -->

From src/main/python/riscv/gtx/config_params.py:
```python
xp           # numpy module (default) or cupy module (GTX_USE_CUDA=1)
to_host      # callable: cupy→numpy bridge (identity on numpy path)
to_device    # callable: numpy→cupy bridge (identity on numpy path)
```

Wave 1b establishes for downstream waves:
- `RegisterFile.read(addr)` returns a Python `int` (xp scalar `.item()` boundary)
- `RegisterFile.write(addr, value)` accepts Python `int` and stores via `xp.int64` cast
- `RegisterFile.__init__(shape)` — NO `device=` kwarg
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port unit/register_file.py — SPR int64 storage to xp; update RegisterFile interface</name>
  <files>src/main/python/riscv/gtx/unit/register_file.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/register_file.py (full file — torch sites at line 19 + ~line 80 per RESEARCH canonical_refs)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-11 SPR-device follow + perf-exception path)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pattern 3 — in-place ops; Open Question #5 — SPR-on-GPU perf)
    - src/main/python/riscv/gtx/config_params.py (post-Wave-0)
  </read_first>
  <behavior>
    - Test 1 `test_register_file_int64`: `RegisterFile(shape=(N,))` allocates `xp.zeros((N,), dtype=xp.int64)`.
    - Test 2 `test_register_file_write_read_roundtrip`: `rf.write(addr, 0xCAFE); assert rf.read(addr) == 0xCAFE`.
    - Test 3 `test_register_file_int64_max`: Writing `0x7FFFFFFFFFFFFFFF` (max int64) preserves the value (no overflow / sign issues).
    - Test 4 `test_no_torch_in_register_file`: `grep -c "torch" src/main/python/riscv/gtx/unit/register_file.py` returns 0.
    - Existing `tests/gtx/test_csr_registry_chain.py` still passes (Task 3 will update it for dtype assertion changes).
  </behavior>
  <action>
    **Line 19** — Replace `import torch` with `from ..config_params import xp`.

    **Line ~80 area** — torch.zeros allocation. Locate `torch.zeros(shape, dtype=torch.int64, device=device)` and replace with `xp.zeros(shape, dtype=xp.int64)`. Drop the `device=` kwarg entirely (xp is device-implicit).

    **All `.copy_(torch.as_tensor(value, dtype=torch.int64))` sites** — Replace with one of these idiomatic xp patterns:
    ```python
    # If single-cell write:
    self._tensor[addr] = xp.int64(value)  # cast Python int to xp scalar

    # If multi-cell or full-array write:
    xp.copyto(self._tensor[slice], xp.asarray(value, dtype=xp.int64))
    ```
    `xp.copyto` is supported on both numpy (NumPy ≥ 1.7) and cupy (matching API per RESEARCH Pattern 3 table).

    **All `.item()` calls** — Keep them. Both `numpy.int64` and `cupy.int64` expose `.item()` returning Python int. This is the host-int boundary used by SPR `read()`.

    **All `.cpu()` / `.numpy()` chains in register_file.py** — Replace with `to_host(...)` if present.

    **Constructor signature** — If `__init__(self, shape, ..., device=DEVICE)` exists, remove the `device=` param. Update any internal callers; downstream Task 2 fixes the npu.py call site.

    **DO NOT touch:** Public method names (`read`, `write`, `read_field`, `write_field`, bit-field accessor names). Keep bit-field semantics identical — only the storage backend changes.

    Per CLAUDE.md surgical-changes rule: only edit what xp port requires.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/register_file.py && uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/register_file.py` returns 0.
    - `grep -c "from ..config_params import xp" src/main/python/riscv/gtx/unit/register_file.py` returns at least 1.
    - `grep -c "xp.zeros.*xp.int64" src/main/python/riscv/gtx/unit/register_file.py` returns at least 1.
    - `grep -c "device=" src/main/python/riscv/gtx/unit/register_file.py` returns 0 (no `device=` kwarg anywhere).
    - `uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov` exits 0 (note: may need Task 3 dtype-assertion updates).
    - `uv run python -c "from riscv.gtx.unit.register_file import RegisterFile; rf = RegisterFile((16,)); rf.write(5, 0xCAFE); assert rf.read(5) == 0xCAFE"` exits 0.
  </acceptance_criteria>
  <done>register_file.py allocates SPR storage on xp, no torch references, basic round-trip + bit-field operations preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Port npu.py — _mxe_accum / _credit_ld / _credit_st / RegisterFile instantiation to xp; replace .cpu() at line 354 with to_host()</name>
  <files>src/main/python/riscv/gtx/npu.py</files>
  <read_first>
    - src/main/python/riscv/gtx/npu.py (full file — torch sites at lines 12, 19, 94-106, 354 per RESEARCH canonical_refs)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (Wave 1 mapping for npu.py)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pattern 4 — FP32-accumulate; Pitfall 3 — atexit ordering)
    - src/main/python/riscv/gtx/unit/register_file.py (post-Task-1 — note constructor signature change: no `device=` kwarg)
    - src/main/python/riscv/gtx/unit/memory.py (post-Wave-1a state)
  </read_first>
  <behavior>
    - Test 1 `test_npu_construct`: `GtxNpu()` constructs without error in a torch-uninstalled venv.
    - Test 2 `test_mxe_accum_shape`: `npu._mxe_accum` is an `xp.ndarray` of expected shape `(GTX_NEST_NUM, GTX_SPU_NUM, ...)` with dtype `xp.float32`.
    - Test 3 `test_credit_ld_st_shape`: `_credit_ld` / `_credit_st` allocated via xp with appropriate int dtype.
    - Test 4 `test_no_torch_in_npu`: `grep -c "torch" src/main/python/riscv/gtx/npu.py` returns 0.
  </behavior>
  <action>
    **Line 12** — Remove `import torch`.

    **Line 19 area** — If `from .config_params import DEVICE` exists, replace with `from .config_params import xp, to_host, to_device`.

    **Lines 94-106 area** — RegisterFile + state-array allocations. Concrete replacements:
    ```python
    # BEFORE (RegisterFile with device=):
    self.lspr = RegisterFile((GTX_NEST_NUM, GTX_LSPR_COUNT), device=DEVICE)
    self.nspr = RegisterFile((GTX_NEST_NUM, GTX_NSPR_COUNT), device=DEVICE)
    self.gspr = RegisterFile((GTX_GSPR_COUNT,), device=DEVICE)
    # AFTER (post-Task-1 RegisterFile has no device kwarg):
    self.lspr = RegisterFile((GTX_NEST_NUM, GTX_LSPR_COUNT))
    self.nspr = RegisterFile((GTX_NEST_NUM, GTX_NSPR_COUNT))
    self.gspr = RegisterFile((GTX_GSPR_COUNT,))

    # BEFORE (state arrays):
    self._mxe_accum = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM, MM_ACCUM_SIZE), dtype=torch.float32, device=DEVICE)
    self._credit_ld = torch.zeros((GTX_NEST_NUM,), dtype=torch.int32, device=DEVICE)
    self._credit_st = torch.zeros((GTX_NEST_NUM,), dtype=torch.int32, device=DEVICE)
    # AFTER:
    self._mxe_accum = xp.zeros((GTX_NEST_NUM, GTX_SPU_NUM, MM_ACCUM_SIZE), dtype=xp.float32)
    self._credit_ld = xp.zeros((GTX_NEST_NUM,), dtype=xp.int32)
    self._credit_st = xp.zeros((GTX_NEST_NUM,), dtype=xp.int32)
    ```
    Use the exact MXE_ACCUM dim constant that already appears in the file (don't invent — copy verbatim from the BEFORE).

    **Line 354 area** — `.cpu()` call. Replace:
    ```python
    # BEFORE: some_tensor.detach().cpu().numpy()
    # AFTER:  to_host(some_array)  # xp-native (no .detach() for numpy/cupy)
    ```
    If the surrounding context expected a torch.Tensor (e.g., for `.detach()`), update the consumer to expect a plain xp.ndarray. Verify by grepping the call site's caller.

    **DO NOT touch:**
    - The custom0/1/2/3 dispatch methods (those are the ROCC base-class virtuals).
    - The WJOIN SystemExit / GTX_NO_EXIT handling (MEMORY abc / FSM invariants from project memory).
    - Any RegisterFile.read()/write() call site signatures (only the allocator changed).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/npu.py && uv run pytest tests/gtx/test_npu_construct.py tests/gtx/test_warp.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/npu.py` returns 0.
    - `grep -c "from .config_params import xp" src/main/python/riscv/gtx/npu.py` returns at least 1.
    - `grep -c "xp.zeros" src/main/python/riscv/gtx/npu.py` returns at least 3 (_mxe_accum, _credit_ld, _credit_st).
    - `grep -c "to_host" src/main/python/riscv/gtx/npu.py` returns at least 1 (line 354 area).
    - `grep -c "device=" src/main/python/riscv/gtx/npu.py` returns 0.
    - `grep -c "DEVICE" src/main/python/riscv/gtx/npu.py` returns 0.
    - `uv run python -c "from riscv.gtx.npu import GtxNpu; npu = GtxNpu(); print(type(npu._mxe_accum).__module__)"` prints `numpy` (default xp).
    - `uv run pytest tests/gtx/test_npu_construct.py tests/gtx/test_warp.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>npu.py constructor uses xp for all state arrays + RegisterFile (no device kwarg); construct/warp tests pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Port tests/gtx/test_csr_registry_chain.py off torch.int64 dtype assertions</name>
  <files>tests/gtx/test_csr_registry_chain.py</files>
  <read_first>
    - tests/gtx/test_csr_registry_chain.py (full file — torch references per CONTEXT D-16)
    - src/main/python/riscv/gtx/unit/register_file.py (post-Task-1 state)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-16 — tests/gtx port scope)
  </read_first>
  <behavior>
    - All `torch.int64` references become `xp.int64` (where xp = numpy in test env) or plain `np.int64`.
    - All `torch.Tensor` dtype assertions become `np.dtype('int64')` comparisons.
    - All `import torch` lines removed.
    - Tests still PASS with semantically equivalent assertions.
  </behavior>
  <action>
    Mechanical 1:1 substitution (use grep to find every site, edit each one):

    1. Top of file: `import torch` → `import numpy as np` (or remove entirely if only used for dtype constants).
    2. `torch.int64` → `np.int64`.
    3. `torch.Tensor` (in type hints or isinstance checks) → `np.ndarray`.
    4. `.dtype is torch.int64` → `.dtype == np.int64` (numpy dtype comparison uses `==`, not `is`).
    5. `torch.zeros(...)` / `torch.tensor(...)` → `np.zeros(...)` / `np.array(...)`.
    6. `.cpu().numpy()` / `.detach().numpy()` → drop the chain entirely (already numpy).
    7. `.item()` → keep (numpy scalar `.item()` exists).

    Update any test name or docstring referencing "torch tensor" to "ndarray". Comments mentioning torch are updated to reflect xp/numpy.

    **DO NOT change test semantics** — only the type/dtype boilerplate. If a test asserts that `rf._tensor.dtype is torch.int64`, after edit it asserts `rf._tensor.dtype == np.int64`. The bit-field roundtrip / RAW value tests stay identical in value.

    For any test using a `device` fixture or `torch.device("cpu")` literal: remove the device parameter; numpy has no device concept.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" tests/gtx/test_csr_registry_chain.py && uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." tests/gtx/test_csr_registry_chain.py` returns 0.
    - `uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov -v` exits 0 (same number of tests pass as before — count not reduced).
    - `grep -c "np.int64\|numpy.int64" tests/gtx/test_csr_registry_chain.py` returns at least 1.
  </acceptance_criteria>
  <done>test_csr_registry_chain.py is torch-free, all tests pass with numpy-based assertions.</done>
</task>

<task type="auto">
  <name>Task 4: Wave 1 gate — 6-op smoke + tile-2 + ABS perf measurement + VRAM/SPR exception decision</name>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-01-WAVE-GATE.md</files>
  <read_first>
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-07, D-08, D-10 verification, D-11 verification)
    - .planning/phases/09-backend-migration-numpy-cupy/09-00-WAVE-GATE.md (Wave 0 baseline number for comparison)
    - tests/gtx/test_regression_fw_full_sweep.py (smoke set entry)
  </read_first>
  <action>
    Run these commands and record results:

    1. Smoke set:
       ```bash
       uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v
       ```
    2. Tile-2:
       ```bash
       uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v
       ```
    3. ABS strict walltime (numpy path, no GPU):
       ```bash
       /usr/bin/time -f "%e" uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov 2>&1 | tail -3
       ```

    Author `.planning/phases/09-backend-migration-numpy-cupy/09-01-WAVE-GATE.md`:

    ```markdown
    # Wave 1 Gate Results

    Date: <YYYY-MM-DD>
    Commit: <sha after Task 3>

    ## Smoke Set (6 ops)
    Result: <PASS | FAIL>
    Output: <captured last 20 lines>

    ## Tile-2 Unit Test
    Result: <PASS | FAIL>

    ## ABS Strict Walltime (D-08: 85-105s)
    Wall: <X.XXs>
    Wave 0 baseline: <from 09-00-WAVE-GATE.md>
    In-budget: <YES | NO>

    ## D-10 DDR-on-GPU Verification
    xp=numpy path: walltime above (no VRAM concern; host RAM)
    xp=cupy path (if GPU box available):
      Command: `GTX_USE_CUDA=1 uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v`
      Result: <SKIP — no GPU available | PASS | OOM>
      Notes: 4 GiB default DDR alloc on cupy. If failed with OOM on <12 GB VRAM card, document `GTX_DDR_SIZE=1G` workaround.

    ## D-11 SPR-on-GPU Perf Verification
    xp=numpy path: walltime above.
    xp=cupy path (if GPU box available):
      Walltime: <X.XXs>
      In-budget (≤105s): <YES | NO>
      Decision: <follow scratchpad device | host-pinned numpy exception>
      If host-pinned: note exit criterion (e.g., "revert to xp when numba cuda.jit lands in P10").

    ## Wave 1 Sign-Off
    - [x] memory.py torch-free, xp.zeros for scratchpads + DDR (Plan 09-01a)
    - [x] register_file.py torch-free, SPR int64 via xp
    - [x] npu.py constructor uses xp; .cpu() → to_host()
    - [x] test_csr_registry_chain.py torch-free
    - [x] Smoke set GREEN
    - [x] Tile-2 GREEN
    - [x] ABS walltime in 85-105s band (numpy path)
    - [x] D-10/D-11 verification documented (SKIP if no GPU)
    ```

    If any step fails, **DO NOT proceed to Wave 2**. Record the failure in the `## Failures` section and signal planner.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v && uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `.planning/phases/09-backend-migration-numpy-cupy/09-01-WAVE-GATE.md` exists.
    - File contains `Result: PASS` for both smoke and tile-2 sections.
    - File contains `Wall: ` line in ABS walltime section with measured value.
    - File contains `In-budget: YES` or documented justification.
    - File contains D-10 + D-11 sections (even if `SKIP — no GPU available`).
    - `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS'` exits 0.
  </acceptance_criteria>
  <done>Wave 1 gate document records GREEN smoke + tile-2 + perf in 85-105s; VRAM/SPR exceptions documented or marked SKIP; Wave 2 unblocked.</done>
</task>

</tasks>

<verification>
- Storage layer fully torch-free: `grep -rn "import torch\|torch\." src/main/python/riscv/gtx/unit/memory.py src/main/python/riscv/gtx/unit/register_file.py src/main/python/riscv/gtx/npu.py | wc -l` returns 0.
- BM-02 invariants: ABS strict still PASS + tile-2 still PASS + perf in 85-105s.
- Wave 2 entry: gate document signed.
</verification>

<success_criteria>
1. register_file.py + npu.py all import xp from config_params and have zero `import torch` references.
2. RegisterFile SPR storage uses xp.zeros(..., dtype=xp.int64); no device= kwargs.
3. GtxNpu constructor instantiates _mxe_accum / _credit_ld / _credit_st via xp.
4. test_csr_registry_chain.py is torch-free with numpy-equivalent assertions.
5. Wave 1 gate doc shows: 6-op smoke PASS + tile-2 PASS + ABS walltime in 85-105s + D-10/D-11 verification recorded.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-01b-SUMMARY.md`
</output>
</content>
</invoke>