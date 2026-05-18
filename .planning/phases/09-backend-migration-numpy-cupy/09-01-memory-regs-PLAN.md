---
phase: 09-backend-migration-numpy-cupy
plan: 01
type: execute
wave: 2
depends_on:
  - 09-backend-migration-numpy-cupy/00-scaffold
files_modified:
  - src/main/python/riscv/gtx/unit/memory.py
  - src/main/python/riscv/gtx/unit/register_file.py
  - src/main/python/riscv/gtx/npu.py
  - tests/gtx/test_csr_registry_chain.py
autonomous: false
requirements:
  - BM-02
user_setup: []

must_haves:
  truths:
    - "`unit/memory.py` allocates `_L2_GLOBAL`, `_L1_GLOBAL`, `_L0_GLOBAL` via `xp.zeros(..., dtype=xp.uint8)` instead of `torch.zeros(..., device=DEVICE)`."
    - "`DDR_MEMORY._bytes` allocation routes through `xp.zeros` (D-10: GPU when xp=cupy)."
    - "`DDR_MEMORY.ensure()` doubling-grow uses `xp.zeros` + slice copy; data preserved across grow."
    - "`ddr_save_to_hex` and `ddr_load_from_hex` invoke `to_host()` before file I/O (no `.cpu()` calls)."
    - "`unit/register_file.py` SPR int64 storage uses `xp.zeros(shape, dtype=xp.int64)`."
    - "`npu.py` `_mxe_accum`, `_credit_ld`, `_credit_st` allocations use xp."
    - "Wave-end perf gate: ABS strict walltime within 85-105s band (D-08). If xp=cupy + SPR-on-device exceeds 105s, RegisterFile reverts to host-pinned numpy exception (documented)."
    - "Wave-end correctness gate: 6-op smoke + tile-2 unit test all PASS."
    - "VRAM budget gate (when xp=cupy): doc note added to memory.py / README that consumer GPU <12 GB should set `GTX_DDR_SIZE=1G`."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/memory.py"
      provides: "Module-level scratchpad allocations + DDR_MEMORY using xp (numpy default, cupy when GTX_USE_CUDA=1)"
      contains: "from ..config_params import xp"
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
    - from: "src/main/python/riscv/gtx/unit/memory.py"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "`from ..config_params import xp, to_host, to_device`"
      pattern: "from ..config_params import xp"
    - from: "src/main/python/riscv/gtx/unit/register_file.py"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "`xp.zeros(..., dtype=xp.int64)` for SPR storage"
      pattern: "xp\\.zeros"
---

<objective>
Wave 1: Port the storage layer — `unit/memory.py` (L0/L1/L2 scratchpads + DDR), `unit/register_file.py` (SPR int64), and the `npu.py` allocation sites for `_mxe_accum`, `_credit_ld`, `_credit_st`, RegisterFile instantiation. Apply D-10 (DDR-on-GPU when xp=cupy) and D-11 (RegisterFile follows scratchpad device). Add VRAM-budget documentation + SPR-perf exception path per CONTEXT verification requirements.

Purpose: All compute ops in Wave 2 will allocate temporaries via `xp` AND read/write to these scratchpads. The storage layer MUST be xp-resident first; otherwise Wave 2's `xp.matmul(L1_view_f16, L1_view_f16)` would mix backends (no-op on numpy, error on cupy). This is the highest-VRAM-risk wave — DDR-on-GPU verification + SPR-perf measurement both gate Wave 2 entry.

Output: 3 source files ported to xp; conformance assertions via `test_csr_registry_chain.py` updates (D-16 starter); a Wave-end gate document recording the ABS perf number and the VRAM-budget decision for cupy.
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
@src/main/python/riscv/gtx/unit/memory.py
@src/main/python/riscv/gtx/unit/register_file.py
@src/main/python/riscv/gtx/npu.py
@CLAUDE.md

<interfaces>
<!-- xp/helpers interface (established by Wave 0). -->

From src/main/python/riscv/gtx/config_params.py:
```python
xp           # numpy module (default) or cupy module (GTX_USE_CUDA=1)
to_host      # callable: cupy→numpy bridge (identity on numpy path)
to_device    # callable: numpy→cupy bridge (identity on numpy path)
```

Wave 1 invariants for downstream waves:
- `mem.l0_byte(nest, spu)` returns an `xp.ndarray[uint8]` of shape `(GTX_L0_SIZE_BYTES,)`
- `mem.l1_byte(nest, spu)` returns an `xp.ndarray[uint8]` of shape `(GTX_L1_SIZE_BYTES,)`
- `mem.l2_byte(nest)` returns an `xp.ndarray[uint8]` of shape `(GTX_L2_SIZE_BYTES,)`
- `DDR_MEMORY.raw()` returns an `xp.ndarray[uint8]` (resident on xp's default device)
- `RegisterFile.read(addr)` returns a Python `int` (xp scalar `.item()` boundary)
- `RegisterFile.write(addr, value)` accepts Python `int` and stores via `xp.int64` cast
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port unit/memory.py — scratchpads + DDR_MEMORY to xp; to_host at file-I/O boundary</name>
  <files>src/main/python/riscv/gtx/unit/memory.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/memory.py (full file — torch sites at lines 6, 16, 22, 48-56, 79, 145, 172, 294, 318 per RESEARCH canonical_refs)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-09, D-10, D-12; D-10 verification checklist)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Code Example 2; Pitfall 5 — DDR VRAM budget; Pitfall 7 — torch.frombuffer)
    - src/main/python/riscv/gtx/config_params.py (post-Wave-0 — `xp`, `to_host`, `to_device`)
  </read_first>
  <behavior>
    - Test 1 `test_memory_layout`: writing `0x3C00` to `mem.l1_byte(0,0)[off]` produces bytes `[0x00, 0x3C]` (LE); `mem.l1_f16(0,0)[off//2]` reads back as `xp.float16(1.0)`.
    - Test 2 `test_ddr_grow`: `DDR_MEMORY.ensure(end_offset)` doubling-grow preserves prior bytes after growing past initial floor.
    - Test 3 `test_ddr_save_to_hex_xp_aware`: `ddr_save_to_hex` produces identical bytes whether xp=numpy or xp=cupy (to_host bridge at file boundary).
    - Test 4 `test_module_level_no_torch`: `grep -c "torch" src/main/python/riscv/gtx/unit/memory.py` returns 0.
    - Smoke: ABS strict still PASS (correctness invariant preserved).
  </behavior>
  <action>
    Mechanical 1:1 port. Concrete edits at each known site:

    **Line 6** — Replace `import torch` with:
    ```python
    from ..config_params import xp, to_host, to_device
    ```

    **Line 16 area** — If there's a `_DEVICE = DEVICE` import-time alias, delete it. The xp is the new device implicit.

    **Line 22 / 48-56 area** — module-level scratchpad allocations. Replace each:
    ```python
    # BEFORE:
    _L2_GLOBAL = torch.zeros((GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=torch.uint8, device=DEVICE)
    _L1_GLOBAL = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=torch.uint8, device=DEVICE)
    _L0_GLOBAL = torch.zeros((GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=torch.uint8, device=DEVICE)
    # AFTER:
    _L2_GLOBAL = xp.zeros((GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=xp.uint8)
    _L1_GLOBAL = xp.zeros((GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=xp.uint8)
    _L0_GLOBAL = xp.zeros((GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=xp.uint8)
    ```

    **Line 79 area** — `class DDR_MEMORY` init / `_DDR_DEVICE = torch.device("cpu")` constant:

    Delete the `_DDR_DEVICE` constant. In `DDR_MEMORY.__init__`, replace:
    ```python
    # BEFORE:
    self._bytes = torch.zeros(size, dtype=torch.uint8, device=_DDR_DEVICE)
    # AFTER (D-10: DDR follows xp; comment notes VRAM-budget warning):
    # D-10: DDR follows xp. On consumer GPUs (<12 GB VRAM), set `GTX_DDR_SIZE=1G`
    # via env var to leave headroom for scratchpads (~25 MB) + CUDA context overhead.
    # See README "GPU memory budget" section.
    self._bytes = xp.zeros(size, dtype=xp.uint8)
    ```

    **Line 145 area** — `ensure()` doubling-grow:
    ```python
    # BEFORE:
    new_arr = torch.zeros(new_size, dtype=torch.uint8, device=_DDR_DEVICE)
    if self._bytes is not None:
        new_arr[:current_size] = self._bytes
    # AFTER:
    new_arr = xp.zeros(new_size, dtype=xp.uint8)
    if self._bytes is not None:
        new_arr[:current_size] = self._bytes  # xp slice-assign works on both numpy and cupy
    ```

    **Line 172 area** — `.view(torch.float16)` byte-reinterpret site:
    ```python
    # BEFORE: self._l1_f16_views = self.l1.view(torch.float16)
    # AFTER:  self._l1_f16_views = self.l1.view(xp.float16)
    ```
    Repeat for any `.view(torch.uint8)` / `.view(torch.uint16)` sites — replace with `xp.uint8` / `xp.uint16`. **DO NOT change `.view(N, M)` reshape sites in this file** (audit list says memory.py only has dtype views).

    **Line 294 area** — `torch.frombuffer(bytearray(...), dtype=torch.uint8)`:
    ```python
    import numpy as _np  # local at top of method, NOT module
    # frombuffer is a numpy-only path (works on cupy too — wraps host bytes).
    # File I/O is host-only by contract; use numpy then to_device if needed.
    arr_host = _np.frombuffer(bytearray(b), dtype=_np.uint8)
    return to_device(arr_host)  # numpy→cupy if xp=cupy, identity if xp=numpy
    ```
    Note: `_np` is needed because `xp.frombuffer` on cupy doesn't accept host bytes directly — the host stage must go through numpy.

    **Line 318 area** — `ddr_save_to_hex` `.detach().cpu().contiguous().numpy()` chain:
    ```python
    # BEFORE: arr_host = self._bytes[start:end].detach().cpu().contiguous().numpy()
    # AFTER:  arr_host = to_host(self._bytes[start:end])  # no-op on numpy, cp.asnumpy on cupy
    # Then: bytes(arr_host) or similar formatting path stays the same.
    ```

    Repeat for any other `.cpu()`/`.numpy()`/`.detach()` chains in this file (audit by grep).

    **DO NOT touch:** function signatures, public API names, `MEMORY` base-class structure, the GTX_L*_SIZE constants, the LE byte-order assumptions.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/memory.py && uv run pytest tests/gtx/test_memory_layout.py tests/gtx/test_dma_roundtrip.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/memory.py` returns 0.
    - `grep -c "from ..config_params import xp" src/main/python/riscv/gtx/unit/memory.py` returns at least 1.
    - `grep -c "_DDR_DEVICE" src/main/python/riscv/gtx/unit/memory.py` returns 0.
    - `grep -c "xp.zeros" src/main/python/riscv/gtx/unit/memory.py` returns at least 4 (3 scratchpads + DDR_MEMORY init + ensure grow).
    - `grep -c "to_host" src/main/python/riscv/gtx/unit/memory.py` returns at least 1 (ddr_save_to_hex boundary).
    - `grep -c "D-10" src/main/python/riscv/gtx/unit/memory.py` returns at least 1 (the VRAM-budget comment).
    - `uv run pytest tests/gtx/test_memory_layout.py -x --no-cov` exits 0.
    - `uv run pytest tests/gtx/test_dma_roundtrip.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>memory.py is torch-free, uses xp for all allocations and views, calls to_host at file I/O boundaries; existing memory/dma_roundtrip tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Port unit/register_file.py — SPR int64 storage to xp; update RegisterFile interface</name>
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
    - Existing `tests/gtx/test_csr_registry_chain.py` still passes (Task 4 will update it for dtype assertion changes).
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

    **Constructor signature** — If `__init__(self, shape, ..., device=DEVICE)` exists, remove the `device=` param. Update any internal callers; downstream Wave 1 Task 3 fixes the npu.py call site.

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
    - `uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov` exits 0 (note: may need Task 4 dtype-assertion updates).
    - `uv run python -c "from riscv.gtx.unit.register_file import RegisterFile; rf = RegisterFile((16,)); rf.write(5, 0xCAFE); assert rf.read(5) == 0xCAFE"` exits 0.
  </acceptance_criteria>
  <done>register_file.py allocates SPR storage on xp, no torch references, basic round-trip + bit-field operations preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Port npu.py — _mxe_accum / _credit_ld / _credit_st / RegisterFile instantiation to xp; replace .cpu() at line 354 with to_host()</name>
  <files>src/main/python/riscv/gtx/npu.py</files>
  <read_first>
    - src/main/python/riscv/gtx/npu.py (full file — torch sites at lines 12, 19, 94-106, 354 per RESEARCH canonical_refs)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (Wave 1 mapping for npu.py)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pattern 4 — FP32-accumulate; Pitfall 3 — atexit ordering)
    - src/main/python/riscv/gtx/unit/register_file.py (post-Task-2 — note constructor signature change: no `device=` kwarg)
    - src/main/python/riscv/gtx/unit/memory.py (post-Task-1 state)
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
    # AFTER (post-Task-2 RegisterFile has no device kwarg):
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
  <name>Task 4: Port tests/gtx/test_csr_registry_chain.py off torch.int64 dtype assertions</name>
  <files>tests/gtx/test_csr_registry_chain.py</files>
  <read_first>
    - tests/gtx/test_csr_registry_chain.py (full file — torch references per CONTEXT D-16)
    - src/main/python/riscv/gtx/unit/register_file.py (post-Task-2 state)
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
  <name>Task 5: Wave 1 gate — 6-op smoke + tile-2 + ABS perf measurement + VRAM/SPR exception decision</name>
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
    Commit: <sha after Task 4>

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
    - [x] memory.py torch-free, xp.zeros for scratchpads + DDR
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
1. memory.py + register_file.py + npu.py all import xp from config_params and have zero `import torch` references.
2. DDR_MEMORY uses xp.zeros for init and ensure() doubling-grow; file I/O routes through to_host.
3. RegisterFile SPR storage uses xp.zeros(..., dtype=xp.int64); no device= kwargs.
4. GtxNpu constructor instantiates _mxe_accum / _credit_ld / _credit_st via xp.
5. test_csr_registry_chain.py is torch-free with numpy-equivalent assertions.
6. Wave 1 gate doc shows: 6-op smoke PASS + tile-2 PASS + ABS walltime in 85-105s + D-10/D-11 verification recorded.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-01-SUMMARY.md`
</output>
