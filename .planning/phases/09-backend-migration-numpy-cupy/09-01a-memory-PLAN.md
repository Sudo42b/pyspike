---
phase: 09-backend-migration-numpy-cupy
plan: 01a
type: execute
wave: 2
# CONTEXT D-05 Wave 1 = plans 09-01a + 09-01b. This is part 1 of 2 (memory.py only).
# B-4 split: original 09-01 contained 5 tasks + 9 surgical edits; split for context budget.
depends_on:
  - "00"
files_modified:
  - src/main/python/riscv/gtx/unit/memory.py
autonomous: true
requirements:
  - BM-02
user_setup: []

must_haves:
  truths:
    - "`unit/memory.py` allocates `_L2_GLOBAL`, `_L1_GLOBAL`, `_L0_GLOBAL` via `xp.zeros(..., dtype=xp.uint8)` instead of `torch.zeros(..., device=DEVICE)`."
    - "`DDR_MEMORY._bytes` allocation routes through `xp.zeros` (D-10: GPU when xp=cupy)."
    - "`DDR_MEMORY.ensure()` doubling-grow uses `xp.zeros` + slice copy; data preserved across grow."
    - "`ddr_save_to_hex` and `ddr_load_from_hex` invoke `to_host()` before file I/O (no `.cpu()` calls)."
    - "`_DDR_DEVICE` literal removed; no `torch.device` reference remains anywhere in memory.py."
    - "VRAM budget gate (when xp=cupy): doc note added to memory.py that consumer GPU <12 GB should set `GTX_DDR_SIZE=1G`."
    - "Wave 1 partial gate (correctness): memory layout + dma roundtrip unit tests PASS."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/memory.py"
      provides: "Module-level scratchpad allocations + DDR_MEMORY using xp (numpy default, cupy when GTX_USE_CUDA=1)"
      contains: "from ..config_params import xp"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/memory.py"
      to: "src/main/python/riscv/gtx/config_params.py"
      via: "`from ..config_params import xp, to_host, to_device`"
      pattern: "from ..config_params import xp"
---

<objective>
Wave 1 (part a): Port the memory layer — `unit/memory.py` — from torch to xp. Apply D-10 (DDR-on-GPU when xp=cupy) with VRAM-budget documentation. Single-file scope keeps this plan small (~2-3 tasks ~50% context).

Purpose: All compute ops in Wave 2 will allocate temporaries via `xp` AND read/write to these scratchpads. The memory layer MUST be xp-resident first; otherwise Wave 2's `xp.matmul(L1_view_f16, L1_view_f16)` would mix backends.

Output: memory.py ported to xp; `_DDR_DEVICE` literal fully removed; ddr_save/load_from_hex use `to_host()` at file boundary; VRAM-budget comment added.
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
@CLAUDE.md

<interfaces>
<!-- xp/helpers interface (established by Wave 0). -->

From src/main/python/riscv/gtx/config_params.py:
```python
xp           # numpy module (default) or cupy module (GTX_USE_CUDA=1)
to_host      # callable: cupy→numpy bridge (identity on numpy path)
to_device    # callable: numpy→cupy bridge (identity on numpy path)
```

Wave 1a invariants for downstream waves:
- `mem.l0_byte(nest, spu)` returns an `xp.ndarray[uint8]` of shape `(GTX_L0_SIZE_BYTES,)`
- `mem.l1_byte(nest, spu)` returns an `xp.ndarray[uint8]` of shape `(GTX_L1_SIZE_BYTES,)`
- `mem.l2_byte(nest)` returns an `xp.ndarray[uint8]` of shape `(GTX_L2_SIZE_BYTES,)`
- `DDR_MEMORY.raw()` returns an `xp.ndarray[uint8]` (resident on xp's default device)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port unit/memory.py — scratchpads + DDR_MEMORY to xp; to_host at file-I/O boundary; remove _DDR_DEVICE</name>
  <files>src/main/python/riscv/gtx/unit/memory.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/memory.py (full file — torch sites at lines 6, 16, 22, 48-56, 79 (_DDR_DEVICE), 101, 145, 172, 294, 318 per RESEARCH canonical_refs)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-09, D-10, D-12; D-10 verification checklist)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Code Example 2; Pitfall 5 — DDR VRAM budget; Pitfall 7 — torch.frombuffer)
    - src/main/python/riscv/gtx/config_params.py (post-Wave-0 — `xp`, `to_host`, `to_device`)
  </read_first>
  <behavior>
    - Test 1 `test_memory_layout`: writing `0x3C00` to `mem.l1_byte(0,0)[off]` produces bytes `[0x00, 0x3C]` (LE); `mem.l1_f16(0,0)[off//2]` reads back as `xp.float16(1.0)`.
    - Test 2 `test_ddr_grow`: `DDR_MEMORY.ensure(end_offset)` doubling-grow preserves prior bytes after growing past initial floor.
    - Test 3 `test_ddr_save_to_hex_xp_aware`: `ddr_save_to_hex` produces identical bytes whether xp=numpy or xp=cupy (to_host bridge at file boundary).
    - Test 4 `test_module_level_no_torch`: `grep -c "torch" src/main/python/riscv/gtx/unit/memory.py` returns 0.
    - Test 5: No `_DDR_DEVICE` or `torch.device` literal anywhere in memory.py (H-5).
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

    **Line 79 + 101 + 145 — `_DDR_DEVICE` removal (H-5 completeness):**

    Delete the `_DDR_DEVICE = torch.device("cpu")` constant at line 79 entirely. In `DDR_MEMORY.__init__` (line ~101), replace:
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
    - **H-5 completeness**: `grep -rn "_DDR_DEVICE\|torch.device" src/main/python/riscv/gtx/unit/memory.py` returns 0 (no residual literals).
    - **H-5 audit**: `grep -rn "_DDR_DEVICE" src/main/python/riscv/gtx/` returns 0 (no other site references the removed constant).
    - `grep -c "xp.zeros" src/main/python/riscv/gtx/unit/memory.py` returns at least 4 (3 scratchpads + DDR_MEMORY init + ensure grow).
    - `grep -c "to_host" src/main/python/riscv/gtx/unit/memory.py` returns at least 1 (ddr_save_to_hex boundary).
    - `grep -c "D-10" src/main/python/riscv/gtx/unit/memory.py` returns at least 1 (the VRAM-budget comment).
    - `uv run pytest tests/gtx/test_memory_layout.py -x --no-cov` exits 0.
    - `uv run pytest tests/gtx/test_dma_roundtrip.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>memory.py is torch-free, uses xp for all allocations and views, calls to_host at file I/O boundaries, _DDR_DEVICE removed everywhere; existing memory/dma_roundtrip tests still pass.</done>
</task>

</tasks>

<verification>
- memory.py fully torch-free: `grep -rn "import torch\|torch\." src/main/python/riscv/gtx/unit/memory.py | wc -l` returns 0.
- `_DDR_DEVICE` literal fully removed across gtx/: `grep -rn "_DDR_DEVICE" src/main/python/riscv/gtx/ | wc -l` returns 0.
- BM-02 partial invariants: memory layout + dma roundtrip unit tests still PASS.
- Plan 09-01b (regs + gate) is the natural next step.
</verification>

<success_criteria>
1. memory.py imports xp from config_params and has zero `import torch` references.
2. DDR_MEMORY uses xp.zeros for init and ensure() doubling-grow; file I/O routes through to_host.
3. `_DDR_DEVICE = torch.device(...)` literal removed; no `torch.device` reference anywhere.
4. VRAM-budget D-10 comment added to memory.py near DDR_MEMORY.__init__.
5. memory_layout + dma_roundtrip unit tests still PASS.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-01a-SUMMARY.md`
</output>
</content>
</invoke>