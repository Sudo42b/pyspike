---
phase: 09-backend-migration-numpy-cupy
plan: 02b
type: execute
wave: 4
depends_on:
  - 09-backend-migration-numpy-cupy/02a-ops
files_modified:
  - src/main/python/riscv/gtx/unit/context/dma_engine.py
  - src/main/python/riscv/gtx/unit/context/mm_engine.py
  - src/main/python/riscv/gtx/unit/context/vec_engine.py
  - src/main/python/riscv/gtx/unit/context/act_engine.py
autonomous: false
requirements:
  - BM-03
user_setup: []

must_haves:
  truths:
    - "`dma_engine.py` is torch-free; `.cpu()` at line 682 replaced with `to_host()`."
    - "`dma_engine.py` `.view(H, L)` reshape sites at lines 438, 492, 547 chain `.view(dtype).reshape(H, L)` per RESEARCH Pitfall 1."
    - "`mm_engine.py`, `vec_engine.py`, `act_engine.py` are torch-free; all temporaries use xp."
    - "`.permute(...)` → `.transpose(...)`, `torch.cat(...)` → `xp.concatenate(...)`, `.contiguous()` → `xp.ascontiguousarray(...)` consistently."
    - "Multi-tile DMA invariant preserved (P8 MTDMA-03 tile-2 unit test stays GREEN)."
    - "Wave-end gate: 6-op smoke + tile-2 + ABS perf in 85-105s all PASS."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/context/dma_engine.py"
      provides: "DMA orchestration on xp; file-I/O boundary uses to_host"
      contains: "from ....config_params import xp"
    - path: "src/main/python/riscv/gtx/unit/context/mm_engine.py"
      provides: "MM dispatch + variant routing on xp"
      contains: "xp"
    - path: "src/main/python/riscv/gtx/unit/context/vec_engine.py"
      provides: "VEC engine dispatch on xp"
      contains: "xp"
    - path: "src/main/python/riscv/gtx/unit/context/act_engine.py"
      provides: "ACT engine dispatch on xp"
      contains: "xp"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/context/dma_engine.py"
      to: "src/main/python/riscv/gtx/unit/memory.py"
      via: "DDR-L2-L1 byte ops on xp.ndarray views"
      pattern: "xp.copyto|to_host"
    - from: "src/main/python/riscv/gtx/unit/context/mm_engine.py"
      to: "src/main/python/riscv/gtx/unit/ins/ops/mm.py"
      via: "calls gemm_core / variant helpers (xp-resident)"
      pattern: "from ..ins.ops.mm import"
---

<objective>
Wave 2b: Port the four compute-engine modules (`dma_engine.py`, `mm_engine.py`, `vec_engine.py`, `act_engine.py`) from torch to xp. These engines orchestrate per-dispatch state machines that call into the Wave 2a-ported op handlers. The DMA engine owns the cross-tile DMA path that Phase 8 just stabilized — it is the highest-byte-exact-risk module in Wave 2.

Purpose: Engines hold cross-instruction state (deferred queues, dispatch context) and bridge dispatch to op layer. Without engine port, the kernel is fragmented across backends and the smoke set will not pass. The `.view(N, M)` reshape gotcha (RESEARCH Pitfall 1) lives concentrated in dma_engine.

Output: 4 engine modules torch-free; cross-tile DMA invariant preserved (P8 MTDMA-03 tile-2 unit test continues to PASS); 6-op smoke set GREEN.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md
@.planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md
@.planning/phases/09-backend-migration-numpy-cupy/09-02a-SUMMARY.md
@.planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md
@src/main/python/riscv/gtx/unit/context/dma_engine.py
@CLAUDE.md

<interfaces>
<!-- Wave 2a + Wave 1 + Wave 0 established the foundation. -->

Storage layer: `mem.l0/l1/l2(...)` returns xp.ndarray.
Op API: gemm_core / _apply_unary / activations all xp-resident with identical signatures.

Engines preserved invariants (P8):
- Deferred-store queue ordering at end_p (Phase 3 invariant).
- Multi-tile DMA orchestration (P8 MTDMA-01).
- `__split` / `__start_plan` / `__start_thread` / `__credit_chk` state reset at tile boundaries (P8 MTDMA-04).
- BE-LE byte-order under GTX_DDR_REVERSED=1 (P8 D-08).

Engine API (preserved):
- `dma_engine.execute_dma_2d(npu, insn, xs1, xs2)` — signature unchanged.
- `mm_engine.dispatch_mm(npu, insn, xs1, xs2, variant)` — signature unchanged.
- `vec_engine.dispatch_vec(npu, insn, xs1, xs2)` — signature unchanged.
- `act_engine.dispatch_act(npu, insn, xs1, xs2)` — signature unchanged.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port unit/context/dma_engine.py — replace .cpu() at line 682, fix .view(H, L) reshape at 438/492/547</name>
  <files>src/main/python/riscv/gtx/unit/context/dma_engine.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/context/dma_engine.py (full file — torch sites at lines 21, 267, 438, 492, 547, 682 per RESEARCH canonical_refs + Pitfall 1 audit)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pitfall 1 — `.view(N, M)` is reshape in torch but invalid in numpy; site list lines 426-431)
    - .planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md (P8 multi-tile DMA invariant — must not regress)
    - tests/gtx/test_multi_tile_dma.py (tile-2 unit test — MUST stay GREEN through this port)
  </read_first>
  <behavior>
    - Test 1 `test_dma_roundtrip`: DDR-L2-L1-L0 byte preservation (P3 invariant).
    - Test 2 `test_ddr_modes`: GTX_DDR_REVERSED=1 vs default LTR byte ordering.
    - Test 3 `test_multi_tile_dma`: P8 tile-2 unit test PASS.
    - Test 4 `test_deferred_store`: deferred-store flush ordering preserved at end_p.
    - Test 5 ABS smoke (96-tile vendor sweep) PASS.
    - Test 6 no torch references.
  </behavior>
  <action>
    **Line 21** — Replace `import torch` with `from ....config_params import xp, to_host, to_device`.

    **Line 267** — Already uses `.view(torch.float16).reshape(...)` per RESEARCH (correct chain — only dtype name change):
    ```python
    # BEFORE: arr.view(torch.float16).reshape(...)
    # AFTER:  arr.view(xp.float16).reshape(...)
    ```

    **Lines 438, 492, 547 — `.view(height, length)` reshape sites (RESEARCH Pitfall 1):**

    These currently use torch's `.view(N, M)` which is reshape in torch. NumPy `.view(N, M)` is invalid (raises in NumPy 2.x). Replace each:
    ```python
    # BEFORE: tensor.view(height, length)
    # AFTER:  arr.reshape(height, length)
    ```

    Be precise. Only replace `.view(N, M)` where N and M are int shape args. DO NOT replace `.view(torch.float16)` style dtype-only views — those become `.view(xp.float16)`.

    **Line 682** — `.cpu()` call:
    ```python
    # BEFORE: arr.detach().cpu()  (or similar chain)
    # AFTER:  to_host(arr)
    ```

    **All other torch ops in this file** — apply the standard mapping table:

    | torch | xp |
    |-------|----|
    | `torch.cat([a, b])` | `xp.concatenate([a, b])` |
    | `tensor.contiguous()` | `xp.ascontiguousarray(tensor)` |
    | `tensor.permute(2, 1, 0)` | `tensor.transpose(2, 1, 0)` |
    | `torch.frombuffer(bytearray, dtype=torch.uint8)` | `np.frombuffer(bytearray, dtype=np.uint8)` then `to_device(...)` if needed |
    | `tensor.copy_(src)` | `xp.copyto(tensor, src)` |
    | `tensor.zero_()` | `tensor.fill(0)` |
    | `tensor.to(torch.float32)` | `arr.astype(xp.float32)` |
    | `torch.zeros(shape, dtype=torch.X)` | `xp.zeros(shape, dtype=xp.X)` |
    | `torch.float16` | `xp.float16` |
    | `torch.uint8` | `xp.uint8` |

    **Multi-tile DMA preservation:** The cross-tile state-reset logic added in Phase 8 (MTDMA-04) lives in this file. DO NOT modify any of:
    - `__split` / `__start_plan` / `__start_thread` / `__credit_chk` handlers.
    - `MAX_SHARED_DMA_BYTES=65535` tile boundary logic.
    - Deferred-store queue management.
    - The vendor-derived tile loop ported in 08-04.

    Only swap torch types for xp equivalents. Same algorithm, same control flow.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/context/dma_engine.py && uv run pytest tests/gtx/test_dma_roundtrip.py tests/gtx/test_multi_tile_dma.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/context/dma_engine.py` returns 0.
    - `grep -c "from ....config_params import xp" src/main/python/riscv/gtx/unit/context/dma_engine.py` returns 1.
    - `grep -nE "\.view\([0-9A-Za-z_]+, ?[0-9A-Za-z_]+\)" src/main/python/riscv/gtx/unit/context/dma_engine.py` returns 0 (no two-arg `.view(N, M)` reshape sites).
    - `grep -c "to_host" src/main/python/riscv/gtx/unit/context/dma_engine.py` returns at least 1.
    - `uv run pytest tests/gtx/test_dma_roundtrip.py -x --no-cov` exits 0.
    - `uv run pytest tests/gtx/test_multi_tile_dma.py -x --no-cov` exits 0 (P8 tile-2 invariant preserved).
    - `uv run pytest tests/gtx/test_ddr_modes.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>dma_engine.py torch-free, .view(N, M) reshape sites fixed, P3/P8 invariants preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Port unit/context/mm_engine.py, vec_engine.py, act_engine.py to xp</name>
  <files>src/main/python/riscv/gtx/unit/context/mm_engine.py, src/main/python/riscv/gtx/unit/context/vec_engine.py, src/main/python/riscv/gtx/unit/context/act_engine.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/context/mm_engine.py (full file)
    - src/main/python/riscv/gtx/unit/context/vec_engine.py (full file)
    - src/main/python/riscv/gtx/unit/context/act_engine.py (full file)
    - src/main/python/riscv/gtx/unit/ins/ops/mm.py (post-Wave-2a — engine calls into these)
    - src/main/python/riscv/gtx/unit/ins/ops/vec.py (post-Wave-2a)
    - src/main/python/riscv/gtx/unit/ins/ops/act.py (post-Wave-2a)
  </read_first>
  <behavior>
    - Test 1 `test_op_mm` (full sweep) PASS.
    - Test 2 `test_op_vec` PASS.
    - Test 3 `test_op_act` PASS.
    - Test 4 ABS smoke + GELU/RELU/SIGMOID/TANH/SOFTMAX PASS.
    - Test 5 grep shows 0 `import torch` in each of the 3 engine files.
  </behavior>
  <action>
    Apply the same mapping table as Task 1 to all three engine files. Concrete steps for each:

    **For each of mm_engine.py / vec_engine.py / act_engine.py:**

    1. Top of file: `import torch` → `from ....config_params import xp` (adjust relative-import depth: from `unit/context/X_engine.py` the path to `config_params.py` is `....config_params`).

    2. Apply standard mapping table:

    | torch | xp |
    |-------|----|
    | `torch.zeros(shape, dtype=torch.X)` | `xp.zeros(shape, dtype=xp.X)` |
    | `torch.float16` / `torch.float32` / `torch.uint8` / `torch.int32` / `torch.int64` | `xp.float16` / `xp.float32` / `xp.uint8` / `xp.int32` / `xp.int64` |
    | `torch.matmul(A, B)` | `xp.matmul(A, B)` |
    | `torch.dot(a, b)` | `xp.dot(a, b)` |
    | `torch.sum(arr)` | `xp.sum(arr)` |
    | `torch.cat(...)` | `xp.concatenate(...)` |
    | `tensor.to(torch.X)` | `arr.astype(xp.X)` |
    | `tensor.contiguous()` | `xp.ascontiguousarray(tensor)` |
    | `tensor.permute(...)` | `tensor.transpose(...)` |
    | `tensor.view(torch.X)` | `tensor.view(xp.X)` |
    | `tensor.view(N, M)` | `tensor.reshape(N, M)` (RESEARCH Pitfall 1) |
    | `tensor.copy_(src)` | `xp.copyto(tensor, src)` |
    | `tensor.clone()` | `arr.copy()` |
    | `torch.where(c, a, b)` | `xp.where(c, a, b)` |

    3. Drop any `device=` kwargs from allocation calls.

    4. Drop any `.detach()` chains (numpy/cupy have no autograd).

    5. If a file imports op helpers (e.g., `from ..ins.ops.mm import gemm_core`), keep those imports — Wave 2a already ported them.

    **DO NOT touch:**
    - Engine state machines (dispatch context, NEST/SPU loop counters).
    - mxe_accum interaction (just verify slot indexing unchanged).
    - Public engine API entry points.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/context/mm_engine.py src/main/python/riscv/gtx/unit/context/vec_engine.py src/main/python/riscv/gtx/unit/context/act_engine.py && uv run pytest tests/gtx/test_op_mm.py tests/gtx/test_op_vec.py tests/gtx/test_op_act.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/context/mm_engine.py` returns 0.
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/context/vec_engine.py` returns 0.
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/context/act_engine.py` returns 0.
    - Each file contains `from ....config_params import xp` (1 occurrence per file).
    - `uv run pytest tests/gtx/test_op_mm.py tests/gtx/test_op_vec.py tests/gtx/test_op_act.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>3 engine modules torch-free; MM/VEC/ACT op tests pass.</done>
</task>

<task type="auto">
  <name>Task 3: Wave 2b gate — 6-op smoke + tile-2 + ABS perf measurement</name>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-02b-WAVE-GATE.md</files>
  <read_first>
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-07, D-08)
    - .planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md (Wave 2a baseline)
  </read_first>
  <action>
    Run gate commands and write `.planning/phases/09-backend-migration-numpy-cupy/09-02b-WAVE-GATE.md`:

    1. Smoke set:
       ```bash
       uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v
       ```
    2. Tile-2:
       ```bash
       uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v
       ```
    3. ABS strict walltime:
       ```bash
       /usr/bin/time -f "%e" uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov 2>&1 | tail -3
       ```

    Document format:
    ```markdown
    # Wave 2b Gate Results

    Date: <YYYY-MM-DD>
    Commit: <sha after Task 2>

    ## Smoke Set
    Result: <PASS | FAIL>

    ## Tile-2
    Result: <PASS | FAIL>
    Notes: dma_engine port preserves P8 MTDMA-03 invariant.

    ## ABS Strict Walltime (D-08: 85-105s)
    Wall: <X.XXs>
    Wave 2a baseline: <from 09-02a-WAVE-GATE.md>
    In-budget: <YES | NO>

    ## Wave 2b Sign-Off
    - [x] dma_engine.py torch-free, .view(N,M) sites fixed, to_host applied
    - [x] mm_engine.py / vec_engine.py / act_engine.py torch-free
    - [x] P3 deferred-store invariant preserved
    - [x] P8 multi-tile DMA invariant preserved
    - [x] Smoke + tile-2 + ABS perf gates GREEN
    ```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v && uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `.planning/phases/09-backend-migration-numpy-cupy/09-02b-WAVE-GATE.md` exists.
    - Smoke and tile-2 sections show `Result: PASS`.
    - ABS walltime in 85-105s or documented justification.
    - Wave 3 unblocked.
  </acceptance_criteria>
  <done>Wave 2b gate doc records GREEN; Wave 3 finalize ready.</done>
</task>

</tasks>

<verification>
- All engines torch-free: `grep -rn "import torch\|torch\." src/main/python/riscv/gtx/unit/context/*.py | wc -l` returns 0.
- P8 multi-tile DMA invariant preserved (tile-2 unit test still PASSes).
- ABS perf in 85-105s window.
</verification>

<success_criteria>
1. dma_engine.py / mm_engine.py / vec_engine.py / act_engine.py all import xp from config_params and have zero `import torch` references.
2. `.view(N, M)` reshape sites at dma_engine.py lines 438/492/547 are `.reshape(N, M)`.
3. `.cpu()` at dma_engine.py line 682 is `to_host(...)`.
4. tile-2 unit test PASS (P8 MTDMA-03 invariant).
5. 6-op smoke set PASS.
6. ABS strict walltime in 85-105s.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-02b-SUMMARY.md`
</output>
