---
phase: 09-backend-migration-numpy-cupy
plan: 02a
type: execute
wave: 3
depends_on:
  - 09-backend-migration-numpy-cupy/01-memory-regs
files_modified:
  - src/main/python/riscv/gtx/unit/ins/ops/spr.py
  - src/main/python/riscv/gtx/unit/ins/ops/mm.py
  - src/main/python/riscv/gtx/unit/ins/ops/vec.py
  - src/main/python/riscv/gtx/unit/ins/ops/act.py
  - src/main/python/riscv/gtx/unit/csr/register.py
autonomous: false
requirements:
  - BM-03
user_setup: []

must_haves:
  truths:
    - "`unit/ins/ops/spr.py` is torch-free; SPR write/read paths use xp scalars."
    - "`unit/ins/ops/mm.py` `gemm_core` + variant helpers use `xp.matmul` / `xp.dot` / `xp.sum` (BLAS dispatch identical to current torch path)."
    - "`unit/ins/ops/vec.py` `_apply_unary` + sasmd/dot/vsum/clamp/cumsum/arange use xp; FP32-internal-accumulate preserved (RESEARCH Pattern 4)."
    - "`unit/ins/ops/act.py` activations (relu, prelu, gelu, tanh, sigmoid, softmax, esum) use xp; FP8 conversion path follows locked strategy from 09-SCOPE-DECISION.md (LUT-only if option B chosen)."
    - "FP8 LUTs (`FP16_TO_FP8_LUT` + `FP8_TO_FP16_LUT`) preserved as import-time uint8/float16 arrays via xp."
    - "Wave-end gate: GELU + RELU + SIGMOID + TANH + SOFTMAX + ABS smoke + tile-2 all PASS; ABS perf in 85-105s."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/ins/ops/spr.py"
      provides: "SPR custom0/1 handlers using xp"
      contains: "from ....config_params import xp"
    - path: "src/main/python/riscv/gtx/unit/ins/ops/mm.py"
      provides: "gemm_core + MM variants on xp (BLAS-equivalent semantics)"
      contains: "xp.matmul"
    - path: "src/main/python/riscv/gtx/unit/ins/ops/vec.py"
      provides: "Vector unary/binary ops + _apply_unary dispatch on xp"
      contains: "xp.abs"
    - path: "src/main/python/riscv/gtx/unit/ins/ops/act.py"
      provides: "Activation functions + FP8 LUT cvt paths on xp"
      contains: "FP16_TO_FP8_LUT"
  key_links:
    - from: "src/main/python/riscv/gtx/unit/ins/ops/mm.py"
      to: "src/main/python/riscv/gtx/unit/memory.py"
      via: "operates on L1/L2 xp.ndarray views allocated by memory.py"
      pattern: "xp.matmul"
    - from: "src/main/python/riscv/gtx/unit/ins/ops/act.py"
      to: "src/main/python/riscv/gtx/unit/ins/ops/act.py"
      via: "import-time FP16_TO_FP8_LUT precompute (xp.uint8 indexing for fp8 cvt)"
      pattern: "FP16_TO_FP8_LUT\\[u16\\]"
---

<objective>
Wave 2a: Port the four op-handler modules (`spr.py`, `mm.py`, `vec.py`, `act.py`) plus the CSR `register.py` doc-string update from torch to xp. These modules implement the per-RoCC-instruction custom0/1 handler dispatch — they are the hot path called once per dispatched instruction. Activation kernels (`act.py`) carry the FP8 strategy locked in Wave 0 (`09-SCOPE-DECISION.md`).

Purpose: Wave 2b (engines) consumes the function-level results from these op handlers. Without the op layer ported, engines cannot transition off torch. The 6-op smoke set (ABS, GELU, RELU, SIGMOID, TANH, SOFTMAX) directly exercises vec + act paths — Wave 2a's gate proves the bit-exact invariant holds.

Output: 4 op modules + 1 CSR register module torch-free; FP8 LUT path verified; smoke set + tile-2 + ABS perf gate GREEN.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md
@.planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md
@.planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md
@.planning/phases/09-backend-migration-numpy-cupy/09-00-SUMMARY.md
@.planning/phases/09-backend-migration-numpy-cupy/09-01-SUMMARY.md
@src/main/python/riscv/gtx/unit/ins/ops/spr.py
@src/main/python/riscv/gtx/unit/ins/ops/mm.py
@src/main/python/riscv/gtx/unit/ins/ops/vec.py
@src/main/python/riscv/gtx/unit/ins/ops/act.py
@src/main/python/riscv/gtx/unit/csr/register.py
@CLAUDE.md

<interfaces>
<!-- Wave 1 + Wave 0 already established xp / to_host / to_device / memory layout. -->
<!-- Wave 2a depends on those. -->

Storage layer (post-Wave-1) returns xp.ndarray for all L0/L1/L2/SPR accesses.

Op API surface (preserved by Wave 2a — backwards-compatible with existing tests):
- `spr.handle_wrspr(...)` / `spr.handle_rdspr(...)` — funct7=0x00 dispatch helpers.
- `mm.gemm_core(A, B, has_bias, bias_fp32) -> xp.ndarray` — FP32-internal-accumulate.
- `mm.gemm_dot(A, B, prior_accum) -> xp.float32` — scalar.
- `mm.gemm_reduce_sum_a(A, prior_accum) -> xp.float32` — scalar.
- `vec._apply_unary(funct7, sub_op, view) -> xp.ndarray` — unified unary dispatch.
- `vec.sasmd_kernel`, `vec.dot_kernel`, `vec.vsum_kernel`, `vec.clamp_*` (signatures unchanged).
- `act.relu`, `act.prelu`, `act.gelu`, `act.tanh`, `act.sigmoid`, `act.softmax`, `act.esum` — fp16 in, fp16 out.
- `act.cvt_qh`, `act.cvt_hq` — FP16↔FP8 (strategy per 09-SCOPE-DECISION.md).
- `act.cvt_ih/hi/hn/sh/hs/dh/hd` — bit-exact int/float conversions on xp.

Established patterns to preserve verbatim:
- FP32-internal-accumulate: `arr.astype(xp.float32) → reduction → .astype(xp.float16)` chain.
- `xp.matmul(A_f32, B_f32).astype(xp.float16)` for gemm_core (BLAS dispatch; identical numerics to current torch.matmul path per RESEARCH Pitfall 2).
- LE byte-order for fp16 view-as-uint16 round trip.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port unit/ins/ops/spr.py — SPR custom0/1 handlers + WRSPR/RDSPR off torch</name>
  <files>src/main/python/riscv/gtx/unit/ins/ops/spr.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/ins/ops/spr.py (full file — single import at line 18 per RESEARCH canonical_refs; mostly integer arithmetic)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Architecture section spr.py entry — "1 torch ref, import only")
    - src/main/python/riscv/gtx/config_params.py (post-Wave-0)
  </read_first>
  <behavior>
    - Test 1 `test_spr_handlers_unchanged`: existing test_spr.py passes (WRSPR/RDSPR semantics preserved).
    - Test 2 no torch references.
    - Test 3 SPR address-field decoding unchanged.
  </behavior>
  <action>
    **Line 18** — Remove `import torch`. Replace with `from ....config_params import xp` ONLY IF xp is actually referenced in the body. Per RESEARCH "1 torch ref, import only" — likely the body doesn't use torch beyond legacy decoration. After removing the import, run pyflakes/grep to confirm no orphan torch references.

    If any torch type hint exists (e.g., `xs1: torch.Tensor`), replace with `xs1: int` (RoCC ISA gives xs1/xs2 as `reg_t` = Python int via pybind11 binding).

    Per CLAUDE.md surgical-changes rule: no other edits.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/ins/ops/spr.py && uv run pytest tests/gtx/test_spr.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/ins/ops/spr.py` returns 0.
    - `uv run pytest tests/gtx/test_spr.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>spr.py torch-free; SPR routing tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Port unit/ins/ops/mm.py — gemm_core + MM variants to xp (BLAS-equivalent)</name>
  <files>src/main/python/riscv/gtx/unit/ins/ops/mm.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/ins/ops/mm.py (full file — torch sites at lines 28, 79 per RESEARCH; ~43 torch refs total)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pattern 4 — FP32-internal-accumulate; Pitfall 2 — BLAS ordering)
    - src/main/python/riscv/gtx/config_params.py (post-Wave-0)
    - vendor/gtx_cpp_reference/gtx/gtx_npu.h (FP16↔FP32 RNE ground truth — referenced if numerics drift)
  </read_first>
  <behavior>
    - Test 1 `test_op_mm`: every MM variant (mm, mm_s, mm_o, mm_v, mm_t, mmc, mmc_s, mmc_o, mmc_v, mmc_t) produces bit-exact output vs NumPy FP32-internal `np.matmul + single np.float16 cast` oracle (preserved from P4).
    - Test 2 `test_mm_chain`: mm.s → mmc.s → mmc accumulator chain preserves mxe_accum FP32 state.
    - Test 3 ABS strict still PASS (mm path is not exercised by ABS, but smoke gate covers regression detection).
    - Test 4 no torch references.
  </behavior>
  <action>
    **Line 28** — Replace `import torch` with `from ....config_params import xp`.

    **Line 79 area** — `_as_f32` helper. Concrete port (RESEARCH Pattern 4 verbatim):
    ```python
    # BEFORE:
    def _as_f32(x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.float32:
            return x.contiguous()
        return x.to(torch.float32).contiguous()
    # AFTER:
    def _as_f32(x):
        if x.dtype == xp.float32:
            return xp.ascontiguousarray(x)
        return xp.ascontiguousarray(x.astype(xp.float32))
    ```

    **`torch.matmul` → `xp.matmul`** (43 references total — grep + replace each, mostly mechanical):

    | torch | xp |
    |-------|-----|
    | `torch.matmul(A, B)` | `xp.matmul(A, B)` |
    | `torch.dot(a, b)` | `xp.dot(a, b)` |
    | `torch.sum(arr)` | `xp.sum(arr)` |
    | `torch.cat([a, b])` | `xp.concatenate([a, b])` |
    | `torch.zeros(shape, dtype=torch.float32)` | `xp.zeros(shape, dtype=xp.float32)` |
    | `tensor.to(torch.float16)` | `arr.astype(xp.float16)` |
    | `tensor.to(torch.float32)` | `arr.astype(xp.float32)` |
    | `tensor.contiguous()` | `xp.ascontiguousarray(tensor)` |
    | `tensor.view(torch.float16)` | `tensor.view(xp.float16)` (byte-reinterpret, NumPy-compatible) |
    | `tensor.view(N, M)` | `tensor.reshape(N, M)` (RESEARCH Pitfall 1 — explicit reshape, not view) |
    | `torch.float32` (constant in dtype comparison) | `xp.float32` |
    | `torch.float16` (constant) | `xp.float16` |

    **Preserve FP32-accumulate discipline:**
    For `gemm_core`: keep the pattern `out_f32 = xp.matmul(A_f32, B_f32); return out_f32.astype(xp.float16)`. This matches current torch behavior (both BLAS-dispatched). Do NOT switch to an explicit 3-loop here — that would change ULP.

    **DO NOT touch:**
    - `firmware_mm_op` packed-rs1 decoding (CORE invariant).
    - `mxe_accum` slot indices (NEST/SPU geometry).
    - The 10 variant helper signatures.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/ins/ops/mm.py && uv run pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns 0.
    - `grep -c "xp.matmul\|xp.dot\|xp.sum" src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns at least 3.
    - `grep -c "from ....config_params import xp" src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns 1.
    - `grep -nE "\.view\([0-9A-Za-z_]+, ?[0-9A-Za-z_]+" src/main/python/riscv/gtx/unit/ins/ops/mm.py` returns 0 lines (no two-arg `.view(N, M)` calls — should be `.reshape`).
    - `uv run pytest tests/gtx/test_op_mm.py -x --no-cov` exits 0.
    - `uv run pytest tests/gtx/test_mm_chain.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>mm.py torch-free, gemm_core + all 10 MM variants produce bit-exact outputs vs NumPy FP32 oracle.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Port unit/ins/ops/vec.py — _apply_unary + sasmd/dot/vsum/clamp/cumsum/arange to xp</name>
  <files>src/main/python/riscv/gtx/unit/ins/ops/vec.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/ins/ops/vec.py (full file — torch sites at lines 20, 67-102 + ~51 torch refs total)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Code Example 3 — vec.py unary port verbatim; P7 kernel inventory section)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-15 unary/binary fusion mnemonic set preserved)
  </read_first>
  <behavior>
    - Test 1 `test_op_vec`: SASMD (add/sub/mul/div IS+VS), DOT, VSUM, CLAMP (min/max/arange/accum), L0/L1 paths — bit-exact vs verify_ref oracle.
    - Test 2 `test_vsum_fp32_accumulate`: `np.float16([1.0, 1e-4]*1000).sum()` via VSUM → FP32-internal-accumulate result (≈0.1), not FP16-truncated (`inf`).
    - Test 3 ABS smoke PASS (vec.abs is the kernel ABS exercises).
    - Test 4 no torch references.
  </behavior>
  <action>
    **Line 20** — Replace `import torch` with `from ....config_params import xp`.

    **Lines 67-102 area** — Core kernels. Port verbatim per RESEARCH Code Example 3:

    ```python
    # Mapping table (apply throughout file):
    # torch.abs(x)         → xp.abs(x)
    # torch.negative(x)    → xp.negative(x)
    # torch.sign(x)        → xp.sign(x)
    # torch.ceil(x)        → xp.ceil(x)
    # torch.trunc(x)       → xp.trunc(x)
    # torch.floor(x)       → xp.floor(x)
    # torch.round(x)       → xp.round(x)
    # torch.sqrt(x)        → xp.sqrt(x)
    # torch.exp(x)         → xp.exp(x)
    # torch.log(x)         → xp.log(x)
    # torch.where(c, a, b) → xp.where(c, a, b)
    # torch.maximum(a, b)  → xp.maximum(a, b)
    # torch.minimum(a, b)  → xp.minimum(a, b)
    # torch.clamp(x, a, b) → xp.clip(x, a, b)  (NumPy uses `clip`, cupy mirrors it)
    # torch.cumsum(x, dim=0) → xp.cumsum(x, axis=0)  (axis keyword, not dim)
    # torch.arange(start, stop, step) → xp.arange(start, stop, step)
    # torch.full_like(t, v) → xp.full_like(t, v)
    # torch.zeros_like(t)   → xp.zeros_like(t)
    # torch.finfo(torch.float32).tiny → xp.finfo(xp.float32).tiny
    # torch.float16(v)     → xp.float16(v)
    # tensor.clone()       → arr.copy()
    # tensor.to(torch.float32) → arr.astype(xp.float32)
    ```

    For `cumsum` specifically: the kwarg name differs (`dim` in torch → `axis` in numpy/cupy). All `torch.cumsum(x, dim=0)` → `xp.cumsum(x, axis=0)`.

    For `clamp` specifically: name differs (`clamp` in torch → `clip` in numpy/cupy). All `torch.clamp(x, min, max)` → `xp.clip(x, min, max)`.

    **`_apply_unary` function** — Port verbatim per RESEARCH Code Example 3 (lines 658-690 of RESEARCH). Reproduce that block exactly.

    **`VSUM` / `DOT` FP32-internal-accumulate:**
    ```python
    # VSUM (current):
    def vsum_kernel(view):
        f32 = view.astype(xp.float32)
        return xp.float16(xp.sum(f32))  # single FP16 cast at end

    # DOT (current):
    def dot_kernel(a, b):
        a32 = a.astype(xp.float32)
        b32 = b.astype(xp.float32)
        return xp.float16(xp.dot(a32, b32))  # scalar fp16
    ```
    Preserve this discipline exactly (VEC-02 P5 invariant).

    **DO NOT touch:**
    - `_VEC_UNARY_MNEMONICS` / `_VEC_BINARY_MNEMONICS` frozensets (D-15: fusion mnemonics).
    - SASMD funct7 dispatch table.
    - L0/L1 path selection logic.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/ins/ops/vec.py && uv run pytest tests/gtx/test_op_vec.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|torch\." src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns 0.
    - `grep -c "from ....config_params import xp" src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns 1.
    - `grep -c "xp.abs\|xp.exp\|xp.where\|xp.cumsum" src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns at least 4.
    - `grep -c "xp.clamp" src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns 0 (must use `xp.clip`).
    - `grep -c "torch.clamp\|\.clamp(" src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns 0.
    - `grep -nE ", *dim=" src/main/python/riscv/gtx/unit/ins/ops/vec.py` returns 0 (axis= used, no dim=).
    - `uv run pytest tests/gtx/test_op_vec.py -x --no-cov` exits 0.
  </acceptance_criteria>
  <done>vec.py torch-free; SASMD/DOT/VSUM/CLAMP all pass; FP32-accumulate discipline preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: Port unit/ins/ops/act.py — activations + FP8 LUT path (strategy per 09-SCOPE-DECISION.md)</name>
  <files>src/main/python/riscv/gtx/unit/ins/ops/act.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/ins/ops/act.py (full file — torch sites at lines 24-25, 45-181 + 79 torch refs total)
    - .planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md (Pitfall 4 — FP8 strategy options; P7 kernel inventory ACT row breakdown lines ~899-922)
    - .planning/phases/09-backend-migration-numpy-cupy/09-SCOPE-DECISION.md (locked FP8 strategy from Wave 0)
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (Phase boundary — FP8 strategy gate)
  </read_first>
  <behavior>
    - Test 1 `test_op_act`: every activation (RELU, PRELU, GELU, TANH, SIGMOID, SOFTMAX, ESUM) bit-exact vs verify_ref oracle; direction asymmetry preserved (forward overwrites ADDRR; reversed overwrites ADDRA).
    - Test 2 `test_op_format`: `cvt_ih/hi/hn/sh/hs/dh/hd` bit-exact for non-FP8 paths.
    - Test 3 (FP8 strategy-conditional):
      - If 09-SCOPE-DECISION.md selected option B (LUT-only): `cvt_qh(arr_f16, scale, offset)` and `cvt_hq(arr_f8, scale, offset)` round-trip bit-exact through `FP16_TO_FP8_LUT` / `FP8_TO_FP16_LUT`.
      - If option C (descope): `cvt_qh` / `cvt_hq` raise `NotImplementedError("FP8 deferred to v1.2")`.
      - If option A (ml_dtypes): test uses `ml_dtypes.float8_e4m3fn` (pyproject change handled in 09-03).
    - Test 4 ABS smoke + GELU/RELU/SIGMOID/TANH/SOFTMAX smoke all PASS.
    - Test 5 no torch references.
  </behavior>
  <action>
    **Lines 24-25** — Replace:
    ```python
    # BEFORE:
    import torch
    from torch import Tensor
    # AFTER:
    from ....config_params import xp
    ```
    Type hints `Tensor` → `xp.ndarray` (or just `Any` / remove — the runtime dispatch doesn't care).

    **Lines 45-117 area — FP8 LUT precompute (RESEARCH calls out these as already-existing):**
    ```python
    # FP16_TO_FP8_LUT precompute (uint8[65536]):
    # Currently:  FP16_TO_FP8_LUT = torch.tensor([...], dtype=torch.uint8)
    # After:      FP16_TO_FP8_LUT = xp.array([...], dtype=xp.uint8)

    # FP8_TO_FP16_LUT precompute (float16[256]):
    # Currently:  FP8_TO_FP16_LUT = torch.tensor([...], dtype=torch.float16)
    # After:      FP8_TO_FP16_LUT = xp.array([...], dtype=xp.float16)
    ```
    The LUT bit-pattern computation logic is unchanged — only the container type. If the precompute uses any `torch.from_numpy` bridge for boot, replace with `xp.asarray(np_intermediate)`.

    **Lines 123-144 area — cvt_qh / cvt_hq (FP8 paths):**

    Branch by `09-SCOPE-DECISION.md` FP8 strategy:

    **Option B (LUT-only — RECOMMENDED):**
    ```python
    def fp16_to_fp8_e4m3(t_fp16):
        # View fp16 as uint16, index into LUT.
        u16 = t_fp16.view(xp.uint16)
        return FP16_TO_FP8_LUT[u16]  # returns uint8

    def fp8_e4m3_to_fp16(t_e4m3):
        # uint8 index into FP8→FP16 LUT.
        return FP8_TO_FP16_LUT[t_e4m3]  # returns float16

    def cvt_qh(arr_f16, scale, offset):
        # Scale + offset applied in fp32 to preserve precision.
        f32 = arr_f16.astype(xp.float32)
        f32 = (f32 + xp.float32(offset)) * xp.float32(scale)
        return fp16_to_fp8_e4m3(f32.astype(xp.float16))

    def cvt_hq(arr_f8, scale, offset):
        f16 = fp8_e4m3_to_fp16(arr_f8)
        f32 = f16.astype(xp.float32)
        f32 = f32 / xp.float32(scale) - xp.float32(offset)
        return f32.astype(xp.float16)
    ```
    Replace existing `tensor.to(torch.float8_e4m3fn)` calls with the LUT-indexed pattern.

    **Option C (descope):**
    ```python
    def fp16_to_fp8_e4m3(t_fp16):
        raise NotImplementedError("FP8 e4m3 deferred to v1.2 (09-SCOPE-DECISION.md)")
    # Same for fp8_e4m3_to_fp16, cvt_qh, cvt_hq.
    ```

    **Option A (ml_dtypes):**
    ```python
    import ml_dtypes  # NEW dep, added to pyproject.toml in 09-03
    def fp16_to_fp8_e4m3(t_fp16):
        return t_fp16.astype(ml_dtypes.float8_e4m3fn)
    ```

    **Lines 147-187 area — Non-FP8 cvt functions** (cvt_ih/hi/hn/sh/hs/dh/hd):

    Mechanical 1:1 port (same mapping table as Task 3 vec.py):
    ```python
    # torch.float32 → xp.float32, torch.float16 → xp.float16, etc.
    # tensor.to(torch.float32) → arr.astype(xp.float32)
    ```

    **Lines 194-220 area — Activations:**

    Apply the mapping table:
    | torch | xp |
    |-------|----|
    | `torch.relu(x)` | `xp.maximum(x, xp.float16(0.0))` (numpy has no `relu`) |
    | `torch.where(x > 0, x, slope * x)` | `xp.where(x > 0, x, slope * x)` |
    | `torch.tanh(x)` | `xp.tanh(x)` |
    | `torch.sigmoid(x)` | `1 / (1 + xp.exp(-x))` (numpy has no `sigmoid`; explicit form) |
    | `torch.gelu(x)` | explicit formula: `0.5 * x * (1 + xp.tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` (approximate GELU, matches torch default) |
    | `torch.softmax(x)` | `xp.exp(x - x.max()) / xp.exp(x - x.max()).sum()` (stable softmax) |

    For GELU specifically: torch.gelu default is the approximate formula above. Verify the constant `0.044715` matches what the verify_ref oracle expects. If torch.gelu was using the EXACT erf-based formula, switch to `0.5 * x * (1 + erf(x / sqrt(2)))` using `xp.special.erf` (numpy: scipy.special.erf — but that's an external dep). For numpy 2.x: use the existing approximate formula (matches vendor C++ which uses tanh-approx per RESEARCH).

    **DO NOT touch:**
    - Activation direction asymmetry table (forward overwrites ADDRR; reversed overwrites ADDRA).
    - `firmware_act` packed-rs1 decoding (ACT invariant).
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch" src/main/python/riscv/gtx/unit/ins/ops/act.py && uv run pytest tests/gtx/test_op_act.py tests/gtx/test_op_format.py -x --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "import torch\|from torch\|torch\." src/main/python/riscv/gtx/unit/ins/ops/act.py` returns 0.
    - `grep -c "from ....config_params import xp" src/main/python/riscv/gtx/unit/ins/ops/act.py` returns 1.
    - `grep -c "FP16_TO_FP8_LUT\|FP8_TO_FP16_LUT" src/main/python/riscv/gtx/unit/ins/ops/act.py` returns at least 2 (LUTs preserved).
    - `grep -c "float8_e4m3fn" src/main/python/riscv/gtx/unit/ins/ops/act.py` returns 0 if option B/C selected; 1+ if option A.
    - `uv run pytest tests/gtx/test_op_act.py -x --no-cov` exits 0.
    - `uv run pytest tests/gtx/test_op_format.py -x --no-cov` exits 0 (or skip cvt_qh/cvt_hq if option C).
  </acceptance_criteria>
  <done>act.py torch-free; FP8 strategy applied per 09-SCOPE-DECISION.md; all activation/cvt tests pass.</done>
</task>

<task type="auto">
  <name>Task 5: Update unit/csr/register.py docstring (torch.Tensor → xp.ndarray reference)</name>
  <files>src/main/python/riscv/gtx/unit/csr/register.py</files>
  <read_first>
    - src/main/python/riscv/gtx/unit/csr/register.py (line 95 area — RESEARCH calls this out as "RegisterFile bit-field torch.Tensor documentation")
    - src/main/python/riscv/gtx/unit/register_file.py (post-Task-2 — actual implementation now uses xp)
  </read_first>
  <action>
    Find line 95 area docstring/comment mentioning `torch.Tensor` in context of RegisterFile bit-field storage. Replace with `xp.ndarray (numpy.ndarray by default, cupy.ndarray when GTX_USE_CUDA=1)`.

    Do NOT change code semantics — this is a docstring-only update. Per CLAUDE.md surgical-changes rule.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && grep -c "torch.Tensor" src/main/python/riscv/gtx/unit/csr/register.py</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "torch.Tensor" src/main/python/riscv/gtx/unit/csr/register.py` returns 0.
    - `grep -c "xp.ndarray" src/main/python/riscv/gtx/unit/csr/register.py` returns at least 1.
    - `uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov` exits 0 (no semantic change so should still pass).
  </acceptance_criteria>
  <done>register.py docstring reflects xp-based storage.</done>
</task>

<task type="auto">
  <name>Task 6: Wave 2a gate — 6-op smoke + tile-2 + ABS perf measurement</name>
  <files>.planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md</files>
  <read_first>
    - .planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md (D-07, D-08)
    - .planning/phases/09-backend-migration-numpy-cupy/09-01-WAVE-GATE.md (Wave 1 baseline)
  </read_first>
  <action>
    Run gate commands and record to `.planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md`:

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
    # Wave 2a Gate Results

    Date: <YYYY-MM-DD>
    Commit: <sha after Task 5>

    ## Smoke Set
    Result: <PASS | FAIL>
    Notes: GELU/RELU/SIGMOID/TANH/SOFTMAX directly exercise the freshly ported act.py kernels.

    ## Tile-2
    Result: <PASS | FAIL>

    ## ABS Strict Walltime (D-08: 85-105s)
    Wall: <X.XXs>
    Wave 1 baseline: <from 09-01-WAVE-GATE.md>
    In-budget: <YES | NO>

    ## Wave 2a Sign-Off
    - [x] spr.py + mm.py + vec.py + act.py + csr/register.py torch-free
    - [x] FP8 strategy applied per 09-SCOPE-DECISION.md
    - [x] gemm_core BLAS-equivalent semantics preserved
    - [x] FP32-internal-accumulate discipline preserved (VSUM, DOT, gemm_*)
    - [x] Activation direction asymmetry preserved
    - [x] Smoke + tile-2 + ABS perf gates GREEN
    ```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike && uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v && uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v</automated>
  </verify>
  <acceptance_criteria>
    - `.planning/phases/09-backend-migration-numpy-cupy/09-02a-WAVE-GATE.md` exists.
    - Both smoke and tile-2 sections show `Result: PASS`.
    - ABS walltime within 85-105s or documented justification.
    - Wave 2b unblocked (engines port can begin).
  </acceptance_criteria>
  <done>Wave 2a gate doc records GREEN gates; Wave 2b entry ready.</done>
</task>

</tasks>

<verification>
- Op modules torch-free: `grep -rn "import torch\|torch\." src/main/python/riscv/gtx/unit/ins/ops/ src/main/python/riscv/gtx/unit/csr/register.py | wc -l` returns 0.
- Bit-exact preserved: all 6 smoke ops + MM/VEC/ACT unit tests pass.
- FP8 strategy committed and applied uniformly.
</verification>

<success_criteria>
1. spr.py / mm.py / vec.py / act.py / csr/register.py all import xp and have no torch references.
2. FP8 path follows the strategy locked in 09-SCOPE-DECISION.md (LUT-only / descope / ml_dtypes).
3. gemm_core preserves BLAS dispatch (xp.matmul) and FP32-internal-accumulate.
4. VEC unit tests pass — SASMD/DOT/VSUM/CLAMP/cumsum bit-exact.
5. ACT unit tests pass — all 7 activations + cvt functions bit-exact (direction asymmetry preserved).
6. Wave 2a gate doc records PASS for smoke + tile-2 + ABS walltime in 85-105s.
</success_criteria>

<output>
After completion, create `.planning/phases/09-backend-migration-numpy-cupy/09-02a-SUMMARY.md`
</output>
