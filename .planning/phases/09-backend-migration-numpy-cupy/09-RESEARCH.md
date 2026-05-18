# Phase 9: Backend Migration — PyTorch → NumPy + CuPy opt-in - Research

**Researched:** 2026-05-18
**Domain:** Array-library migration (torch.Tensor → numpy.ndarray + cupy.ndarray dual-backend via `xp` alias) + numba `@njit` revival as `guvectorize`-with-target dual-source
**Confidence:** HIGH (locked decisions in CONTEXT.md, existing torch surface fully audited in current src) / MEDIUM (numba `guvectorize(target='cuda')` + cupy interop — official docs say it works but project has no precedent) / LOW (28-kernel scope estimate — actual implementation cost depends on guvectorize signature match)

## Summary

Phase 9 replaces `torch.Tensor` with `numpy.ndarray` across all `src/main/python/riscv/gtx/*` modules. A new module-scope alias `xp` (resolved at import-time in `config_params.py`) selects `numpy` by default or `cupy` when `GTX_USE_CUDA=1` env-var is set AND `cupy` is importable (fail-loud RuntimeError otherwise, per D-03). PyTorch is removed completely from runtime + dev + test surface; CuPy ships as `pip install spike[cuda]` extras.

**The codebase audit surfaced four migration realities that override the CONTEXT's references to P7 layout:**
1. **P7's 28 `@njit` kernels no longer exist as decorated kernels.** Between commits `e74b3f0`, `e169af5`, `fca5117` the `gemm_core.py`/`vec_core.py`/`act_core.py` files (with `@njit` decorators) were **merged into op modules** (`unit/ins/ops/{mm,vec,act}.py`) **and rewritten as torch tensor ops** — `@njit` decorations were dropped. The 28-kernel count from CONTEXT still holds (3 mm + 7 vec + 18 act = 28), but they currently use `torch.matmul`/`torch.cumsum`/`torch.where`/etc., not numba njit. **D-13/D-14 must be re-scoped accordingly**: this is "first-time numba application to numpy ports", not "swap njit backend".
2. **`tests/gtx/conftest.py:18` currently fails collection with `pytest.exit` if CUDA unavailable** ("ORDER.md constraint — DDR은 CPU, 나머지 메모리 계층은 반드시 cuda"). This directly contradicts D-04 (DEVICE removal) — Wave 0 MUST port conftest before any Wave 1 work, or every test run on a no-GPU box breaks.
3. **`torch.float8_e4m3fn` is used in `unit/ins/ops/act.py:128,136`**. NumPy 2.x has **no native FP8 dtype**. The conversion functions `fp16_to_fp8_e4m3` / `cvt_qh` must either (a) pull `ml_dtypes` as a new dep (violates CLAUDE.md "no new runtime deps"), (b) port to uint8 bit-pattern manipulation using the LUTs that already exist (FP16_TO_FP8_LUT precomputed at import), or (c) descope FP8 in Wave 2 since vendor regression doesn't exercise it. Plan-stage must choose.
4. **The 28 "kernels" are not all pure-array — many are `(NDArray) → NDArray` but `torch.dot` / `torch.matmul` / `torch.cumsum` / `torch.where` /`torch.frombuffer` calls inside engines/dispatch are also part of the migration surface (242 `torch.*` references across 10 source files)**. The hot path Wave 1/2 ports are not just kernel renames.

**Primary recommendation:**

- Wave 0 scaffold: ① add `xp` alias + `to_host`/`to_device` helpers in `config_params.py` (D-01/D-12). ② port `tests/gtx/conftest.py` to make CUDA optional (so Wave 1 starts on a clean test gate). ③ Add a Wave-0 task to inventory and freeze the FP8 strategy (option a/b/c above). ④ Decide on numba scope (D-13 options A/B/C in CONTEXT).
- Wave 1 (memory + register_file): pure mechanical `torch.zeros(..., dtype=torch.uint8)` → `xp.zeros(..., dtype=xp.uint8)`. Keep DDR on host for now (defer D-10 GPU-DDR to a Wave 1 sub-task with explicit smoke), see "DDR-on-GPU concerns" below.
- Wave 2 (ops + engines): 28 kernels translated 1:1 with `torch.X` → `xp.X` mapping. FP8 strategy applied. ABS smoke gate at end.
- Wave 3 (tloop + verify + tests + pyproject + numba revival): D-13/D-14 numba layer landed on the **numpy** kernel signatures (after they exist as ndarray functions). Final torch removal.

The numba `guvectorize(target='cuda')` route is feasible per official docs, but the **cupy.ndarray ↔ numba `@cuda.jit` zero-copy path via `__cuda_array_interface__` is well-supported only for fp32/fp64/int**. Numba has a `cuda.fp16` module but **no native fp16 support in `@cuda.jit` kernel signatures** as of 2026 ([numba/numba#4402](https://github.com/numba/numba/issues/4402)). Since this codebase is fp16-heavy, the cuda-jit kernels likely need to upcast to fp32 inside the kernel (FP32-internal-accumulate pattern that the project already uses for correctness, so this is not a regression). Document in Wave 3 plan.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Area 1 — xp alias scaffold (D-01..D-04):**

- **D-01** xp alias lives in `config_params.py` (extends existing `DEVICE` SSOT). No separate `backend.py` module.
- **D-02** Backend resolve = import-time eager + frozen. `xp = numpy if not GTX_USE_CUDA else _import_cupy()` at module top-level.
- **D-03** `GTX_USE_CUDA=1` AND cupy missing → fail-loud `RuntimeError` with `pip install spike[cuda]` recovery hint. **Silent fallback explicitly forbidden** (260518-ffr regression precedent).
- **D-04** `DEVICE` symbol removed (no shim). `config_params.py:25` + `__init__.py:88` re-exports gone. External code importing `DEVICE` MUST get `ImportError`.

**Area 2 — Migration strategy (D-05..D-08):**

- **D-05** PR shape = 4 waves (0 = scaffold; 1 = memory + register_file; 2 = ops + engines; 3 = tloop + verify + tests + pyproject).
- **D-06** Dual-import allowed in waves 1/2, **must be fully removed by end of Wave 3**. Each wave end MUST keep ABS strict byte-exact GREEN.
- **D-07** Per-wave gate = 6 vendor ops (ABS + GELU + RELU + SIGMOID + TANH + SOFTMAX) + tile-2 unit test (`tests/gtx/test_multi_tile_dma.py`).
- **D-08** ABS strict perf budget = **±10% of 94.82s baseline (commit `2b0c66e`)** → 85–105s target window.

**Area 3 — CuPy placement (D-09..D-12):**

- **D-09** L0/L1/L2 scratchpads on GPU when xp=cupy. Module-level `_L2_GLOBAL`/`_L1_GLOBAL`/`_L0_GLOBAL` switched to `xp.zeros(...)`. ~25 MB per NEST-set on GPU.
- **D-10** **DDR on GPU as well (when xp=cupy)** — divergent from current explicit-CPU contract at `unit/memory.py:79 _DDR_DEVICE = torch.device("cpu")`. **Plan-stage MUST verify**: (1) 4 GiB default size vs consumer GPU VRAM (8 GB → 50% headroom for everything else); (2) `ddr_save_to_hex` / `ddr_load_from_hex` paths must add `xp.asnumpy()` before file I/O; (3) doubling-grow `ensure()` works on cupy. See "DDR-on-GPU concerns" below for detailed risks.
- **D-11** `RegisterFile` (SPR int64 storage) follows scratchpads' device. **Plan-stage MUST verify** by measuring ABS perf at end of Wave 1 — if dispatch-frequency SPR access causes 5x slowdown (like 260518-ffr), plan-stage takes an exception path (host-pinned SPR).
- **D-12** Cross-device transfer = two helpers `to_host(arr)` / `to_device(arr)` in `config_params.py`. When xp=numpy, both are no-ops returning `arr` as-is. When xp=cupy, `to_host = cp.asnumpy`, `to_device = cp.asarray`.

**Area 4 — Numba × xp (D-13..D-17):**

- **D-13** All 28 P7 njit kernels get dual-impl + numba CUDA backend. **Scope warning**: plan-stage MUST offer the 3 sub-options (A: P9=numpy + cupy raw, P10=cuda kernels / B: all in P9 / C: hot-path-only ~5–7 kernels with `cuda.jit`) to user for sign-off before Wave 3 begins. **Important context**: 28 njit kernels do NOT exist as `@njit` in current src (see Summary #1) — Phase 9 is reapplying numba layer to numpy ports.
- **D-14** Universal source = `numba.guvectorize` + target switching. **Caveat**: not all 28 kernels are guvectorize-shaped (state mutation, conditional return, scalar-out reductions). Plan-stage MUST audit and split into "convertible to guvectorize" vs "needs dual-source `@njit` + `@cuda.jit`".
- **D-15** `tloop_buffer._execute_fused` = 1:1 drop-in. `torch.abs` → `xp.abs`, `torch.float16` view → `xp.float16` view, `.copy_()` → `np.copyto(dst, src)` (NumPy 2.0+) or slice assignment.
- **D-16** Tests/gtx port = 3 files + conftest (54 torch refs total). conftest's CUDA gate must be reworked.
- **D-17** pyproject.toml — torch fully removed; `[project.optional-dependencies] cuda = ["cupy-cuda12x>=13.0"]`. `[cuda-jit]` extras separation is plan-stage discretion.

### Claude's Discretion

- `to_host()` / `to_device()` exact signature (dtype preservation, view-vs-copy semantics)
- guvectorize-convertible audit format (markdown table vs separate appendix)
- Backend fixture location (conftest.py vs separate helper module)
- `[cuda-jit]` extras separation (numba in base or only in `[cuda-jit]`)
- CUDA kernel unit test mock-vs-real-GPU policy (CI has no GPU runner)
- 28-kernel scope option A/B/C — plan-stage estimates 1 week and asks user

### Deferred Ideas (OUT OF SCOPE)

- CUDA kernel perf optimization (shared memory, warp shuffle, `cupy.RawKernel`, `ElementwiseKernel`) — v1.2
- pybind11 trampoline torch::Tensor removal — separate phase
- Wheel cp313+ extension — P1 D-08 cp310-cp312 stays
- Numba dispatch overhead optimization — v1.2 perf phase
- CuPy memory pool tuning (only if D-10 verification surfaces conflict)
- P10 phase split (if D-13 option A selected)

## Project Constraints (from CLAUDE.md)

- **Pure Python only** — `C++ 추가 코드 금지`. xp alias / numpy / cupy / numba is Python-only.
- **NumPy ≥ 2.0** (P1 D-08, pyproject.toml `numpy>=2.0,<3`). Use NumPy 2.0+ APIs only.
- **No new runtime deps beyond extras** — cupy MUST be `[cuda]` extras, NOT base dep. **numba is currently `[fast]` extras** (NJIT-07). Plan-stage decides if it merges into `[cuda-jit]` or stays separate.
- **Bit-exact ULP regression** — `verify.py --fp16 --ulp 1 --atol 0.001` MUST pass. ABS strict (96 tiles × 196609 hex lines) is the gate.
- **Tests = `uv run pytest`** (memory `reference_test_runner.md`) — system torch broken via libcusparseLt. All Phase 9 verification commands use `uv run`.
- **manylinux2014_x86_64** baseline. cibuildwheel pipeline preserved. No GPU runners in CI — cupy tests will SKIP-gracefully.
- **Little-endian assertion** at `src/main/python/riscv/gtx/__init__.py:37` — keep verbatim. ndarray FP16 view semantics same LE assumption.

## Phase Requirements

**⚠️ Status:** BM-01..BM-06 are defined ONLY in `.planning/ROADMAP.md` lines 274-280 (success criteria). They are NOT yet in `.planning/REQUIREMENTS.md`. **Plan-stage must add a task to transcribe** BM-01..06 into REQUIREMENTS.md `### Milestone v1.1 Post-Ship Polish` section before Wave 3 closes, otherwise the requirements coverage table stays inconsistent.

| ID | Description (transcribed from ROADMAP) | Research Support |
|----|---------|---|
| BM-01 | xp alias scaffold + DEVICE env contract — `import torch` count = 0 across `src/main/python/riscv/gtx/`; `GTX_USE_CUDA` gate works | Standard Stack — numpy/cupy `get_array_module` pattern, D-01..D-04 locked. Implementation in Wave 0. |
| BM-02 | numpy port of mem layer — `memory.py` (DDR + scratchpads), `register_file.py` (SPR int64). ABS strict PASS preserved. | `torch.zeros(dtype=torch.uint8, device=…)` → `xp.zeros(dtype=xp.uint8)` mechanical. D-10/D-11 verify steps explicitly listed. Wave 1. |
| BM-03 | numpy port of dispatch + ops — `ops/{mm,vec,act,spr}.py`, `dma_engine.py`, FP8 LUTs. GELU + RELU + SIGMOID + TANH + SOFTMAX strict PASS. | API mapping table below. FP8 strategy decision pending (Wave 0 task). Wave 2. |
| BM-04 | numpy port of tloop/sloop fusion — `tloop_buffer._execute_fused`, `_verify.compare_hex` no longer torch. ABS perf within ±10%. | D-15 1:1 drop-in. `tloop_buffer.py:466-486` mapping clear. Wave 3. |
| BM-05 | cupy opt-in extras + GPU smoke test gated on `GTX_USE_CUDA` — `pyproject.toml [cuda]` extras work; smoke test PASSes byte-identical to numpy | CuPy 13.x numpy-compat API. `cupy.asnumpy()` polyfill for file I/O. Wave 3 + CI matrix. |
| BM-06 | CLAUDE.md dependency policy + wheel size delta recorded | Independent wheel size measurement script. PyTorch wheel ~774 MB on Linux x86_64 manylinux as of 2026-03 (per pytorch/pytorch#177050). Expected delta: -50 to -200 MB depending on how PyTorch transitively pulls cublas/cudnn. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | ≥2.0,<3 (pinned in pyproject) | Default array backend; replaces torch.Tensor | Already declared. NumPy 2.x has IEEE 754 binary16 RNE FP16 + view-as-dtype + bit-exact behavior matching torch's CPU path. ([NumPy 2.4 dtype docs](https://numpy.org/doc/stable/reference/arrays.dtypes.html)) |
| cupy-cuda12x | ≥13.0,<15 | Opt-in GPU backend (drop-in numpy API replacement) | Standard get_array_module + xp alias pattern. v13 is NumPy 1.26 compat baseline; v14 (Jan 2026) shipped with NumPy 2.x semantics. ([CuPy v14 release](https://docs.cupy.dev/en/stable/overview.html)) |

**Version verification (run before pinning in pyproject.toml):**
```bash
uv pip show numpy | grep Version       # already ≥2.0,<3
# Cupy version dance: bump floor to 13 (NumPy 2.0 baseline compat); leave ceiling
# open. v14 shipped 2026-01 and brings full NumPy 2 semantics → may want
# `cupy-cuda12x>=13,<15` to allow v14 once tested.
```

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numba | ≥0.61.2,<0.66 (already in `[fast]` extras, see NJIT-07) | Reapply `@njit` (CPU) + `@cuda.jit`/`@guvectorize(target='cuda')` for GPU | Wave 3 D-13/D-14. **Note**: numba currently in `[fast]` extras — D-17 decision is whether to move to `[cuda-jit]` or keep `[fast]` and let `[cuda]` only depend on cupy. |
| ml_dtypes | ≥0.4 | Provides `float8_e4m3fn` numpy dtype | **ONLY if** plan-stage selects FP8 option (a) (see Summary #3). Otherwise NOT pulled. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `xp` alias | `cupy.get_array_module(arr)` per-call | Per-call lookup adds dispatch overhead; not viable for hot path. `xp` resolved once at import (D-02) is the right choice. |
| `to_host`/`to_device` helpers | `np.asarray(arr) / cp.asarray(arr)` raw | Bare `np.asarray(cp_arr)` works but is implicit — helpers make device crossings explicit (DMA boundary discipline, per D-12). |
| `numba.guvectorize` target='cuda' | Hand-written `cupy.RawKernel` strings | RawKernel is more powerful but invents a new source language. guvectorize keeps a single numpy-style Python source. ([Numba CUDA ufunc docs](https://numba.readthedocs.io/en/stable/cuda/ufunc.html)) |
| Remove FP8 conversions | Use `ml_dtypes` (`pip install ml-dtypes`) | New runtime dep violates CLAUDE.md "no new deps". FP8 LUT bit-pattern path (option b in Summary #3) keeps numpy-only and uses precomputed `FP16_TO_FP8_LUT` already in act.py:67-117. |
| Remove `DEVICE` symbol | Keep `DEVICE` as alias for `"cpu"` string | D-04 locked: no backwards-compat shim. |

**Installation:**
```bash
# Base:
uv pip install -e .                    # numpy only

# JIT speed (CPU):
uv pip install -e ".[fast]"            # numba

# CUDA + JIT:
uv pip install -e ".[cuda,fast]"       # cupy-cuda12x + numba (or new [cuda-jit])
```

## Architecture Patterns

### Recommended Project Structure (Phase 9 deltas only)

```
src/main/python/riscv/gtx/
├── config_params.py        # MODIFIED: add `xp` alias + `to_host`/`to_device` helpers
│                           #          remove `DEVICE` symbol
├── __init__.py             # MODIFIED: remove `import torch` ImportError surface (line 80-84)
│                           #          remove `from .config_params import DEVICE` (line 88)
├── _verify.py              # MODIFIED: torch.frombuffer → np.frombuffer (line 45-46)
├── tloop_buffer.py         # MODIFIED: _execute_fused → xp.* (lines 415-486)
├── sloop_buffer.py         # (No torch direct use — verify after wave 3)
├── npu.py                  # MODIFIED: _mxe_accum / _credit_ld / _credit_st zeros
│                           #          (lines 98-106); RegisterFile device arg → xp
│                           #          arg
├── unit/
│   ├── memory.py           # MODIFIED: _L2_GLOBAL / _L1_GLOBAL / _L0_GLOBAL → xp.zeros
│   │                       #          DDR_MEMORY → xp.zeros + ensure() doubling-grow on xp
│   │                       #          ddr_load/save_to_hex → to_host() at file boundary
│   │                       #          torch.frombuffer → np.frombuffer + to_device
│   ├── register_file.py    # MODIFIED: torch.zeros(int64) → xp.zeros(int64);
│   │                       #          .copy_(torch.as_tensor(value, int64)) → xp.copyto
│   │                       #          or arr[...] = value (xp.int64 cast)
│   ├── csr/                # (no torch use, no changes)
│   ├── context/
│   │   ├── dma_engine.py   # MODIFIED: .view(torch.dtype) → .view(xp.dtype)
│   │   │                   #          .copy_() → array slice assign or np.copyto
│   │   │                   #          .permute() → np.transpose
│   │   │                   #          .contiguous() → np.ascontiguousarray
│   │   │                   #          torch.cat → np.concatenate
│   │   │                   #          .cpu() at file/DDR boundary → to_host()
│   │   └── ...
│   └── ins/
│       └── ops/
│           ├── mm.py       # MODIFIED: 43 torch refs. gemm_core uses torch.matmul →
│           │               #          xp.matmul (BLAS dispatch — keep ULP discipline
│           │               #          from RESEARCH-prior Pitfall 2). torch.cat → np.concatenate.
│           ├── vec.py      # MODIFIED: 51 torch refs. _apply_unary (torch.abs/sign/…)
│           │               #          → xp.abs/sign/…. torch.cumsum → np.cumsum.
│           │               #          torch.where → np.where. torch.full_like → np.full_like.
│           ├── act.py      # MODIFIED: 79 torch refs. Activation kernels (gelu/tanh/sigmoid/
│           │               #          softmax/esum). FP8 strategy applied (see FP8 section).
│           └── spr.py      # MODIFIED: 1 torch ref (import only). Other math is integer.
└── (no new modules)

tests/gtx/
├── conftest.py             # MODIFIED: replace torch.cuda.is_available() gate with
│                           #          GTX_USE_CUDA gate; remove `import torch`
├── _mocks.py               # (audit needed — likely no changes)
├── test_csr_registry_chain.py  # MODIFIED: torch.int64 → np.int64, tensor.dtype check
└── test_mcast_copy_mem.py  # MODIFIED: 17 torch refs. seed helpers + assertions.
```

### Pattern 1: xp Alias with Import-Time Resolution (D-01/D-02)

**What:** Eagerly resolve numpy vs cupy at module import; freeze module-level `xp` reference. All gtx modules `from .config_params import xp` and use `xp.array`/`xp.zeros`/etc.

**When to use:** All array allocation, dtype views, math operations.

**Example:**
```python
# config_params.py (after Wave 0)
from __future__ import annotations
import os
import numpy as _np


def _resolve_backend():
    if os.environ.get("GTX_USE_CUDA", "").strip() not in ("1", "true", "TRUE"):
        return _np, _identity, _identity

    # GTX_USE_CUDA=1: cupy is mandatory.
    try:
        import cupy as _cp
    except ImportError as exc:
        raise RuntimeError(
            f"GTX_USE_CUDA=1 set but cupy is not importable ({exc}). "
            f"Install with: pip install 'spike[cuda]'"
        ) from exc

    return _cp, _cp.asnumpy, _cp.asarray


def _identity(arr):
    return arr


xp, to_host, to_device = _resolve_backend()

# xp is numpy or cupy (module). Use `from .config_params import xp` everywhere.
# to_host(arr): numpy-cpu copy (no-op for xp=numpy)
# to_device(arr): inverse (no-op for xp=numpy)
```

**Sources:** [CuPy basics — xp alias pattern](https://docs.cupy.dev/en/stable/user_guide/basic.html), [CuPy `get_array_module`](https://docs.cupy.dev/en/stable/reference/generated/cupy.get_array_module.html). HIGH confidence.

### Pattern 2: dtype-view bit-reinterpret (CRITICAL)

**What:** torch and numpy both support `.view(dtype)` for byte-reinterpret, but **subtle differences** in shape mechanics. Both reinterpret bytes (no copy) when dtypes match in size or array shape is compatible.

**When to use:** Every L0/L1/L2 buffer access goes through `.view(np.uint8/np.float16/np.uint16)` to alias uint8 storage as fp16/uint16.

**Example:**

```python
# torch (current src/main/python/riscv/gtx/unit/memory.py:212)
self._l0_f16_views = self.l0.view(torch.float16)
# Strict bit-reinterpret. Storage stays aliased.

# numpy equivalent (Wave 1):
# Storage as flat uint8 contiguous:
self._l0_f16_views = self.l0.view(np.float16)
# Same byte-reinterpret. Storage aliased.
```

**Subtle gotcha — chained view + reshape (tloop_buffer.py:466-470):**

```python
# torch:
src_f16 = (
    l2[src_base:src_base + total_bytes]
    .view(torch.float16)     # uint8 → float16 (length halves)
    .view(n, vec_size)       # second view is RESHAPE in torch (same dtype)
)

# numpy: torch's `tensor.view(N, M)` (same-dtype reshape) is NumPy's `.reshape(N, M)`
# numpy's `.view(dtype)` is dtype-only. Chained must split:
src_f16 = (
    l2[src_base:src_base + total_bytes]
    .view(np.float16)
    .reshape(n, vec_size)
)
```

**Sources:** [NumPy 2.4 dtype docs](https://numpy.org/doc/stable/reference/arrays.dtypes.html), [PyTorch view vs reshape](https://discuss.pytorch.org/t/implementation-of-numpy-function-view-uint8/27920). HIGH confidence.

### Pattern 3: In-place operations (D-15)

**What:** torch's `.copy_()` / `.zero_()` are in-place. NumPy has equivalents but spelled differently. ndarray slice assignment is in-place.

| torch | numpy equivalent | Notes |
|-------|------------------|-------|
| `dst.copy_(src)` | `np.copyto(dst, src)` or `dst[...] = src` | Both copy elementwise. `np.copyto` is the cleaner direct port. |
| `t.zero_()` | `t.fill(0)` or `t[...] = 0` | In-place fill. |
| `t.contiguous()` | `np.ascontiguousarray(t)` | Returns C-contiguous view (or copy if needed). |
| `torch.cat([a, b])` | `np.concatenate([a, b])` | Same semantics. |
| `t.permute(2, 1, 0)` | `t.transpose(2, 1, 0)` | NumPy `.transpose` takes axes; same semantics. |
| `torch.frombuffer(bytearray(b), dtype=torch.uint8)` | `np.frombuffer(bytearray(b), dtype=np.uint8)` | Direct equivalent. |
| `torch.as_tensor(x, dtype=torch.int64)` | `np.asarray(x, dtype=np.int64)` | Direct equivalent. |
| `t.to(torch.float32)` | `t.astype(np.float32)` | Type cast (copy unless dtype matches). |

**Cupy path:** All of these have the **same name** on `cupy.ndarray` — that's the value of cupy's numpy-API compat. So `xp.copyto(dst, src)` works on both. ([CuPy ndarray reference](https://docs.cupy.dev/en/stable/reference/ndarray.html)) HIGH confidence.

### Pattern 4: Reduction ops keep FP32-internal-accumulate discipline (existing P4/P5 invariant)

**What:** Project-wide discipline is "FP16 input → FP32 accumulate → FP16 output" for matmul/dot/sum/cumsum. torch's `_as_fp32` helper is mirrored in numpy via `arr.astype(np.float32)`.

**Example (mm.py:48-52, port):**
```python
# Before:
def _as_f32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype is torch.float32:
        return x.contiguous()
    return x.to(torch.float32).contiguous()

# After:
def _as_f32(x):
    if x.dtype == xp.float32:
        return xp.ascontiguousarray(x)
    return xp.ascontiguousarray(x.astype(xp.float32))
```

**Critical pitfall:** `np.matmul` dispatches to BLAS for large matrices and has been shown (project's own P4 RESEARCH "np.matmul Bit-Exactness Analysis") to drift up to 4 ULP / 0.0078 abs vs explicit FP32 scalar accumulate on 41/500 random 16×16×16 FP16-cast-to-FP32 trials. **Current src uses `torch.matmul` (mm.py:83) — which also calls BLAS**. So the FP32 BLAS dispatch is already what current code does; xp.matmul will dispatch to the same numpy BLAS. Behaviorally **no change** vs current ABS strict baseline.

For ULP-strict ops (e.g. `gemm_dot`, `gemm_reduce_sum_a`), the current src uses `torch.dot` / `torch.sum` (both BLAS). Wave 2 plan-stage should verify on Wave 1-end gate that these don't introduce drift, because we're swapping torch's BLAS dispatch for numpy's BLAS dispatch — both go through OpenBLAS/MKL on the wheels, so likely identical. If they aren't, fall back to explicit Python for-loop accumulate (as the original P4/P5 commits did before P7 numba-ization).

**Source:** Project history (e74b3f0/e169af5/fca5117 refactor commits). HIGH confidence.

### Pattern 5: `numba.guvectorize` with target switching (D-14)

**What:** Generic ufunc factory with single source supporting `target='cpu'`, `target='parallel'`, `target='cuda'`. Signature is `'(n)->(n)'` for elementwise, `'(m,k),(k,n)->(m,n)'` for matmul, etc.

**When to use:** Kernels whose shape is expressible as a generalized ufunc layout (one or more N-D inputs → one N-D output with declared shape relation).

**Example:**
```python
import numpy as np
from numba import guvectorize

@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)', target='cpu', nopython=True)
def relu_gufunc(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] if x[i] > 0.0 else 0.0

# Same source, target='cuda':
@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)', target='cuda')
def relu_gufunc_cuda(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] if x[i] > 0.0 else 0.0
```

**Target selection at runtime** (per CONTEXT D-14, "universal source + target switching"):

```python
# Wave 3 pattern:
if HAS_NUMBA and xp is _cp:
    _target = 'cuda'
elif HAS_NUMBA:
    _target = 'cpu'
else:
    _target = None  # bare python/numpy fallback

@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)', target=_target)
def relu_gufunc(x, out):
    ...
```

**Caveat 1 — Which P7 kernels are guvectorize-shaped?** Detailed inventory in next section.

**Caveat 2 — fp16 in `@cuda.jit`:** Numba's `@cuda.jit` does NOT support fp16 in kernel signatures (numba/numba#4402). Workaround: declare signatures as `void(float32[:], float32[:])` and upcast at the Python boundary (`f32 = f16.astype(np.float32)`). This is what current torch code already does (see Pattern 4).

**Sources:** [Numba universal functions docs](https://numba.readthedocs.io/en/stable/user/vectorize.html), [Numba CUDA Ufuncs](https://numba.readthedocs.io/en/stable/cuda/ufunc.html), [Numba fp16 status #4402](https://github.com/numba/numba/issues/4402). HIGH confidence on guvectorize mechanism; MEDIUM on the fp16 workaround being acceptable for ABS strict gate.

### Anti-Patterns to Avoid

- **❌ Don't use `cupy.get_array_module(arr)` per-call.** It's a per-call lookup. D-02 locks import-time resolution; respect it.
- **❌ Don't pollute hot path with `to_host()/to_device()`.** They are valid only at the **DMA / file I/O boundary** (per D-12). Sprinkling them in op kernels defeats GPU residency.
- **❌ Don't use `.view()` to reshape (only for dtype reinterpret).** NumPy's `.view(dtype)` is dtype-only. Chain `.view(dtype).reshape(...)` for both.
- **❌ Don't rely on `torch.Tensor.device` semantics for ndarray.** numpy.ndarray has no `.device` attribute. cupy.ndarray has `.device` returning a cupy Device object — but routing code on `.device == cpu` checks won't generalize. Use the xp alias as the device proxy.
- **❌ Don't add silent fallback for cupy import failure under `GTX_USE_CUDA=1`** (D-03). Fail-loud RuntimeError only. Past regression precedent (260518-ffr).
- **❌ Don't keep a `DEVICE` compat shim** (D-04). Let `ImportError` surface so callers update explicitly.
- **❌ Don't try to make `@cuda.jit` consume fp16 directly.** Upcast at the Python boundary.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| numpy/cupy device check at every call | Custom `is_gpu_array(x)` helper | xp alias resolved at import (D-02) | Per-call cost; current pattern explicit and frozen. |
| FP16 ↔ FP32 round-trip | Manual bit-twiddle | `arr.astype(np.float32)` + `.astype(np.float16)` | NumPy 2.x has IEEE 754 binary16 RNE built in (FOUND-01 verified). |
| Host↔device copy bookkeeping | Custom transfer classes | `to_host` / `to_device` helpers (D-12) | One-line semantics; xp=numpy = no-op. |
| Generic CPU/GPU ufuncs | Hand-written `cupy.RawKernel` strings | `numba.guvectorize(target=...)` (D-14) | Single Python source; standard ecosystem pattern. ([CuPy ↔ Numba interop](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)) |
| Replacing `numba.objmode` for transcendentals | Hand-rewrite gelu/tanh/sigmoid/softmax/esum | Keep `numba.objmode` block (NJIT-03 pattern) when target='cpu' | NJIT-03 already proved this is the ULP-clean path; transcendentals stay in objmode at target='cpu', or use `numba.cuda.libdevice.*` at target='cuda'. |
| FP8 e4m3 dtype on numpy | Install ml_dtypes | Use existing FP16↔FP8 LUTs in act.py (uint8 indexing) — see FP8 section below | Avoids new runtime dep (CLAUDE.md). LUTs already precomputed at import. |
| cupy 4 GiB allocation | Calling raw `cuMemAlloc` | `cp.zeros(N, dtype=cp.uint8)` (uses CuPy memory pool) — set `CUPY_GPU_MEMORY_LIMIT` env if needed | [CuPy memory mgmt](https://docs.cupy.dev/en/stable/user_guide/memory.html) |
| Detecting whether numpy or cupy in test fixture | Bespoke test backend marker | A `xp_fixture` in `conftest.py` reading `os.environ.get("GTX_USE_CUDA")` | One xp source of truth; no test-time backend negotiation. |

**Key insight:** The xp pattern + numba `guvectorize` target switching is the **standard ecosystem solution** (dask/xarray/scipy all use it). Don't invent new abstractions. Implementation cost is ~1 line per torch.* → xp.* rename + 1 unifying decorator factory in `_jit.py`.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | None. Project has no persistent on-disk datastores beyond `tests/gtx/data/golden/*.hex` (LE FP16 binary). Goldens are byte-pattern; reading them via `np.frombuffer(bytes, dtype=np.float16)` or `bytes` decoding (current `_verify.py:_parse_hex`) produces identical results regardless of backend. | Verified — no migration. |
| **Live service config** | None. No external services. | Verified — no migration. |
| **OS-registered state** | None. No OS-level integrations. | Verified — no migration. |
| **Secrets/env vars** | `GTX_USE_CUDA` (new), `GTX_DDR_SIZE`, `GTX_DDR_REVERSED`, `GTX_DDR_INIT`, `GTX_DDR_DUMP*`, `GTX_VENDOR_TEST_DIR`, `GTX_NO_EXIT`, `PYSPIKE_LIBS`, `PYSPIKE_EXTS`, `PYSPIKE_FAULTHANDLER`, `GTX_DEBUG_TILE_TRACE` (none — D-03 in P8 forbade it). **All env-var NAMES are unchanged** by Phase 9. The only new addition is `GTX_USE_CUDA` (D-01..D-03). | Document `GTX_USE_CUDA` in README + CLAUDE.md "Configuration" section. |
| **Build artifacts** | `build/lib.linux-x86_64-cpython-{310,314}/riscv/gtx/{gemm_core,vec_core,act_core}.py` — **stale from pre-refactor wheel builds**. These files do NOT exist in current `src/`. They could cause confusion if a developer searches the repo for `@njit`. | Recommend `rm -rf build/` as Wave 0 hygiene task. Document in Wave 0 plan. Also: `src/main/python/spike.egg-info/PKG-INFO` references "numba>=0.61.2,<0.66" — this regenerates on `pip install -e .`, not a permanent artifact. |

**Special — pyproject.toml `[tool.uv.sources]` lines 196-201:**
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
torchvision = [{ index = "pytorch-cu126" }]
```
These pin torch resolution to PyTorch CUDA 12.6 wheel registry. **D-17 must remove all four entries** (the `[[tool.uv.index]]`, the `[tool.uv.sources]` table for torch/torchvision, and lines 60-61 declaring `torch`/`torchvision` as deps). Otherwise `uv pip install -e .` may fail or fetch stale torch.

## Common Pitfalls

### Pitfall 1: `tensor.view(N, M)` is reshape in torch, not in numpy

**What goes wrong:** Direct copy `torch.tensor.view(torch.float16).view(n, vec_size)` to `np.ndarray.view(np.float16).view(n, vec_size)` silently produces wrong shapes because numpy's `.view(N, M)` is **NOT** reshape — it's a no-op on shape when given non-dtype args (actually it errors in NumPy 2.x).

**Why it happens:** API drift between torch.Tensor.view (dual-purpose) and ndarray.view (dtype-only since NumPy 1.x).

**How to avoid:** Mechanical conversion: `t.view(D).view(N, M)` → `arr.view(D).reshape(N, M)`. Audit all `.view(` sites with grep. Hot sites found:
- `tloop_buffer.py:466-470` (`.view(torch.float16).view(n, vec_size)`)
- `unit/context/dma_engine.py:267` (`.view(torch.float16).reshape(...)` — already uses reshape ✓)
- `unit/context/dma_engine.py:438,492,547` (`.view(height, length)` — these are torch reshape calls, port to `.reshape(height, length)`)
- `unit/memory.py:172` (`.view(dtype)` for dtype reinterpret — direct port)

**Warning signs:** `AttributeError`, shape mismatches in tests, or NumPy 2.x `TypeError: view() takes ...` raise.

### Pitfall 2: `np.matmul`/`np.dot`/`np.sum` BLAS ordering vs explicit FP32 accumulate

**What goes wrong:** `np.matmul(A_f32, B_f32)` dispatches to BLAS, which uses pairwise summation and may drift up to 4 ULP from C++ scalar accumulate on FP16 inputs (project's own P4 RESEARCH finding).

**Why it happens:** BLAS uses Kahan / pairwise summation for cache efficiency; vendor C++ uses scalar accumulate.

**How to avoid:** Current src **already calls `torch.matmul`** at mm.py:83, so the current ABS-strict baseline is already BLAS-dispatched. Wave 2 port to `np.matmul` should produce **the same numbers** (both go through wheel-shipped OpenBLAS or MKL). If a regression occurs:
1. Wave 1-end gate fails on a vendor op → trigger fallback to explicit Python 3-loop (gemm_core build artifact version showed this)
2. OR: apply `@njit` to the 3-loop in Wave 3 to recover perf

**Warning signs:** ULP > 0 mismatches on `test_njit_parity` GEMM test, or strict-mode FAIL on MM op vendor sweep.

### Pitfall 3: `torch.cuda.is_available()` chains and atexit ordering (260518-ffr)

**What goes wrong:** Importing torch in the test suite triggers CUDA init even with explicit CPU device, causing 5x slowdown + atexit ordering bugs (config_params.py:9-23 history). Phase 9 deletes torch from runtime, but `tests/gtx/conftest.py:15-21` still calls `torch.cuda.is_available()`.

**Why it happens:** torch import side-effects.

**How to avoid:** Wave 0 task: rewrite conftest.py CUDA gate as a `GTX_USE_CUDA` env-var check that imports cupy if requested:
```python
# tests/gtx/conftest.py (after Wave 0)
import os, pytest

if os.environ.get("GTX_USE_CUDA"):
    try:
        import cupy
    except ImportError:
        pytest.exit("GTX_USE_CUDA=1 but cupy is not installed", returncode=1)
# Default: no CUDA check, runs on CPU-only.
```

**Warning signs:** `tests/gtx/conftest.py:18` `if not torch.cuda.is_available(): pytest.exit(...)` — currently breaks collection if torch is uninstalled.

### Pitfall 4: `torch.float8_e4m3fn` has no NumPy equivalent

**What goes wrong:** `unit/ins/ops/act.py:128,136` cast to `torch.float8_e4m3fn` via `tensor.to(dtype)`. Numpy 2.x has no native fp8 dtype. ml_dtypes (external pkg) provides it but adds a new dep.

**Why it happens:** PyTorch shipped fp8 in 2.2+ for ML workloads; numpy ecosystem lags.

**How to avoid:** Three Wave-0 options (plan-stage chooses):

**Option (a) — Pull ml_dtypes** (violates CLAUDE.md "no new deps"):
```python
import ml_dtypes
fp8_arr = arr_fp16.astype(ml_dtypes.float8_e4m3fn)
```

**Option (b) — Use existing LUT (numpy-only, RECOMMENDED)**:
The file already precomputes `FP16_TO_FP8_LUT` (uint8[65536]) and `FP8_TO_FP16_LUT` (float16[256]) at import time (act.py:67-117). Port the cvt functions to LUT-indexed uint8 buffers instead of dtype casts:
```python
def fp16_to_fp8_e4m3(t_fp16):
    u16 = t_fp16.view(xp.uint16)
    return FP16_TO_FP8_LUT[u16]  # returns uint8

def fp8_e4m3_to_fp16(t_e4m3):
    return FP8_TO_FP16_LUT[t_e4m3]  # uint8 index → float16
```

**Option (c) — Descope FP8**:
Vendor regression sweep doesn't exercise FP8 conversions (project memory `reference_vendor_cpp.md` + ROADMAP Phase 9 only mentions ABS/GELU/etc.). Leave `cvt_qh`/`cvt_hq` raising `NotImplementedError` until v1.2.

**Recommendation:** Option (b). Already-built LUTs cover the path natively.

**Warning signs:** `AttributeError: module 'numpy' has no attribute 'float8_e4m3fn'` during Wave 2 import.

### Pitfall 5: cupy DDR-on-GPU at 4 GiB exhausts consumer VRAM (D-10)

**What goes wrong:** Consumer GPUs (RTX 4060 8 GB, RTX 3070 8 GB) → `cp.zeros(4 * 1024**3, dtype=cp.uint8)` succeeds but leaves <4 GB for everything else (scratchpads, CUDA context, cupy memory pool overhead, libraries). The cupy memory pool default-allocates blocks larger than requested for fragmentation avoidance.

**Why it happens:** Current src forces `_DDR_DEVICE = torch.device("cpu")` precisely to avoid this — D-10 reverses that.

**How to avoid:** Wave 1 plan-stage MUST:
1. Add `CUPY_GPU_MEMORY_LIMIT` documentation to README. Set conservative default.
2. Lower default DDR size when xp=cupy: if VRAM < 12 GB, default to 1 GiB DDR (`GTX_DDR_SIZE=1G` recommendation in README).
3. Verify `ensure()` doubling-grow works on cupy:
   ```python
   new_arr = xp.zeros(new_size, dtype=xp.uint8)
   new_arr[:current_size] = self._bytes
   ```
   Both ops are cupy-compatible (memory.py:145).
4. Verify `ddr_save_to_hex` adds `to_host(self._bytes)` before `bytes()` call (memory.py:318 currently uses `.detach().cpu().contiguous().numpy()`).

**Warning signs:** `cupy.cuda.memory.OutOfMemoryError`, vendor sweep failing only when `GTX_USE_CUDA=1` set on a small-VRAM card.

### Pitfall 6: numba `@cuda.jit` does not support fp16 in signatures

**What goes wrong:** Writing `@cuda.jit('void(float16[:], float16[:])')` raises `NotImplementedError` or runtime type error. Numba's `cuda.fp16` module provides operations BUT kernel signatures + dispatch use fp32 only.

**Why it happens:** numba/numba#4402 — fp16 in `@cuda.jit` is on the issue list since 2019, no land date.

**How to avoid:** D-14 universal-source kernels declare fp32 signatures only. Caller upcasts:
```python
@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)', target=target)
def relu_gufunc(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] if x[i] > 0.0 else 0.0

# Wrapper:
def relu(arr_fp16):
    f32_in = arr_fp16.astype(xp.float32)
    f32_out = xp.empty_like(f32_in)
    relu_gufunc(f32_in, f32_out)
    return f32_out.astype(xp.float16)
```

This matches the project's existing FP32-internal-accumulate discipline (Pattern 4), so it's a non-regression.

**Warning signs:** Compilation error from `@cuda.jit` when launching, or unexpected dtype mismatch errors.

### Pitfall 7: `torch.from_numpy(...)` / `torch.frombuffer(...)` semantics on bytes

**What goes wrong:** `_verify.py:45-46` uses `torch.frombuffer(bytes_object, dtype=torch.float16)` — torch.frombuffer is **CPU-only**, returns a CPU tensor. NumPy `np.frombuffer` is equivalent for reading byte data.

**Why it happens:** Direct API match.

**How to avoid:** Port mechanically. Note: `np.frombuffer` requires the buffer to be writable when passed to `bytearray` (memory.py:294 already uses `bytearray()` wrapper, so direct port works).

### Pitfall 8: silent ImportError swallow in `__init__.py:54-68`

**What goes wrong:** Phase 9's `__init__.py` removes the `import torch` ImportError surface at line 80-84. But the `try: from . import npu` block at line 54-68 still silently swallows ImportError as ImportWarning. If Wave 1 introduces a new import error (e.g., bad cupy version), the warning hides it.

**Why it happens:** Documented in memory `project_gtx_extension_silent_import_failure.md`.

**How to avoid:** Wave 0 task: tighten the `__init__.py` try/except to log AT LEAST the exception type and message at INFO/WARNING level. Project memory notes `uv run python -W error::ImportWarning -c "import riscv.gtx"` as diagnostic.

**Warning signs:** vendor regression suite rc=255 with "couldn't find extension 'gtx'" — Wave 1 should `-W error::ImportWarning` in conftest.py.

## Code Examples

Verified patterns from official sources + project sites.

### Example 1: xp alias resolution (Wave 0 — config_params.py)

```python
# Source: project's own D-01/D-02/D-03 decisions + CuPy basics docs
from __future__ import annotations
import os
import numpy as _np


def _identity(arr):
    return arr


def _resolve_backend():
    if os.environ.get("GTX_USE_CUDA", "").strip() not in ("1", "true", "TRUE"):
        return _np, _identity, _identity

    try:
        import cupy as _cp
    except ImportError as exc:
        raise RuntimeError(
            "GTX_USE_CUDA=1 set but cupy is not importable. "
            "Install with: pip install 'spike[cuda]'"
        ) from exc

    return _cp, _cp.asnumpy, _cp.asarray


xp, to_host, to_device = _resolve_backend()


# Topology constants (unchanged):
GTX_NEST_NUM: int = 4
GTX_SPU_NUM: int = 16
DEFAULT_DDR_SIZE: int = 4 * 1024 * 1024 * 1024
# ... (rest unchanged)
```

### Example 2: memory.py module-level scratchpad (Wave 1)

```python
# Source: project's unit/memory.py:46-57 (port to xp)
from .config_params import xp, GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES, ...

_L2_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_L2_SIZE_BYTES),
    dtype=xp.uint8,
)
_L1_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES),
    dtype=xp.uint8,
)
_L0_GLOBAL = xp.zeros(
    (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES),
    dtype=xp.uint8,
)


class DDR_MEMORY(MEMORY):
    def __init__(self, size=DEFAULT_DDR_SIZE):
        # D-10: DDR follows xp (was hard-coded to CPU).
        self._bytes = xp.zeros(size, dtype=xp.uint8)

    def ensure(self, end_offset):
        # Doubling-grow path. xp.zeros + slice copy works on both backends.
        cap = self.maximum_ddr()
        if end_offset > cap:
            raise ValueError(...)
        current_size = self.getsize()
        if end_offset > current_size:
            new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)
            new_size = min(new_size, cap)
            new_arr = xp.zeros(new_size, dtype=xp.uint8)
            if self._bytes is not None:
                new_arr[:current_size] = self._bytes
            self._bytes = new_arr
        return self._bytes

    def ddr_save_to_hex(self, filename, addr, size):
        # File I/O boundary: cross to host.
        ddr_src = self.ddr.raw()
        ...
        region_host = bytes(to_host(ddr_src[start:end]))  # cupy→numpy if needed
        # ... rest unchanged (string/bytes formatting)
```

### Example 3: vec.py unary kernel port (Wave 2)

```python
# Source: project's unit/ins/ops/vec.py:139-177 ported
from ....config_params import xp
import numpy as np  # for transcendental constants only (np.float32(...))

def _apply_unary(funct7: int, sub_op: int, view):
    if funct7 == 0x1D:   # SIGN: abs / neg / sign / step
        if sub_op == 0:
            return xp.abs(view)
        if sub_op == 1:
            return xp.negative(view)
        if sub_op == 2:
            return xp.sign(view)
        if sub_op == 3:
            return (view > xp.float16(0.0)).astype(xp.float16)
    if funct7 == 0x1E:   # ROUND
        if sub_op == 0:
            return xp.ceil(view)
        if sub_op == 1:
            return xp.trunc(view)
        if sub_op == 2:
            return xp.floor(view)
        if sub_op == 3:
            return xp.round(view)
    if funct7 == 0x1C:   # MATH (FP32 accumulator)
        f32 = view.astype(xp.float32)
        if sub_op == 0:
            return xp.sqrt(f32).astype(xp.float16)
        if sub_op == 1:
            return xp.exp(f32).astype(xp.float16)
        if sub_op == 2:
            tiny = xp.finfo(xp.float32).tiny
            return xp.where(
                f32 > 0.0,
                xp.log(xp.maximum(f32, tiny)),
                xp.zeros_like(f32),
            ).astype(xp.float16)
    return view.copy()  # ndarray copy (was .clone() on torch.Tensor)
```

### Example 4: tloop_buffer._execute_fused (Wave 3 — D-15)

```python
# Source: project's tloop_buffer.py:415-486 ported
from .config_params import xp


def _execute_fused(npu, frames) -> None:
    from .unit.ins.ops.vec import _apply_unary
    ...
    # Read N rows from L2 as a single (N, vec_size) fp16 array.
    src_f16 = (
        l2[src_base:src_base + total_bytes]
        .view(xp.float16)         # uint8 → float16 byte reinterpret
        .reshape(n, vec_size)     # was .view(n, vec_size) in torch
    )
    result_f16 = _apply_unary(funct7, sub_op, src_f16)

    dst_view = l2[dst_base:dst_base + total_bytes]
    xp.copyto(dst_view, result_f16.reshape(-1).view(xp.uint8))   # was dst_view.copy_(...)

    l1 = npu.mem.l1_byte(nest, spu)
    last_src = src_offs[-1]
    xp.copyto(l1[l_lo:l_lo + l_len], l2[last_src:last_src + l_len])
    xp.copyto(l1[s_lo:s_lo + l_len], result_f16[-1].view(xp.uint8))
    ...
```

### Example 5: numba.guvectorize universal source (Wave 3)

```python
# Source: Numba CUDA Ufuncs docs + project's NJIT-03 objmode pattern
from numba import guvectorize, objmode

# Resolve target once at module import (after xp is known):
from .config_params import xp
import numpy as np
try:
    import cupy as _cp
    _XP_IS_CUPY = (xp is _cp)
except ImportError:
    _XP_IS_CUPY = False

if _XP_IS_CUPY:
    _TARGET = 'cuda'
else:
    _TARGET = 'cpu'  # or 'parallel' for medium arrays


# Elementwise ReLU (P7 act_core kernel #1):
@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)',
             nopython=True, target=_TARGET)
def _relu_gufunc(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] if x[i] > 0.0 else 0.0


# Transcendental (gelu) — objmode escape on CPU target only:
@guvectorize(['void(float32[:], float32[:])'], '(n)->(n)',
             nopython=True, target=_TARGET)
def _gelu_gufunc(x, out):
    for i in range(x.shape[0]):
        # On CPU: objmode escape for glibc-clean tanhf. On CUDA: native libdevice.
        ...
```

**Caveat:** `numba.objmode` is **NOT supported in `@cuda.jit` / `target='cuda'`**. Plan-stage Wave 3 must split transcendental kernels into target-specific paths:
- CPU target: `@njit + objmode` (preserves NJIT-03 ULP=0 invariant)
- CUDA target: use `numba.cuda.libdevice.*` (e.g., `cuda.libdevice.tanhf(x)`) — accuracy contract is `libdevice` IEEE 754 which matches CUDA C++ math; verify against vendor C++ in Wave 3 gate.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=7,<9 + pytest-cov + pytest-benchmark (already in `[dev]` extras) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' --no-cov -v` |
| Full suite command | `uv run pytest tests/gtx/ --no-cov -v` (or `-k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX'` for smoke set) |
| Phase gate | `uv run pytest tests/gtx/ -v` (full) + `uv run pytest tests/gtx/test_njit_perf.py --benchmark-only` (perf) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BM-01 | xp alias resolves to numpy by default | unit | `uv run python -c "from riscv.gtx.config_params import xp; assert xp.__name__ == 'numpy'"` | ❌ Wave 0 — new test in `tests/gtx/test_xp_alias.py` |
| BM-01 | GTX_USE_CUDA=1 + no cupy → RuntimeError | unit | `GTX_USE_CUDA=1 uv run python -c "import riscv.gtx" 2>&1 \| grep RuntimeError` | ❌ Wave 0 — new test |
| BM-01 | `grep -rn 'import torch' src/main/python/riscv/gtx/` returns 0 | smoke | `bash -c "[[ $(rg -c 'import torch\|from torch' src/main/python/riscv/gtx/) == 0 ]]"` | ❌ Wave 3 — assertion |
| BM-02 | DDR doubling-grow preserves data | unit | `uv run pytest tests/gtx/test_dma_roundtrip.py::test_ddr_grow -x --no-cov` | ✅ exists (P3) |
| BM-02 | RegisterFile int64 storage matches torch numerics | unit | `uv run pytest tests/gtx/test_csr_registry_chain.py -x --no-cov` | ✅ exists, MODIFIED in Wave 1 |
| BM-02 | scratchpads zero-init at GtxNpu construction | unit | `uv run pytest tests/gtx/test_npu_construct.py -x --no-cov` | ✅ exists |
| BM-03 | ABS strict byte-exact preserved | integration | `uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' -x --no-cov -v` | ✅ exists |
| BM-03 | GELU/RELU/SIGMOID/TANH/SOFTMAX strict PASS | integration | `uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py' -k 'GELU or RELU or SIGMOID or TANH or SOFTMAX' -x --no-cov -v` | ✅ exists |
| BM-03 | Tile-2 unit test (P8 MTDMA-03) preserved | unit | `uv run pytest tests/gtx/test_multi_tile_dma.py -x --no-cov -v` | ✅ exists |
| BM-04 | ABS perf within ±10% of 94.82s | benchmark | `uv run pytest tests/gtx/test_abs_perf.py --benchmark-only` | ❌ Wave 3 — extend `tests/gtx/test_njit_perf.py` or new test |
| BM-04 | tloop fusion still fires on ABS | unit | `uv run pytest tests/gtx/test_tloop_fusion.py -x --no-cov -v` | ❌ Wave 3 (may exist as part of P8) — verify |
| BM-05 | cupy opt-in produces byte-identical ABS | manual (no GPU CI) | `GTX_USE_CUDA=1 uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' -x --no-cov -v` | ❌ Wave 3 — gated on GPU runner availability; mark as `@pytest.mark.gpu` and SKIP if no cupy |
| BM-06 | Wheel size delta ≤ 0 MB vs pre-migration | manual | `uv build --wheel && du -h dist/spike-*.whl` (compare against pre-migration tag) | ❌ Wave 3 — manual UAT step |
| BM-06 | CLAUDE.md "Dependencies" updated | doc | `grep -i 'cupy\|GTX_USE_CUDA' CLAUDE.md` | ❌ Wave 3 — doc update task |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS' --no-cov -v` (~95s)
- **Per wave merge:** 6-op smoke gate + tile-2 + perf bench
  - `uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py' -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v`
  - `uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v`
  - End of Wave 1 + Wave 3: `uv run pytest tests/gtx/test_njit_perf.py --benchmark-only` (perf budget D-08)
- **Phase gate:** full 84-op vendor sweep + tile-2 + perf bench + `grep -rn "import torch" src/main/python/riscv/gtx/` (must = 0)

### Wave 0 Gaps

- [ ] `tests/gtx/test_xp_alias.py` — new file covering BM-01 (3 unit tests: default, cupy-opt-in, fail-loud)
- [ ] `tests/gtx/conftest.py` — refactor CUDA gate to GTX_USE_CUDA gate (covers BM-01)
- [ ] `tests/gtx/test_abs_perf.py` — extend `test_njit_perf.py` or new file for BM-04 perf budget (94.82s ± 10%)
- [ ] `tests/gtx/test_fp8_strategy.py` — IF plan-stage selects option (b) (LUT-based FP8), verify round-trip cvt_qh ↔ cvt_hq

*(If plan-stage selects option (c) descope FP8, the test is replaced with `NotImplementedError` assertion.)*

## Open Questions

1. **Numba `[cuda-jit]` vs `[fast]` extras separation (D-17 deferred to plan-stage):**
   - What we know: numba already in `[fast]` for CPU. `[cuda]` (D-17) only adds cupy.
   - What's unclear: should cuda kernels require BOTH `[fast]` (for `@cuda.jit`) AND `[cuda]` (for cupy)? Or merge as `[cuda-jit] = ["numba>=...", "cupy-cuda12x>=..."]`?
   - Recommendation: Plan-stage propose **two extras**: `[cuda] = ["cupy-cuda12x>=13"]` (cupy only, for non-jit usage) and `[cuda-jit] = ["spike[fast,cuda]"]` (composite of fast+cuda). User can `pip install spike[cuda-jit]` for the full GPU + JIT path. Keeps `[fast]` (CPU-only JIT) intact for backwards compat.

2. **D-13 scope option A/B/C (28-kernel dual-impl):**
   - What we know: CONTEXT D-13 says all 28 must get dual-impl. Plan-stage MUST offer A/B/C.
   - What's unclear: Which kernels are actually hot-path on vendor regression? P8 work hinted ABS/GELU/RELU/SIGMOID/TANH/SOFTMAX (6 ops) are the smoke set — not the same set as numba's 28 kernels.
   - Recommendation: Plan-stage prepare the 28-kernel inventory table (next section) + measure ABS perf with `--benchmark` after Wave 2 to see which kernels dominate. Then propose option C (hot-path-only) if walltime is acceptable.

3. **FP8 strategy (Wave 0 decision needed):**
   - What we know: torch.float8_e4m3fn used in 2 sites. NumPy 2.x has no native fp8. ml_dtypes is the standard external. Project's act.py already has FP16↔FP8 LUTs precomputed.
   - What's unclear: whether vendor regression actually exercises cvt_qh/cvt_hq paths. If yes → option (b) (LUT). If no → option (c) (descope).
   - Recommendation: Plan-stage runs `uv run pytest tests/gtx/ -v --collect-only -k 'cvt or fp8'` to see if any test exercises these paths. If 0 tests, descope to v1.2.

4. **D-10 GPU-DDR VRAM budget verification:**
   - What we know: 4 GiB default + ~25 MB scratchpads → 4.025 GB total minimum. Consumer GPUs from 8 GB upward should fit. Sub-8GB cards risk OOM.
   - What's unclear: cupy memory pool overhead for `xp.zeros(4*1024**3, ...)` allocation pattern in practice on RTX 3060 6GB / RTX 4060 8GB.
   - Recommendation: Plan-stage adds `GTX_DDR_SIZE=1G` recommended default to README when xp=cupy; documents `CUPY_GPU_MEMORY_LIMIT` env var.

5. **D-11 SPR-on-GPU dispatch frequency (perf measure):**
   - What we know: 260518-ffr regression showed 5x slowdown from small-array GPU dispatch.
   - What's unclear: Whether `RegisterFile.tensor` access at every dispatch (likely ≥ 1M times during ABS sweep) triggers similar.
   - Recommendation: Plan-stage Wave 1-end gate measures ABS perf with xp=cupy. If > 105s, fall back to `RegisterFile` keeping `_tensor` on host (numpy) regardless of xp. Documented exception per CONTEXT.

6. **REQUIREMENTS.md missing BM-01..06:**
   - What we know: BM-01..06 only in ROADMAP.md (success criteria text). Not transcribed to REQUIREMENTS.md.
   - Recommendation: Wave 3 plan adds a task: "Transcribe BM-01..06 from ROADMAP.md to REQUIREMENTS.md `### Milestone v1.1` section + update Traceability table + Coverage count".

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| numpy | Default backend (all Waves) | ✓ | (project pinned ≥2.0,<3) | — |
| cupy-cuda12x | xp=cupy path (Waves 1-3) | ? | needs `uv pip install spike[cuda]` to provision | xp=numpy default; cupy not required for ABS strict gate |
| numba | `[fast]` extras (Wave 3 D-13/D-14) | ✓ already in `[fast]` extras | ≥0.61.2,<0.66 | If numba absent: pure-numpy fallback (NJIT-01 pattern) |
| ml_dtypes | FP8 option (a) only — NOT recommended | ✗ (not installed; would be new dep) | — | Option (b): LUT-based FP8 (no new dep). Option (c): descope FP8. |
| CUDA toolkit / driver | cupy-cuda12x runtime | depends on dev box | needs CUDA 12.x runtime | xp=numpy default; CI on cloud has no GPU → tests SKIP gracefully |
| pytest-benchmark | Perf gate D-08 | ✓ already in `[dev]` | ≥4.0,<6 | — |

**Missing dependencies with no fallback:** None — all hard deps already present.

**Missing dependencies with fallback:** cupy + CUDA toolkit (xp=numpy path is full default).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `import torch` + DEVICE | xp alias resolved at import (D-01/D-02) | This phase | Removes hard pytorch dep; ~50-200 MB wheel size delta. |
| `tensor.view(N, M)` reshape | `arr.view(dtype).reshape(N, M)` chain | Wave 2/3 | NumPy 2.x removed positional N,M from `.view()`. |
| `numba.cuda.jit` fp16 in signature | Upcast to fp32 at Python boundary | Wave 3 | numba/numba#4402 — fp16 not supported in @cuda.jit signatures. |
| `torch.float8_e4m3fn` | LUT-based FP8 (existing act.py:67-117) OR descope | Wave 2 | NumPy has no native fp8. |
| Per-call device detection | xp resolved once at import | This phase | Eliminates per-instruction dispatch overhead. |

**Deprecated/outdated:**
- `build/lib.linux-x86_64-cpython-{310,314}/riscv/gtx/{gemm_core,vec_core,act_core}.py` — stale wheel artifacts from pre-refactor builds; do NOT consult as current code. Action: `rm -rf build/` in Wave 0.
- `src/main/python/spike.egg-info/PKG-INFO` references torch wheel build; regenerates on `pip install -e .` — not a permanent issue, just be aware D-17 changes the next regenerated version.

## P7's 28 Kernels — Inventory + guvectorize Audit

**Total: 28 (3 mm + 7 vec + 18 act)** — matches CONTEXT D-13.

**Current state:** All 28 exist as functions in `unit/ins/ops/{mm,vec,act}.py` but **none decorated with `@njit`**. P9 must (a) port to numpy and (b) re-add JIT layer.

### MM kernels (3) — `unit/ins/ops/mm.py`

| Kernel | Lines | Current torch surface | guvectorize convertible? | Notes |
|--------|-------|----------------------|--------------------------|-------|
| `gemm_core(A, B, has_bias, bias_fp32)` | 55-98 | `torch.matmul` + cast | ✓ `'(m,k),(k,n)->(m,n)'` if no bias; with bias needs 2-stage (matmul + add) | BLAS-dispatched; numba's gemm support is solid on CPU; CUDA target should use `cupy.matmul` instead of numba (faster). |
| `gemm_reduce_sum_a(A, prior_accum)` | 100-108 | `torch.sum` + float cast | ✓ `'(n)->()'` (reduction) | scalar output — guvectorize handles. |
| `gemm_dot(A, B, prior_accum)` | 111-120 | `torch.dot` + float cast | ✓ `'(n),(n)->()'` | scalar output. |

### VEC kernels (7 + `_apply_unary` macro = 10 subops) — `unit/ins/ops/vec.py`

| Kernel | Lines | Current torch surface | guvectorize convertible? | Notes |
|--------|-------|----------------------|--------------------------|-------|
| `sasmd_kernel(a, b, op)` | 41-63 | branching torch ops | ❌ `op` is runtime int → needs wrapper. 4 separate gufuncs (add/sub/mul/div). | Split into 4 small gufuncs and dispatch in Python wrapper. |
| `dot_kernel(a, b)` | 66-72 | torch.dot | ✓ `'(n),(n)->()'` | scalar output. |
| `vsum_kernel(view)` | 75-77 | torch.sum | ✓ `'(n)->()'` | scalar output. |
| `clamp_min_kernel(a, scalar)` | 80-82 | torch.clamp | ✓ `'(n),()->(n)'` | scalar input via guvectorize. |
| `clamp_max_kernel(a, scalar)` | 85-87 | torch.clamp | ✓ `'(n),()->(n)'` | — |
| `accum_kernel(a)` | 90-96 | torch.cumsum | ⚠️ guvectorize supports `'(n)->(n)'` but cumsum is sequential — `target='cuda'` is per-element parallel; needs manual prefix-scan or stays in cupy `cp.cumsum` (BLAS-style) | Recommend: skip guvectorize, use `xp.cumsum` directly. |
| `arange_kernel(n, start, step)` | 99-103 | torch.arange + math | ❌ scalar→array (not a guvectorize shape) | Direct `xp.arange + scalar math`, no JIT needed (one-shot). |
| `_apply_unary(funct7, sub_op, view)` | 139-177 | 10 different ops (abs/neg/sign/step/ceil/trunc/floor/round/sqrt/exp/log) | ⚠️ Split into 10 separate gufuncs (`abs`, `neg`, ...), each `'(n)->(n)'` | Major refactor — Python dispatch dispatches to chosen gufunc. |

### ACT kernels (18) — `unit/ins/ops/act.py`

| Kernel | Lines | Current torch surface | guvectorize convertible? | Notes |
|--------|-------|----------------------|--------------------------|-------|
| `fp8_e4m3_to_fp16(t_e4m3)` | 123-124 | torch fp8 cast | ⚠️ Depends on FP8 strategy (see Pitfall 4). If option (b): just LUT indexing → `'(n)->(n)'` works. | Plan-stage decides. |
| `fp16_to_fp8_e4m3(t_fp16)` | 127-128 | torch fp8 cast | ⚠️ Same — LUT indexing. | — |
| `cvt_qh(arr_f16, scale, offset)` | 131-136 | mixed scale+offset+fp8 | ⚠️ Same. | — |
| `cvt_hq(arr_f8, scale, offset)` | 139-144 | same | ⚠️ Same. | — |
| `cvt_ih(arr_f16, scale, offset)` | 147-153 | int↔fp16 with scale | ✓ `'(n),(),()->(n)'` | guvectorize-friendly. |
| `cvt_hi(arr_f16, scale, offset)` | 155-161 | same | ✓ | — |
| `cvt_hn(arr_i32, scale, offset)` | 163-169 | int32→fp16 | ✓ | — |
| `cvt_sh(arr_f32)` | 171-172 | fp32→fp16 | ✓ `'(n)->(n)'` | — |
| `cvt_hs(arr_f16)` | 176-177 | fp16→fp32 | ✓ | — |
| `cvt_dh(arr_f64)` | 181-182 | fp64→fp16 | ✓ | — |
| `cvt_hd(arr_f16)` | 186-187 | fp16→fp64 | ✓ | — |
| `relu(arr_f16)` | 194-196 | torch.relu | ✓ `'(n)->(n)'` | trivially elementwise. |
| `prelu(arr_f16, slope)` | 198-200 | torch.where + slope | ✓ `'(n),()->(n)'` | — |
| `gelu(arr_f16)` | 202-204 | torch.gelu (uses tanh+constants) | ⚠️ Convertible BUT transcendental — needs `objmode` for ULP=0 (NJIT-03 invariant). CUDA target: use `cuda.libdevice.tanhf`. | Verify against vendor C++ in Wave 3 gate. |
| `tanh(arr_f16)` | 206-208 | torch.tanh | ⚠️ Same as gelu. | — |
| `sigmoid(arr_f16)` | 210-212 | torch.sigmoid (uses exp) | ⚠️ Same. | — |
| `softmax(arr_f16)` | 214-216 | torch.softmax | ⚠️ Multi-step (max/exp/sum/div) — likely best as raw xp calls. | Skip guvectorize. |
| `esum(arr_f16, max_val, init_accum)` | 218-220 | torch.exp + sum | ⚠️ Same. | — |
| `pool_max(arr_f16, kernel_size)` | 228-230 | torch.max_pool | ✓ but `(n,k)->(m)` shape varies | Plan-stage decides. |
| `pool_avg(arr_f16, kernel_size)` | 234-236 | torch.avg_pool | ✓ same | — |

**Summary classification:**
- **Cleanly guvectorize-convertible**: 14 kernels (gemm 3, vec 4 [dot/vsum/clamp_min/clamp_max], act 7 cvts [ih/hi/hn/sh/hs/dh/hd] + relu + prelu)
- **Needs Python wrapper + multi-gufunc**: 5 (sasmd 4-way, _apply_unary 10-way)
- **Transcendental — needs target-specific path**: 5 (gelu, tanh, sigmoid, softmax, esum) — matches NJIT-03's "5 transcendental kernels" set verbatim
- **Skip JIT, use native xp**: 4 (accum, arange, pool_max, pool_avg) — vectorized one-shot or native bulk ops
- **FP8-dependent (strategy decision blocks)**: 4 (fp8_to_fp16, fp16_to_fp8, cvt_qh, cvt_hq)

**Plan-stage option C (hot-path-only)** likely covers: gemm_core (CPU has BLAS, CUDA uses cupy.matmul), relu, the 7 cvt operations, and the 5 transcendentals. ~13 kernels — manageable scope. Skip guvectorize on the rest and use bare xp ops.

## Sources

### Primary (HIGH confidence)

- Project sources (all paths absolute):
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/phases/09-backend-migration-numpy-cupy/09-CONTEXT.md` — 17 locked decisions D-01..D-17
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/ROADMAP.md` lines 266-285 — Phase 9 success criteria (BM-01..06 origin)
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md` — P8 invariants to preserve
  - `/mnt/e/14_NIGHTLY/pyspike/CLAUDE.md` lines 16-34 — NumPy backend constraint
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/config_params.py:1-25` — DEVICE SSOT + comments
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/__init__.py:54-88` — torch import surface + DEVICE re-export
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/memory.py:46-79,145,294-301` — module-level scratchpad alloc + DDR doubling-grow + frombuffer site
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/register_file.py:1-200` — RegisterFile int64 storage
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/unit/ins/ops/{vec.py,act.py,mm.py,spr.py}` — 28 kernel sites
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/tloop_buffer.py:415-486` — _execute_fused fusion path
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/gtx/_verify.py:9,45-46` — compare_hex torch.frombuffer
  - `/mnt/e/14_NIGHTLY/pyspike/tests/gtx/conftest.py:1-21` — current CUDA-required gate
  - `/mnt/e/14_NIGHTLY/pyspike/pyproject.toml:60-61,190-202` — torch deps + uv index for cu126
- Official docs:
  - [CuPy Basics — xp alias pattern](https://docs.cupy.dev/en/stable/user_guide/basic.html)
  - [CuPy `get_array_module`](https://docs.cupy.dev/en/stable/reference/generated/cupy.get_array_module.html)
  - [CuPy Interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html) — `__cuda_array_interface__` + numba interop
  - [CuPy Memory Management](https://docs.cupy.dev/en/stable/user_guide/memory.html) — memory pool, `CUPY_GPU_MEMORY_LIMIT`
  - [Numba Universal Functions](https://numba.readthedocs.io/en/stable/user/vectorize.html) — guvectorize syntax
  - [Numba CUDA Ufuncs](https://numba.readthedocs.io/en/stable/cuda/ufunc.html) — target='cuda' patterns
  - [Numba CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) — version 3
  - [Numba fp16 meta issue (#4402)](https://github.com/numba/numba/issues/4402) — `@cuda.jit` fp16 signature limitation
  - [NumPy 2.4 dtype docs](https://numpy.org/doc/stable/reference/arrays.dtypes.html) — byte-order, view semantics
  - [ml_dtypes on PyPI](https://pypi.org/project/ml-dtypes/) — fp8 dtype provider
  - [PyTorch wheel size investigation #177050](https://github.com/pytorch/pytorch/issues/177050) — 2026 size trajectory

### Secondary (MEDIUM confidence)

- [GPU-Accelerated Python: Writing CUDA Kernels with Numba — Medium 2026-04](https://medium.com/@aditi.sikarwar25/gpu-accelerated-python-writing-cuda-kernels-with-numba-0af9f55d97c6) — guvectorize + cuda example
- [CuPy v14 release notes (Jan 2026)](https://github.com/cupy/cupy/releases) — NumPy 2.x semantics in v14

### Tertiary (LOW confidence)

- None retained — all key claims verified against official docs or project source.

## Metadata

**Confidence breakdown:**

- xp alias scaffold (D-01..D-04): HIGH — pattern is standard ecosystem; D-03 fail-loud has direct precedent (260518-ffr)
- Migration mechanics (Wave 1/2 mappings): HIGH — torch ↔ numpy API correspondences are 1:1 documented
- CuPy device placement (D-09..D-12): MEDIUM — D-10 GPU-DDR has unverified VRAM budget consequences; D-11 has explicit perf-measure exception path
- Numba × xp integration (D-13/D-14): MEDIUM — guvectorize target switching is documented but project has no precedent; fp16-in-cuda is a real limitation requiring upcast at boundary
- 28-kernel scope (D-13): LOW on actual implementation cost — depends on plan-stage A/B/C selection and FP8 strategy
- FP8 strategy: HIGH-MEDIUM — numpy lacks native fp8 is well-documented; option (b) LUT-based path proven viable since LUTs already in act.py
- Test infrastructure (D-16): HIGH — 54 torch refs in `tests/gtx/` audited; conftest.py CUDA gate explicitly identified
- pyproject.toml (D-17): HIGH — `[project.dependencies]` torch entry + `[tool.uv.sources]` cuda12.6 index documented

**Research date:** 2026-05-18
**Valid until:** 2026-06-18 (30 days for stable libraries; less if CuPy v15 or Numba 0.66+ ship with new fp16 support)

## RESEARCH COMPLETE

**Phase:** 9 — Backend migration: PyTorch → NumPy + CuPy opt-in
**Confidence:** HIGH overall (LOW only on the 28-kernel implementation cost, deferred to plan-stage A/B/C choice)

### Key Findings

1. **Codebase reality differs from CONTEXT description**: P7's 28 njit kernels were merged into op modules between commits `e74b3f0`/`e169af5`/`fca5117` and **rewritten as torch tensor ops with NO @njit decorators**. Phase 9 is first-time numba application to numpy ports, not "swap njit backend". Kernel count of 28 still matches (3+7+18).
2. **`tests/gtx/conftest.py:18` currently REQUIRES CUDA via torch** — must be ported in Wave 0 or Wave 1+ tests cannot collect on no-GPU box.
3. **`torch.float8_e4m3fn` has no numpy equivalent** — Wave 0 must decide: (a) ml_dtypes dep, (b) LUT-based using existing FP16↔FP8 LUTs in act.py (RECOMMENDED), or (c) descope FP8.
4. **`tensor.view(N, M)` is reshape in torch; in numpy `.view()` is dtype-only** — pervasive `.view(N, M)` sites at `tloop_buffer.py:466-470`, `dma_engine.py:438/492/547` need `.reshape(N, M)`. Bit-reinterpret `.view(dtype)` ports 1:1.
5. **D-10 GPU-DDR at 4 GiB risks consumer VRAM exhaustion** — plan-stage must add `GTX_DDR_SIZE=1G` recommended default when xp=cupy + `CUPY_GPU_MEMORY_LIMIT` env documentation.
6. **`@cuda.jit` does not support fp16 in kernel signatures** (numba#4402) — kernels declare fp32, callers upcast at Python boundary (matches project's existing FP32-internal-accumulate discipline so no regression).
7. **REQUIREMENTS.md missing BM-01..06** — plan-stage must add transcription task to Wave 3.

### File Created

`/mnt/e/14_NIGHTLY/pyspike/.planning/phases/09-backend-migration-numpy-cupy/09-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack (numpy/cupy/numba) | HIGH | Versions pinned; cupy 13 ↔ numpy 2.0 compat confirmed via official docs |
| Architecture (xp alias, helpers) | HIGH | Standard ecosystem pattern (dask/xarray/scipy) |
| Pitfalls (view/reshape, FP8, fp16-cuda) | HIGH | Each verified against official docs or project src |
| 28-kernel JIT scope | LOW | Depends on plan-stage A/B/C decision; FP8 strategy branch |
| DDR-on-GPU (D-10) | MEDIUM | VRAM budget verification deferred to plan-stage |
| Perf budget (D-08 ±10%) | MEDIUM | Cannot pre-measure; tloop_buffer fusion preserved should hit ≤105s |

### Open Questions

1. `[cuda-jit]` extras separation (plan-stage decision)
2. D-13 28-kernel scope option A/B/C (plan-stage proposes to user with 1-week estimate)
3. FP8 strategy a/b/c (plan-stage decides in Wave 0)
4. D-10 GPU-DDR VRAM acceptance criteria (plan-stage Wave 1-end gate)
5. D-11 SPR-on-GPU perf exception (plan-stage Wave 1-end gate)
6. BM-01..06 transcription to REQUIREMENTS.md (Wave 3 task)

### Ready for Planning

Research complete. Planner can now produce 4 Wave PLAN.md files. Wave 0 MUST include FP8 strategy decision + conftest CUDA gate refactor + 28-kernel scope user sign-off. Subsequent waves follow CONTEXT D-05 sequence.
