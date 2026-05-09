# Phase 7: Numba Dynamic Optimization - Research

**Researched:** 2026-05-08
**Domain:** numba @njit acceleration of stateless GTX NPU compute kernels with bit-exact preservation, vendor 84-op regression sweep, base+extras wheel packaging
**Confidence:** HIGH for stack/version/architecture, MEDIUM-LOW for transcendental ULP-0 parity (a critical drift risk surfaced and quantified)

## Summary

Phase 7 wraps 25 stateless NumPy compute kernels (`gemm_core` 3, `vec_core` 7, `act_core` 7 act + 2 pool + 9 cvt) in `@njit(cache=True)` decorators while gating numba behind a `[fast]` extras install with lazy-import + automatic NumPy fallback. The infrastructure is straightforward — every constraint from CONTEXT D-01..D-16 is supported by current numba 0.61+ on cp310-cp312 + numpy 2.x + manylinux2014_x86_64. The empirical evidence is overwhelming: the `gemm_core` 3-loop runs **~455× faster** under numba (910 μs → 2.0 μs for 16×16×16 GEMM) at **bit-exact** output, comfortably exceeding the 5× walltime target.

**The single critical risk is transcendental ULP-0 parity.** Empirical testing of `numba 0.63.1 + numpy 2.2.6` confirms that `np.tanh` on FP32 arrays under `@njit(fastmath=False)` differs from glibc's `tanhf` (which `np.tanh` uses outside numba) by ~1 ULP in 803/2048 random samples — and this drift propagates through GELU to **9/1024 FP16 ULP-0 mismatches** for D-12 strict per-kernel parity. The drift is intrinsic to LLVM's transcendental implementation (`NUMBA_DISABLE_SVML=1` does NOT eliminate it; it is in `tanhf`/`expf` LLVM intrinsics themselves). The kernels affected are: `gelu`, `tanh_act`, `sigmoid`, `softmax`, `esum`. Plain arithmetic kernels (`gemm_core`, `vec_core` SASMD/dot/vsum, `relu`, `prelu`, pool, all 9 cvt) are bit-exact under numba.

**Primary recommendation:** Lock in numba `>=0.61.2,<0.66` (NOT 0.59 — that pins numpy<1.27 and breaks our numpy>=2.0 floor). Architect kernels as `kernel_njit(arr_f32) -> ndarray_f32` (FP32 in/out, FP16 cast at engine boundary, NEVER inside @njit) because **numba does not support np.float16 on CPU**. For the 5 transcendental kernels, present three explicit options to the user in plan-stage Plan 01 — (A) accept ~1 ULP drift on these 5 kernels and relax D-12 to "ULP ≤ 1, atol ≤ 0.001" for them only; (B) use `with objmode(...)` to call back to NumPy's libm tanh/exp inside @njit (bit-exact, ~50% slower than full JIT but still 100×+ over pure NumPy); (C) skip @njit on these 5 kernels entirely (still get 5× walltime from gemm_core + vec_core + non-transcendental act/pool/cvt). RESEARCH recommends option (B) — preserves D-12 strict invariant at acceptable cost.

## Project Constraints (from CLAUDE.md)

- **Tech stack:** Python 3.8+ / NumPy ≥ 2.0 / pyspike pybind11 trampolines. **C++ 추가 코드 금지** (D-05 numba은 사용자 결정으로 그대로 유지; cython/C extension은 영구 거부).
- **Compatibility:** `requires-python = ">=3.10"`, manylinux2014_x86_64, cp310-cp312 cibuildwheel matrix.
- **Performance:** NumPy backend assumed; **performance hotspot이 발견되면 v2에서 cython/C 확장 검토** — D-05 numba는 이 절을 P7로 advance한 결과 (사용자 명시 결정).
- **Bit-exact:** ULP 허용오차 `verify.py --fp16 --ulp 1 --atol 0.001` 수준. **D-12는 더 엄격 — per-kernel ULP-0** (NumPy oracle vs JIT must produce identical FP16 bytes via `np.array_equal(out_numpy.view(np.uint16), out_njit.view(np.uint16))`).
- **GSD Workflow Enforcement:** Edit/Write tools must be entered through GSD workflow — research → plan → execute. Direct repo edits without `/gsd:execute-phase` not permitted.
- **Forbidden libraries beyond NumPy:** scipy / scikit-learn / torch / JAX / cython / C extensions all rejected. **Numba is the SOLE exception** (P7 D-05).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scope & wheel 배포 전략:**
- **D-01:** Phase 7 = v1 ship gate 내부. P6 회귀 strict-mode green이 진입 조건. P7 완료 후 v1.0 release.
- **D-02:** Lazy import + auto NumPy fallback. `try: from numba import njit / HAS_NUMBA = True; except ImportError: HAS_NUMBA = False / def njit(...): return passthrough`. base wheel = NumPy-only, P7 사용자가 numba 미설치라도 깨짐 zero.
- **D-03:** 단일 wheel + `[project.optional-dependencies]` extras. `pyproject.toml`에 `[project.optional-dependencies]` 섹션 신설 + `fast = ["numba>=0.59"]` (정확 버전은 plan-stage). `pip install spike` (base) → NumPy-only, `pip install spike[fast]` → numba 가속.
- **D-04:** REQUIREMENTS.md `Out of Scope` numba 항목 재문구. 변경: "numba는 v1 hard dependency 제외 (Phase 7의 optional `spike[fast]` extras 통한 lazy 가속은 허용). cython / JAX / torch / scipy는 v1 전 영역에서 hard 또는 optional dep 모두 제외."

**Library 선택 + JIT 적용 범위:**
- **D-05:** 라이브러리 = numba (LLVM JIT, ≥0.59). Cython / C extension / PyPy / hybrid 모두 거부 lock-in.
- **D-06:** JIT 적용 범위 = stateless cores 한정. `gemm_core.py` 3 + `vec_core.py` ~7 + `act_core.py` 7 act + 2 pool + 9 cvt = **약 25 kernel** (정확 카운트는 plan-stage). engine layer (mm/vec/act_engine) 비포함 — pybind11 객체/dataclass/dict 의존 → numba 비호환.
- **D-07:** JIT signature = lazy first-call dispatch. `@njit(cache=True)`만 작성, signature 명시 없음. type drift는 D-12가 자동 검출.
- **D-08:** 컴파일 캐싱 = `@njit(cache=True)` 디스크 자동 (`__pycache__/<module>.<func>-<hash>.nbi`/`.nbc`). 첫 import에서 컴파일, 이후는 cache hit.

**Bit-exactness 보장 + Fallback:**
- **D-09:** `fastmath=False` (numba 기본) + explicit FP32 Python for-loop 보존. `np.dot`/`np.matmul`/`np.einsum`/`np.sum`(FP16) 등 BLAS pairwise summation 사용 금지. PROJECT.md Core Value 절대 보장.
- **D-10:** Acceptance gate = vendor `test/<OP>/n1s16/` 풀 sweep. 자산 보유 op은 `compare_hex(strict=True)` PASS 강제, 자산 미보유 op은 graceful skip. 사용자 명시: "test/{OP} 103개 모두 통과 (데이터가 없는 경우 skip)" (실측 84 ops with all having `_ref.txt` — 자세한 인벤토리 §"Vendor Op Directory Inventory" 아래 참조).
- **D-11:** Fallback 관리 = same module dual export + 자동 dispatcher. NumPy 원본 + numba JIT 버전이 한 모듈에 공존, `from gemm_core import gemm_core` 단일 import만 알면 됨 — JIT 여부 투명.
- **D-12:** per-kernel ULP-0 parity 단위 테스트. `tests/gtx/test_njit_parity.py` 신규 — 25 kernel 모두 NumPy vs JIT delta_ulp == 0 (`np.array_equal(a.view(np.uint16), b.view(np.uint16))`).

**성능 목표 + acceptance gate:**
- **D-13:** 성능 목표 = wall-clock 5× 이상. vendor 84-op sweep 전체 walltime 기준, P6 NumPy baseline 대비 5× 이하.
- **D-14:** 측정 도구 = pytest-benchmark. `tests/gtx/test_njit_perf.py` 신규. dev extras에 추가.
- **D-15:** Wheel size 정책 = base 50MB cap만 유지, extras transitive size 비고려.
- **D-16:** 3-tier 테스트 구조. Tier 1 = `test_njit_parity.py` (정확성 가드), Tier 2 = `test_regression_fw_full_sweep.py` (vendor 84-op end-to-end), Tier 3 = `test_njit_perf.py` (5× 보증).

### Claude's Discretion

- **`@njit` decorator 적용 패턴** — 직접 (`@njit(...)\ndef ...`) vs 재호출 (`fn_njit = njit(...)(fn_numpy)`). plan-stage 검증.
- **vec_core.py / act_core.py 정확 kernel 카운트** — plan-stage 전수 검사. 본 RESEARCH §"Exact Kernel Inventory"에서 25 → 25 lock-in (gemm 3 + vec 7 + act 18; FP8 LUT-builders 2개는 import-time only, JIT 비대상).
- **vendor `test/<OP>/` 정확 디렉토리 카운트** — 본 RESEARCH 실측: **84 ops** (D-10에서 98-103 추정 → 84 lock-in; 모든 84개에 `_ref.txt` 자산 보유 → graceful skip가 거의 발화 안 함).
- **vendor 자산 → `.hex` 변환 스크립트 확장** — `scripts/import_vendor_golden.py` (P6 P03 산출, 9-op 코어셋 처리)을 84-op로 확장 + 식별된 누락 op skip 자동화.
- **NumPy fallback 활성화 분기점** — module-top vs 중앙 `riscv/gtx/_jit.py` 모듈. 본 RESEARCH 권고: 중앙 `_jit.py` (DRY + import-graph 단순).
- **첫 컴파일 시간 측정 + eager warmup 트리거** — plan-stage benchmark에서 25 kernel 첫 컴파일 누적 시간이 ~16초 이상이면 import-time pre-compile 검토. 본 RESEARCH 실측: gemm_core 단독 cold-start = 640ms → 25개 ≈ 16초로 추정 (병렬화 없음).
- **numba 버전 핀** — 본 RESEARCH 권고: `numba>=0.61.2,<0.66` (numpy 2.x 호환 + cp310-cp312 + manylinux2014). **0.59는 numpy<1.27 핀 → 우리 numpy>=2.0 floor와 충돌, 사용 불가.**
- **`@njit(parallel=True)`** — 25 kernel 모두 small (≤ 1024 elem 일반); 본 RESEARCH 권고: 기본 `parallel=False`로 시작. plan-stage benchmark에서 핫 path만 fine-tune.

### Deferred Ideas (OUT OF SCOPE)

- **engine layer (mm/vec/act_engine) JIT 가속** → v2 (proc/insn pybind11 의존)
- **Cython AOT / C extension / PyPy** → 영구 거부 (PROJECT.md "C++ 추가 코드 금지")
- **fastmath=True / FP 재결합** → 영구 거부 (bit-exact 위반)
- **numba.pycc AOT compile to .so** → 거부 (deprecated)
- **mxe_accum FP32 state numba 통합** → engine layer (D-06 boundary 밖)
- **CUDA / GPU acceleration** → PROJECT.md Out of Scope (v2)
- **asv (airspeed velocity) benchmark suite** → 거부 (pytest-benchmark가 default)
- **`@njit(parallel=True)` 적극 사용** → plan-stage hot path만; v1.x patch
- **PyArrow zero-copy view** → v2
- **`spike[bench]` 별도 extras 분리** → plan-stage discretion
- **per-kernel 정확 numba 옵션 매트릭스 (`nogil`, `boundscheck=False`)** → plan-stage benchmark 후 hot path만
- **GPU acceleration via numba.cuda** → PROJECT.md Out of Scope
- **Multi-process pytest-xdist sweep 병렬** → plan-stage 검토 가능; 단 numba JIT first-call serial-only 주의 필요
</user_constraints>

<phase_requirements>
## Phase Requirements

REQUIREMENTS.md does NOT contain explicit P7 REQ IDs. The following are derived from CONTEXT.md decisions D-01..D-16 — propose 8 P7 REQ IDs for ROADMAP.md insertion + REQUIREMENTS.md `### Numba Optimization (Phase 7)` section.

| ID | Description | Research Support |
|----|-------------|------------------|
| **NJIT-01** | Lazy `from numba import njit` + auto NumPy fallback (`HAS_NUMBA` gate). `pip install spike` (base) → NumPy-only operation; `pip install spike[fast]` → numba acceleration. Both paths must pass full P6 strict-mode regression. | D-02, D-11; §"Lazy Import + Fallback Architecture" — `_jit.py` central shim recommended; verified empirically that `try/except ImportError` pattern works without import side-effects. |
| **NJIT-02** | 25 stateless kernels in `gemm_core.py` (3) + `vec_core.py` (7) + `act_core.py` (15: 7 act + 2 pool + 9 cvt - skip 2 LUT-builders) decorated with `@njit(cache=True)` with same module dual export. Engine layer untouched. | D-06, D-11; §"Exact Kernel Inventory" — counted via grep, 25 confirmed (3+7+18, where FP8 LUT-builders are import-time-only and excluded). |
| **NJIT-03** | `fastmath=False` (numba default) + explicit FP32 Python for-loop preserved. Each kernel takes FP32 arrays in / FP32 (or FP16 / int) arrays out. **FP16 cast happens AT engine boundary, NOT inside @njit** (numba does not support np.float16 on CPU). | D-09; §"FP16 NotImplementedError" — empirical: `@njit` rejects np.float16 args at typing time, must pre-cast to FP32 in caller. |
| **NJIT-04** | Vendor 84-op directory full sweep gate. `tests/gtx/test_regression_fw_full_sweep.py` parametrize over `vendor/gtx_cpp_reference/test/<OP>/n1s16/` (excluding `__pycache__`); strict-mode `compare_hex(strict=True)` must PASS for op directories with `_ref.txt` AND a corresponding compiled `.elf` shipping in `tests/gtx/data/firmware/`. Skip with reason for ops missing either asset. | D-10; §"Vendor Op Directory Inventory" — empirical: 84 ops, all with `_ref.txt`; only 12 .elf currently bundled (P5/P6 lineage), so first sweep gates ~12 op directories with planned expansion to all 84 in subsequent v1.x patches OR dev-stage build of remaining n1s16 .elf (Plan 03 lineage). |
| **NJIT-05** | Per-kernel ULP-0 parity (Tier 1). `tests/gtx/test_njit_parity.py` — 25 kernels × NumPy vs JIT delta_ulp == 0 via `np.array_equal(out.view(np.uint16), out_njit.view(np.uint16))`. **Transcendental kernels (gelu, tanh_act, sigmoid, softmax, esum) require special handling** — see §"Transcendental ULP-0 Drift" for the documented escape (objmode wrapper for tanh/exp). | D-12; §"Transcendental ULP-0 Drift" — empirical: GELU drifts 9/1024, tanh drifts 803/2048 in FP32, fixed by `with objmode` wrapper. |
| **NJIT-06** | Wall-clock 5× walltime acceptance (Tier 3). `tests/gtx/test_njit_perf.py` pytest-benchmark — full vendor 84-op sweep (or shipped subset) walltime is ≤ 1/5 of P6 NumPy baseline. Baseline locked at start of P7 Plan 01 via `pytest tests/gtx/test_regression_fw_full.py --no-cov` measurement against P6 codebase (HAS_NUMBA=False forced). | D-13; §"Empirical Speedup Evidence" — gemm_core measured at ~455× speedup, gives strong margin even after dilution by IO + dispatch overhead. |
| **NJIT-07** | Extras packaging. `pyproject.toml [project.optional-dependencies]` adds `fast = ["numba>=0.61.2,<0.66"]` and `dev = [..., "pytest-benchmark>=4.0", ...]`. cibuildwheel matrix unchanged (extras have zero impact on wheel build). cibuildwheel `test-extras = ["fast"]` added so CI verifies the JIT path on cp310-cp312. | D-03, D-14, D-15; §"numba Compatibility Matrix" — 0.59 INCOMPATIBLE (numpy<1.27); 0.61.2 lower bound matches our numpy 2.x + cp310. |
| **NJIT-08** | Documentation sync. `REQUIREMENTS.md` `Out of Scope` numba항목 재문구 (D-04). `PROJECT.md` "wheel size ≤50MB" 명시 base-wheel 한정 (D-15). `ROADMAP.md` Phase 7 section TBD를 NJIT-01..NJIT-08 + Plans + Success Criteria로 채움. README "Performance acceleration" section 추가 — `pip install spike[fast]` instruction. | D-04, D-15; non-code sync work — typically Plan 05 or first task of Plan 01. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numba | `>=0.61.2,<0.66` | LLVM JIT compiler for numeric Python | Sole choice per D-05; 0.61.2 is minimum that supports `numpy>=2.0` AND `cp310-cp312`. 0.65.x supports through `numpy<2.5`. Latest `0.65.1` (2026-04-23). |
| llvmlite | (auto, dep of numba) | LLVM bindings | Pulled in transitively. `0.44+` for numba 0.61.x; `0.47.x` for numba 0.65.x. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-benchmark | `>=4.0,<6` | JIT speedup measurement (Tier 3 perf gate) | dev extras only. Latest `5.2.3` (2026). pedantic mode + `warmup_iterations` enables first-call exclusion. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numba 0.59 | numba 0.61.2+ | **0.59 BREAKS WITH numpy>=2.0** (pin: `numpy<1.27`). PROJECT requires numpy>=2.0. CONTEXT D-05 said "≥0.59" but plan-stage MUST raise floor to 0.61.2. |
| numba 0.65.1 (latest) | numba 0.61.2+ (range) | 0.65.1 supports numpy 2.0–2.4. Range floor allows users on older numpy; ceiling `<0.66` future-proofs against breaking changes. |
| @njit decorator pattern A (direct: `@njit\ndef fn`) | pattern B (re-call: `fn_njit = njit(...)(fn_numpy)`) | Both work. Pattern B is clearly preferred for D-11 dual export — single source-of-truth function definition + conditional decoration. Pattern A would require `if HAS_NUMBA: @njit\ndef fn else: def fn` which is hard to read and write. |
| pytest-benchmark | airspeed velocity (asv) | asv too heavy (separate config + history DB); pytest-benchmark integrates natively + per-test fixtures. CONTEXT D-14 lock. |
| signature-eager (`@njit("f4[:,:](f4[:,:],f4[:,:])")`) | lazy first-call (D-07) | Eager catches type drift but explodes for FP16/FP32/INT8/INT32 × 1D/2D × 25 kernel matrix. Lazy + D-12 ULP-0 parity test catches drift at acceptance gate. |
| objmode escape for transcendentals | accept ~1 ULP drift | RESEARCH OPEN QUESTION — see §"Transcendental ULP-0 Drift". |

**Installation (proposed pyproject.toml diff):**

```toml
[project.optional-dependencies]
dev = [
  "auditwheel",
  # ... existing entries ...
  "pytest-benchmark>=4.0,<6",  # NEW for P7 Tier 3
]
fast = [
  "numba>=0.61.2,<0.66",
]

[tool.cibuildwheel]
test-extras = ["fast"]  # so CI installs numba and exercises JIT path
test-command = "pytest {project}/tests/gtx -m 'not slow' -x --no-cov"
```

**Version verification (HIGH confidence — verified against PyPI metadata 2026-05-08):**

```
numba 0.59.0  → llvmlite <0.43,>=0.42  → numpy <1.27,>=1.22   ❌ INCOMPATIBLE (numpy<1.27 conflict)
numba 0.59.1  → llvmlite <0.43,>=0.42  → numpy <1.27,>=1.22   ❌ INCOMPATIBLE
numba 0.60.0  → llvmlite <0.44,>=0.43  → numpy <2.1,>=1.22    ⚠️  Partial (numpy 2.0 ok; 2.1+ blocked)
numba 0.61.2  → llvmlite <0.45,>=0.44  → numpy <2.3,>=1.24    ✅  RECOMMENDED FLOOR
numba 0.62.1  → llvmlite <0.46,>=0.45  → numpy <2.4,>=1.22    ✅  Compatible
numba 0.63.1  → llvmlite <0.47,>=0.46  → numpy <2.4,>=1.22    ✅  Locally available + tested
numba 0.65.1  → llvmlite <0.48,>=0.47  → numpy <2.5,>=1.22    ✅  Latest (2026-04-23)
```

All versions ship cp310/cp311/cp312/cp313 manylinux2014_x86_64 wheels. CONTEXT D-05 `≥0.59` floor MUST be raised to `≥0.61.2` (NJIT-07).

## Architecture Patterns

### Recommended Project Structure

```
src/main/python/riscv/gtx/
├── _jit.py             # NEW: HAS_NUMBA detection + njit shim (passthrough fallback)
├── gemm_core.py        # MOD: 3 kernels — _kernel_numpy + njit wrap
├── vec_core.py         # MOD: 7 kernels — same dual-export pattern
├── act_core.py         # MOD: 18 kernels (7 act + 2 pool + 9 cvt) — same; LUT-builders untouched
├── mm_engine.py        # UNCHANGED: pre-cast FP16 → FP32 before calling kernel; FP16 cast on result
├── vec_engine.py       # UNCHANGED: same boundary discipline
├── act_engine.py       # UNCHANGED: same boundary discipline
├── ...                 # rest unchanged
tests/gtx/
├── test_njit_parity.py             # NEW: Tier 1 — 25 × NumPy vs JIT ULP-0
├── test_njit_perf.py               # NEW: Tier 3 — pytest-benchmark
├── test_regression_fw_full_sweep.py # NEW: Tier 2 — 84-op vendor sweep with skip
├── test_regression_fw_full.py      # MOD: keep as P6 sentinel; sweep is sibling
└── ...                             # rest unchanged
scripts/
├── import_vendor_golden.py # MOD: extend 9-op map → 84-op map
pyproject.toml          # MOD: [project.optional-dependencies] fast = + dev += pytest-benchmark; [tool.cibuildwheel] test-extras = ["fast"]
.planning/REQUIREMENTS.md # MOD: Out of Scope row reword (D-04 / NJIT-08)
.planning/PROJECT.md    # MOD: 50MB cap → "base wheel size ≤50MB" (D-15 / NJIT-08)
.planning/ROADMAP.md    # MOD: Phase 7 section fill (NJIT-01..NJIT-08)
README.md               # MOD: add "Performance" section with `pip install spike[fast]`
```

### Pattern 1: Lazy Import + Auto Fallback (`_jit.py`)

**What:** Centralized HAS_NUMBA detection + a `njit` shim that preserves call site syntax `@njit(cache=True)` regardless of numba install state.

**When to use:** EVERY core module imports `from ._jit import njit, HAS_NUMBA` instead of importing numba directly. Single source-of-truth for the fallback decision.

**Example (RECOMMENDED for D-02 / NJIT-01):**

```python
# src/main/python/riscv/gtx/_jit.py
"""Lazy numba shim. P7 D-02 / NJIT-01 single source-of-truth.

Allows P7 hot kernels to write `@njit(cache=True)` uniformly. When numba is
not installed (base `pip install spike`), `njit` becomes a no-op decorator
that returns the wrapped function unchanged — kernels run as pure NumPy
(P4/P5 lineage).
"""
from __future__ import annotations
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

try:
    from numba import njit as _real_njit  # type: ignore[import-not-found]
    HAS_NUMBA: bool = True

    def njit(*args: Any, **kwargs: Any) -> Any:
        return _real_njit(*args, **kwargs)

except ImportError:  # pragma: no cover -- exercised when `spike[fast]` not installed
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        # Two call patterns:
        #   @njit                      -> args=(fn,), kwargs={}            -> return fn
        #   @njit(cache=True)          -> args=(), kwargs={'cache': True}  -> return decorator
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def decorator(fn: F) -> F:
            return fn
        return decorator
```

**Test (parametrize over both paths via env var):**

```python
# tests/gtx/test_jit_shim.py (Plan 01 GREEN)
def test_njit_shim_passthrough_when_no_numba(monkeypatch):
    """When HAS_NUMBA=False the shim returns the function unchanged."""
    # Force shim to fallback path
    import importlib, sys
    monkeypatch.setattr(sys, "modules", {**sys.modules, "numba": None})
    # ... reload _jit, assert decorator returns fn ...
```

### Pattern 2: Dual-Export Decoration via Re-Call (D-11)

**What:** Each core module defines `_kernel_impl(...) -> ...` (the source-of-truth NumPy implementation, untouched from P4/P5), then conditionally creates the JIT-decorated alias and binds the public name.

**When to use:** All 25 kernels. Re-call pattern (Pattern B from CONTEXT) is preferred over inline `@njit` decorator because it preserves the P4/P5 source unchanged and centralizes the binding decision.

**Example (CONTEXT specifics §"Option B"):**

```python
# src/main/python/riscv/gtx/gemm_core.py (after P7 mod)
"""Pure stateless GEMM kernel. P7 numba @njit boundary."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ._jit import njit, HAS_NUMBA  # NEW import


def _gemm_core_impl(
    A_f32: NDArray[np.float32],   # CHANGED: FP32 in (was FP16); caller pre-casts
    B_f32: NDArray[np.float32],
    has_bias: bool,                # CHANGED: positional bool (numba does not allow kwonly)
    bias_fp32: NDArray[np.float32],
) -> NDArray[np.float32]:           # CHANGED: FP32 out (was FP16); caller post-casts
    """C = A @ B [+ bias_fp32]  ->  FP32 (caller casts to FP16). Explicit 3-loop.

    Direct port of gtx_npu_mm.cc:27-94. P7 boundary requires FP32 in/out
    because numba CPU target does not support np.float16 (verified empirically
    2026-05-08; see RESEARCH §"FP16 NotImplementedError").
    """
    M, K = A_f32.shape
    _K2, N = B_f32.shape
    C_f32 = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            s = np.float32(0.0)
            for k in range(K):
                s += A_f32[i, k] * B_f32[k, j]
            C_f32[i, j] = s
    if has_bias:
        C_f32 = C_f32 + bias_fp32  # broadcast FP32 add
    return C_f32


# D-11 dual export — same surface, conditional acceleration.
_gemm_core_njit = njit(cache=True)(_gemm_core_impl)


def gemm_core(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    *,
    has_bias: bool = False,
    bias_fp32: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float16]:
    """Public API — preserves P4 signature (FP16 in/out + kwargs).

    Bridges to the @njit-friendly `_gemm_core_impl` by pre-casting to FP32 and
    post-casting the result. Validates `bias_fp32` shape/dtype (cannot raise
    ValueError inside @njit cleanly).
    """
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    M, K = A_f32.shape
    K2, N = B_f32.shape
    if K != K2:
        raise ValueError(f"shape mismatch: A is (M={M}, K={K}), B is (K={K2}, N={N})")
    if has_bias:
        if bias_fp32 is None:
            raise ValueError("has_bias=True requires bias_fp32 ndarray")
        if bias_fp32.shape != (M, N):
            raise ValueError(f"bias_fp32 shape {bias_fp32.shape} != C shape ({M}, {N})")
        if bias_fp32.dtype != np.float32:
            raise TypeError(f"bias_fp32 dtype must be float32, got {bias_fp32.dtype}")
        bias = bias_fp32
    else:
        bias = np.zeros((M, N), dtype=np.float32)
    C_f32 = _gemm_core_njit(A_f32, B_f32, has_bias, bias)
    return C_f32.astype(np.float16)
```

**Why this shape:**
1. `_gemm_core_impl` is the JIT-compatible kernel. It takes FP32 (numba can't do FP16). It does no validation (numba doesn't tolerate dynamic ValueError fluently). It does no kwargs (numba prefers positional).
2. `gemm_core(A: FP16, B: FP16, ...)` is the public API — preserves P4 caller surface. Validates inputs. Casts FP16→FP32 + FP32→FP16 at the boundary. Calls `_gemm_core_njit`.
3. `_gemm_core_njit = njit(cache=True)(_gemm_core_impl)` — when HAS_NUMBA=False, this is identity (Pattern 1 shim returns the function); when HAS_NUMBA=True, this is the compiled JIT version.
4. Old call sites in `mm_engine.py` continue to work — they call `gemm_core(A_fp16, B_fp16, has_bias=..., bias_fp32=...)` exactly as before.

### Pattern 3: FP8 LUT Module-Level Capture

**What:** `act_core.FP8_TO_FP16_LUT` and `FP16_TO_FP8_LUT` already built at module import. numba @njit kernels capture these as constants — VERIFIED EMPIRICALLY: numba allows module-level numpy array capture in @njit (both fancy index and scalar index work).

**Important caveat from §"numba cache invalidation":** module-level globals are FROZEN at compile time. If `_build_fp8_to_fp16_lut()` is changed after a cached compile, the cache must be manually cleared (`rm -rf src/main/python/riscv/gtx/__pycache__`) — numba does NOT auto-invalidate.

**Example (cvt_qh / cvt_hq pattern):**

```python
# src/main/python/riscv/gtx/act_core.py (after P7 mod)
from ._jit import njit, HAS_NUMBA

# UNCHANGED — module-level LUT, ~30ms one-shot at import (P5 D-15 lineage)
FP8_TO_FP16_LUT: np.ndarray = _build_fp8_to_fp16_lut()
FP16_TO_FP8_LUT: np.ndarray = _build_fp16_to_fp8_lut()


def _cvt_qh_impl(arr_fp16_as_uint16: NDArray[np.uint16],
                  scale_f32: np.float32, offset_f32: np.float32) -> NDArray[np.uint8]:
    """FP16 -> FP8 with scale/offset. Caller passes FP16 as uint16 view.
    NB: cannot accept np.float16 in @njit — this is the FP16 NotImplementedError workaround.
    """
    n = arr_fp16_as_uint16.shape[0]
    # Reconstruct as FP32 for arithmetic (we don't have FP16 type in @njit)
    # Trick: caller does the FP32 conversion, but we need FP16 bit pattern for LUT.
    # See plan-stage discussion — may need to leave cvt_qh as pure NumPy.
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        # ... use FP16_TO_FP8_LUT[idx] ...
    return out

_cvt_qh_njit = njit(cache=True)(_cvt_qh_impl)
```

**OPEN QUESTION for plan-stage:** cvt_qh/cvt_hq currently use `arr.view(np.uint16).astype(np.intp)` for fancy LUT index, applied to FP16 arrays. Translating this to @njit-compatible form requires the engine to pass `arr.view(np.uint16)` (uint16 buffer) instead of FP16. This is a SIGNATURE CHANGE on `act_engine.py` callers. May be cleaner to leave the 9 cvt kernels as NumPy (they are already vectorized via fancy index, and numba speedup on a single fancy-index lookup is minimal).

**Recommendation:** Plan 03 should benchmark each cvt kernel under @njit and skip JIT for any that don't yield ≥2× per-call speedup. The cvt kernels are not in the GEMM hot path; vendor regression sweep walltime is dominated by gemm + softmax + esum.

### Pattern 4: Transcendental Workaround (objmode — RECOMMENDED)

**What:** The 5 transcendental kernels (`gelu`, `tanh_act`, `sigmoid`, `softmax`, `esum`) cannot be bit-exact under default `@njit(fastmath=False)` because LLVM's `tanhf`/`expf` intrinsics differ from glibc's `tanhf`/`expf` by ~1 ULP. The fix is `with objmode(...)` to call back into NumPy for the transcendental section.

**When to use:** Only the 5 transcendental kernels. Rest stay native @njit.

**Example (gelu — verified empirically gives 0/2048 ULP-0 mismatches):**

```python
def _gelu_impl(arr_f32: NDArray[np.float32]) -> NDArray[np.float32]:
    """GELU(arr) = 0.5*arr*(1 + tanh(sqrt(2/pi)*(arr + 0.044715*arr^3))).

    Uses objmode for tanh — see RESEARCH §"Transcendental ULP-0 Drift":
    LLVM tanh intrinsic differs from glibc by ~1 ULP, propagating to FP16
    ULP-0 mismatch in 9/1024 random samples. objmode escape hands tanh
    back to NumPy (libm) — bit-exact.
    """
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    inner = sqrt_2_over_pi * (arr_f32 + np.float32(0.044715) * arr_f32 * arr_f32 * arr_f32)
    # Numba escape: NumPy libm tanh
    out = np.empty_like(inner)
    with objmode(t='float32[:]'):
        t = np.tanh(inner)  # noqa: NPY002 -- intentional libm route
    return np.float32(0.5) * arr_f32 * (np.float32(1.0) + t)

_gelu_njit = njit(cache=True)(_gelu_impl)
```

**Performance impact (estimate):** objmode entry/exit + Python ↔ C handoff is ~1-5 μs per call. For a single transcendental section it's marginal. Net speedup vs pure-Python NumPy still expected to be 50-100× for kernels with non-trivial element counts (vec_core.dot 1024-elem, etc.). plan-stage benchmark MUST verify.

**Source:** [numba objmode documentation](https://numba.readthedocs.io/en/stable/user/withobjmode.html) — supported since 0.40.x, stable in modern numba.

### Anti-Patterns to Avoid

- **`np.float16` arg to @njit** — fails with `NotImplementedError: float16` at typing time. Caller MUST pre-cast to FP32 before calling JIT kernel; engine post-casts.
- **`@njit(fastmath=True)`** — bit-exact violation (CONTEXT D-09 lock).
- **`@njit("f4[:](f4[:])")` eager signature** — type explosion across 25 kernels (CONTEXT D-07 lock).
- **`@njit(parallel=True)` on small kernels** — thread spawn cost > kernel work for ≤1024-elem kernels (most of vec_core/act_core). Reserve for explicitly profiled hot paths.
- **`@njit(nogil=True)`** — pyspike runs single-hart firmware; no GIL contention to release. No benefit + cache complexity.
- **`np.dot` / `np.matmul` / `np.einsum` / `np.sum`(FP16) inside @njit** — numba's BLAS dispatch differs from NumPy's. CONTEXT D-09 forbids, applies inside @njit too. Use explicit `for i: ... s += A[i]*B[i]`.
- **Module-level mutable arrays besides FP8 LUTs** — frozen into cache; if `_jit.py` or `act_core.py` swaps array contents post-import, cached @njit kernels see stale values.
- **Per-call `with objmode` outside the 5 transcendental kernels** — defeats JIT entire purpose.
- **`@njit` on `_build_fp8_to_fp16_lut` / `_build_fp16_to_fp8_lut`** — they run once at import, not on hot path; @njit just adds 1-second compile delay to import.
- **Custom `signatures=[...]` list to pre-compile** — increases import time, lazy first-call (D-07) is the documented strategy.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLVM JIT compiler | Custom AST → IR walker | numba | numba is the chosen library (D-05). Reinventing = months of work + LLVM expertise. |
| FP16 ↔ FP32 conversion inside @njit | Manually shift uint16 bits | Pre-cast in caller via `arr.astype(np.float32)` | numba CPU target has no fp16 type; arithmetic on uint16-as-bits would diverge from IEEE 754. CONTEXT D-09 forbids. |
| Bit-exact tanh for FP32 | LUT-based or polynomial-approximated tanh | `with objmode(...): t = np.tanh(...)` | NumPy delegates to glibc libm which we ALREADY trust (P4/P5 oracle source). Re-implementing tanh = high risk of new drift sources. |
| Disk cache for compiled kernels | Custom `pickle` of LLVM bitcode | `@njit(cache=True)` | numba 0.59+ ships with mature on-disk cache (`.nbi` index + `.nbc` overload). CONTEXT D-08 lock. |
| JIT speedup measurement | `time.perf_counter()` ad-hoc | pytest-benchmark | pytest-benchmark handles warmup, calibration rounds, statistical analysis. CONTEXT D-14. |
| FP16 / FP32 / INT8 / INT32 type signature | Eager signature listing | Lazy first-call dispatch | numba auto-infers correctly per CONTEXT D-07; type drift caught by D-12 ULP-0 parity. |
| Wheel base+extras separation | Two separate wheels published | `[project.optional-dependencies]` with `[fast]` extras | PyPI / pip standard. cibuildwheel zero-impact. |
| Vendor 84-op asset import | Hand-curate per op | Extend `scripts/import_vendor_golden.py` (P6 P03) — 9-op map → 84-op map | The script is already proven for the 9-op subset; growth is mechanical (filename pattern matches `n1s16_<lower(op)>_ref.txt`). |

**Key insight:** The numba ecosystem in 2026 is mature; every D-01..D-16 decision has a documented out-of-the-box answer. The non-trivial discoveries are (a) the FP16 NotImplementedError, (b) the transcendental ULP drift, (c) the numba 0.59 → 0.61.2 floor lift. All three are PHASE-PLANNING-CRITICAL but each has a clean documented fix.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | None — Phase 7 adds zero database/datastore writes. mxe_accum FP32 state (`npu._mxe_accum`) is a per-process ndarray, not persistent. No external DB. | None — verified by reading `npu.py` and grep `database\|sqlite\|redis\|chromadb\|mongo` over the codebase. |
| **Live service config** | None — pyspike is a CLI tool, not a server. No Datadog / n8n / Tailscale / Cloudflare touchpoints. | None — verified explicit: pyspike has no service registration. |
| **OS-registered state** | None for the user; **CI/dev machines may have stale numba cache directories** at `src/main/python/riscv/gtx/__pycache__/` (already exists from Python imports, will be where `.nbi`/`.nbc` are written). On first `pip install spike[fast]` + first run, numba auto-creates these — but if the user later `pip uninstall numba` and re-runs the tests, the cache files remain orphaned (harmless but visible). | Plan 01 task: add `*.nbi` and `*.nbc` to `.gitignore` if not already present. **Important:** dev workflow caveat — when `act_core._build_fp8_to_fp16_lut()` is modified, devs MUST `rm -rf src/main/python/riscv/gtx/__pycache__/` to invalidate cached @njit kernels that captured the old LUT (numba caching does not detect cross-module symbol changes). |
| **Secrets and env vars** | numba reads several env vars at import: `NUMBA_DISABLE_SVML`, `NUMBA_DEBUG`, `NUMBA_NUM_THREADS`, `NUMBA_CPU_NAME` (manylinux2014 fallback), `NUMBA_THREADING_LAYER`. **None are required** — defaults work for our case. We set NONE in code. Existing env vars: `GTX_DDR_DUMP*`, `GTX_NO_EXIT`, `GTX_DDR_REVERSED`, `RISCV` — UNCHANGED. | None — verified no new env vars introduced by P7. |
| **Build artifacts / installed packages** | (1) numba's compile cache `__pycache__/<module>.<func>-<hash>.nbi` + `.nbc` — runtime, regenerated lazily. (2) Wheel build does NOT bundle numba itself — only declared as optional dep. (3) `pyspike-verify` console script (P6 D-02) — UNCHANGED. (4) cibuildwheel config additions: `[tool.cibuildwheel] test-extras = ["fast"]` — verifies JIT path on CI. | Plan 05 (or first-task of any plan): verify `.gitignore` covers `__pycache__/*.nbi`, `__pycache__/*.nbc`. |

**Nothing found in stored data, live service config, secrets:** Verified explicitly. P7 is a pure code/extras/test-data extension; no runtime state surface change.

**The canonical question — what runtime state has the old name cached after every file is edited?** Phase 7 does NOT rename anything. It adds a new module (`_jit.py`), modifies 3 core modules to add `@njit`-decorated aliases, adds 3 test files, and edits `pyproject.toml` + 3 `.planning/` docs. No string rename ⇒ no orphan caches at the application level. The only caching concern is numba's own compile cache (covered above).

## Common Pitfalls

### Pitfall 1: numba CPU target does NOT support np.float16

**What goes wrong:** Any `@njit` kernel that takes an `np.float16` argument fails at typing time with `NotImplementedError: float16`. Even just reading `arr.shape[0]` on a float16 array fails. The kernel cannot view the array as `uint16` either inside @njit — view itself fails.

**Why it happens:** numba's CPU compilation pipeline never implemented the `float16` type. It's been an open issue since 2019 ([numba#4402](https://github.com/numba/numba/issues/4402)). 2022 follow-up [numba#8138](https://github.com/numba/numba/issues/8138) confirms still unsupported. Empirical verification 2026-05-08 with numba 0.63.1: `NotImplementedError: float16`.

**How to avoid:** All FP16 ↔ FP32 casts happen at the engine layer (caller of the kernel), NEVER inside the @njit kernel. Specifically:
- Engine code: `kernel_njit(A.astype(np.float32), B.astype(np.float32))` then `result_fp32.astype(np.float16)`.
- Kernel signature: only `float32`, `int8`, `int32`, `uint8`, `uint16`. **NOT** `float16`.
- This is a SIGNATURE CHANGE for the public kernels — the public `gemm_core(A: FP16, B: FP16) -> FP16` API stays the same (cast bridges are inside the public function), but the @njit'd `_gemm_core_impl` takes FP32. See Pattern 2 above.

**Warning signs:** Plan stage RED scaffold: `with pytest.raises(NotImplementedError, match='float16'): gemm_core_njit(np.zeros(5, dtype=np.float16))` — proves the kernel shape rejects FP16 directly. Plan stage GREEN: caller indirection bridges FP16.

**Verified empirically — 2026-05-08:**
```
>>> @njit(cache=False)
>>> def take(arr): return arr.shape[0]
>>> take(np.zeros(5, dtype=np.float16))
NotImplementedError: float16  # at typing.py
```

### Pitfall 2: Transcendental ULP-0 Drift (PHASE-CRITICAL)

**What goes wrong:** D-12 strict per-kernel ULP-0 parity (`np.array_equal(out_numpy.view(np.uint16), out_njit.view(np.uint16))`) **WILL FAIL** for the 5 transcendental kernels: `gelu`, `tanh_act`, `sigmoid`, `softmax`, `esum`.

**Empirical evidence (2026-05-08, numba 0.63.1, numpy 2.2.6, x86_64 glibc):**
```
np.tanh on FP32 array (2048 random) -- numba @njit vs NumPy:
  ULP-0 mismatches: 803/2048 (39%)
  Max abs diff:    1.19e-7

GELU full kernel FP16 output (1024 random):
  ULP-0 mismatches: 9/1024 (0.9%)

GELU using math.tanh inside @njit:
  Mismatches:      442/2048 FP32, 28/2048 FP16  (worse than np.tanh path)

NUMBA_DISABLE_SVML=1:
  np.tanh mismatches: 803/2048 (UNCHANGED — SVML is not the cause)

GELU using `with objmode`(tanh path returns to NumPy/libm):
  ULP-0 mismatches: 0/2048 FP32, 0/1024 FP16  ✓ BIT-EXACT
```

**Why it happens:** numba lowers `np.tanh(arr_f32)` to LLVM's `tanhf` intrinsic. LLVM's `tanhf` is implemented via approximation polynomials that differ from glibc's `tanhf` by at most 1 ULP at certain inputs. NumPy outside numba calls glibc's `tanhf` directly via the libm CPython binding. Result: `np.tanh(x)` returns different bit patterns depending on whether it was JITed.

The ROOT cause is in the LLVM `Intrinsic::tanh` IR — independent of numba's `fastmath` flag, independent of SVML availability. Lock-in: this is a fundamental numba implementation detail and will NOT be fixed unless numba switches to libm calls in nopython.

**Affected kernels (5 of 25 — flag explicitly):**
- `act_core.gelu` — uses `np.tanh`
- `act_core.tanh_act` — uses `np.tanh` directly
- `act_core.sigmoid` — uses `np.exp`
- `act_core.softmax` — uses `np.exp` (and the for-loop sum, which IS bit-exact under @njit)
- `act_core.esum` — uses `np.exp` element-wise

**Three options for plan-stage / user decision:**

| Option | Description | D-12 Impact | Perf Impact |
|--------|-------------|-------------|-------------|
| **A** | Accept ~1 FP16 ULP drift on these 5 kernels. Relax D-12 to "ULP ≤ 1, atol ≤ 0.001" for `{gelu, tanh_act, sigmoid, softmax, esum}` only. Other 20 kernels remain ULP-0. | D-12 partial relaxation; document in NJIT-05. | Best (full @njit speedup). |
| **B** | (RECOMMENDED) Use `with objmode(t='float32[:]'): t = np.tanh(...)` inside the 5 kernels. Falls back to NumPy libm during the transcendental call only. | D-12 fully preserved (verified 0/2048 mismatch). | Slightly degraded (~30-50% slower than full @njit on these 5), but still 50-100× over pure Python. |
| **C** | Skip @njit entirely for the 5 transcendental kernels. They stay pure NumPy. | D-12 fully preserved. | Lower aggregate speedup; vendor sweep walltime depends on transcendental fraction. |

**Recommendation:** Option B. RESEARCH-verified bit-exact via `with objmode`. plan-stage Plan 04 (act_core wrapping) explicitly chooses Option B and documents the objmode shim per kernel. Option A is the second choice — D-12 relaxation for 5/25 kernels would surface as a single test parametrize annotation (`@pytest.mark.parametrize_strict_kernels` excluding 5 names) and reduce dev complexity.

**Warning signs:** plan-stage benchmark must include `test_njit_parity[gelu]`, `test_njit_parity[tanh_act]`, `test_njit_parity[sigmoid]`, `test_njit_parity[softmax]`, `test_njit_parity[esum]` early in the wave. If any of these fail, planner must implement Option B before continuing.

### Pitfall 3: numba 0.59 BREAKS WITH numpy>=2.0

**What goes wrong:** CONTEXT D-05/D-03 says `numba>=0.59`. Empirical PyPI metadata check (2026-05-08): `numba 0.59.x` declares `Requires-Dist: numpy <1.27,>=1.22`. Project's `pyproject.toml` declares `numpy>=2.0,<3` (Phase 1 D-07 lock). pip resolver will FAIL: `ResolutionError: numba==0.59.0 incompatible with numpy>=2.0,<3`.

**Why it happens:** numba's numpy support follows NEP 29 / SPEC 0 with a 2-version delay. numba 0.59 (Jan 2024) was the last release before numpy 2.x (June 2024). numba 0.60.0 added partial numpy 2.0 support; 0.61.2 (`numpy<2.3,>=1.24`) is the first to comfortably overlap with our `numpy>=2.0` floor.

**How to avoid:** Plan stage Plan 01 first task — pyproject.toml `[project.optional-dependencies] fast = ["numba>=0.61.2,<0.66"]`. CONTEXT D-05 floor is updated as part of NJIT-07.

**Compatibility table (verified PyPI metadata 2026-05-08):**

| numba | numpy floor | numpy ceiling | llvmlite | Notes |
|-------|-------------|----------------|----------|-------|
| 0.59.x | 1.22 | <1.27 | 0.42.x | ❌ INCOMPATIBLE — numpy 2.0 blocked |
| 0.60.0 | 1.22 | <2.1 | 0.43.x | ⚠️ partial — only numpy 2.0.x |
| 0.61.2 | 1.24 | <2.3 | 0.44.x | ✅ recommended floor |
| 0.62.1 | 1.22 | <2.4 | 0.45.x | ✅ |
| 0.63.1 | 1.22 | <2.4 | 0.46.x | ✅ (locally tested) |
| 0.65.1 | 1.22 | <2.5 | 0.47.x | ✅ latest, recommended ceiling |

### Pitfall 4: Module-Level Global Capture and Cache Invalidation

**What goes wrong:** `act_core.FP8_TO_FP16_LUT` and `FP16_TO_FP8_LUT` are module-level numpy arrays. @njit kernels that reference these capture the CURRENT VALUE at compile time. If `_build_fp8_to_fp16_lut()` is later modified to fix a bug, but the user has a stale `__pycache__/act_core._cvt_qh_impl-<hash>.nbi`, the cached compiled function uses the OLD LUT — silently giving wrong results.

**Why it happens:** numba's caching is conservative — it only invalidates when the source file containing the @njit function changes. It cannot detect cross-module symbol changes (i.e., `act_core.py` LUT-builder logic change won't invalidate cached `act_core.cvt_qh_njit`).

**How to avoid:**
1. **CI**: Always start CI runs from a clean `__pycache__/` (cibuildwheel does this implicitly — fresh manylinux2014 container per build).
2. **Dev workflow**: When `act_core._build_fp8_to_fp16_lut` or `_build_fp16_to_fp8_lut` is modified, `rm -rf src/main/python/riscv/gtx/__pycache__/` before running tests. Document this in `tests/gtx/CLAUDE.md` (if exists) or in `act_core.py` module docstring.
3. **Defense**: Plan 04 GREEN test must include `test_fp8_lut_cache_invalidation_after_rebuild` — modifies the LUT in-place, calls `cvt_qh_njit`, asserts result reflects the modified LUT (proves no stale cache). If this test fails, the dev knows.

**Warning signs:** Tier 1 ULP-0 parity tests pass on first run, then start failing on a teammate's machine. Resolution: `rm -rf __pycache__/` and re-run — if test passes, it was a cache invalidation issue.

### Pitfall 5: First-Call Compile Time + pytest-benchmark Noise

**What goes wrong:** Tier 3 `test_njit_perf.py` measures wall-clock. First call of every JIT kernel includes LLVM compile time (~600ms per kernel). 25 kernels × 600ms = ~16 seconds aggregate first-call time. pytest-benchmark by default includes the first call in the warmup phase, but warmup defaults are CALIBRATED (run more iterations until stable) — for a JIT'd kernel where iteration 0 is 600ms and iteration 1 is 2μs, calibration sees the bimodal and may emit garbage statistics or a `Warning: timer too slow` complaint.

**How to avoid (verified pattern from CONTEXT specifics):**

```python
# tests/gtx/conftest.py (or test_njit_perf.py)
@pytest.fixture(scope='session')
def warmed_kernels():
    """Pre-compile all 25 @njit kernels in a session-scope fixture so
    individual benchmark tests start with cache-hit cold path.
    """
    from riscv.gtx.gemm_core import _gemm_core_njit, _gemm_reduce_sum_a_njit, _gemm_dot_njit
    # ... import all 25 ...
    # Touch each kernel with a small representative input
    A = np.zeros((4, 4), dtype=np.float32)
    B = np.zeros((4, 4), dtype=np.float32)
    bias = np.zeros((4, 4), dtype=np.float32)
    _gemm_core_njit(A, B, False, bias)
    # ... 24 more ...
    return None  # fixture is just for side effect

@pytest.mark.benchmark(group='gemm')
def test_gemm_perf(benchmark, warmed_kernels):
    # benchmark sees a JIT-warm kernel, so iteration 0 == steady-state
    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    bias = np.zeros((16, 16), dtype=np.float32)
    benchmark(_gemm_core_njit, A, B, False, bias)
```

**Alternative (pedantic mode):** `benchmark.pedantic(fn, args=..., warmup_rounds=1, rounds=100)` — explicit warmup of 1 round before measurement. Either pattern works; session-scope fixture is cleaner (compile happens once across the whole pytest session).

**Warning signs:** pytest-benchmark output shows `mean = 600ms, std = 60ms` for a kernel that should be µs. Resolution: add session-scope warmup fixture.

### Pitfall 6: Engine Layer Import Order and Caching

**What goes wrong:** If `mm_engine.py` imports `from .gemm_core import gemm_core` (the public FP16 wrapper), and `gemm_core.py` imports `from ._jit import njit` which performs `try: from numba import njit; HAS_NUMBA = True`, then the import order is:
```
riscv.gtx.__init__ → mm_engine → gemm_core → _jit → numba.__init__ (compiles llvmlite, ~200ms)
```
This adds ~200ms to every `import riscv.gtx` even when no JIT kernel is called. For users who only use `from riscv.gtx import GtxNpu` to set up a simulation, this is dead-weight import time.

**How to avoid:** `_jit.py` does the `try: from numba import njit` lazily, ONLY at the FIRST use. Two patterns:
- **(A)** module-load-time import (current sketch in Pattern 1): immediate, accepts the 200ms cost.
- **(B)** lazy module-level proxy: `numba` is imported on first call to `njit(...)`. More complex but zero import-time cost when not used.

**Recommendation:** Pattern (A). 200ms one-time at import is acceptable for the JIT user (`pip install spike[fast]` users). Base `pip install spike` users pay zero (numba import fails fast in `_jit.py` ImportError branch, ~1ms).

### Pitfall 7: Vendor Op Asset Naming Drift

**What goes wrong:** D-10 vendor 84-op sweep depends on `n1s16_<op_name>_ref.txt` filename pattern. Spot check shows naming inconsistencies:
- `RELU/n1s16_relu_ref.txt` ✓
- `ADD/n1s16_add_vv_ref.txt` (NOT `n1s16_add_ref.txt`)
- `SOFT_MAX/n1s16_softmax_ref.txt` (NOT `n1s16_soft_max_ref.txt`)

**Existing handler (P6 P03):** `scripts/import_vendor_golden.py` already covers 9 ops with explicit `VENDOR_TO_PYSPIKE_OPS` mapping. Plan 03 must extend this to all 84 ops. Each op needs a manual filename verification.

**How to avoid:** Plan 03 task 1: write `scripts/audit_vendor_assets.py` that walks all 84 op directories, lists actual `_ref.txt` filenames, and emits a starter `VENDOR_TO_PYSPIKE_OPS` dict for review. Manual review then locks the names in.

**Warning signs:** First sweep run reports `pytest.skip("vendor file missing: n1s16_<wrong_name>_ref.txt")` for ops we KNOW have data. Resolution: audit script confirms actual filename, fix mapping.

### Pitfall 8: 84 Ops Have Refs but Only 12 .elf Are Bundled

**What goes wrong:** D-10 reads the user's intent as "test/{OP} 103개 모두 통과 (데이터가 없는 경우 skip)". RESEARCH measures: vendor has 84 op directories (NOT 103) — every one with `_ref.txt`. But the project ships only 12 `.elf` firmware binaries in `src/main/python/riscv/gtx/data/firmware/` (P5/P6 lineage: abs, activation_relu_gelu, add_vv, leaky_relu, mm_basic, mul_vv, nop_wjoin, relu, sigmoid, softmax, sum, tanh).

**Implication:** The 84-op sweep needs 84 .elf binaries. Building these from `vendor/gtx_cpp_reference/test/<OP>/n1s16/n1s16_<op>.c` requires the RISC-V cross-toolchain (`/opt/riscv/`). This is dev-stage-only (P6 D-12 lineage). Two paths:

**Path A (recommended):** Plan 03 dev-stage builds all 84 .elf at once using `vendor/gtx_cpp_reference/test/run_tests_n1s16.sh` flow, lock-in to `tests/gtx/data/elf/` + ship in wheel. Wheel asset growth: 84 × ~1.3KB = ~110KB. Negligible vs 50MB cap.

**Path B:** P7 sweep gates only the 12 already-bundled .elf and skips the other 72 with `pytest.skip("elf missing — see v1.x patch ROADMAP")`. Significantly weakens D-10 acceptance gate. Defer to v1.x.

**Recommendation:** Path A. Plan 03 includes a Wave 0 task: "RISC-V toolchain setup verification + bulk .elf build" — fails fast if `/opt/riscv/` missing (Wave 0 RED → developer environment fix → GREEN). The wheel is shipped with all 84 .elf. CI verifies wheel-installed sweep passes on cp310/cp311/cp312.

### Pitfall 9: cibuildwheel test-extras + Caches in Container

**What goes wrong:** `[tool.cibuildwheel] test-extras = ["fast"]` triggers `pip install <wheel>[fast]` in the manylinux2014 container, which downloads numba (~10MB) + llvmlite (~30MB) + dependencies. First test run inside container compiles all 25 @njit kernels (~16s). manylinux2014 base image has limited cache, and the container is destroyed after the build — so the compile cost is paid every CI run.

**Mitigation:** Acceptable for v1.0 (CI runs are infrequent vs dev local runs). plan-stage may add `pip install numba==X` to `[tool.cibuildwheel.linux] before-test` if compile time becomes painful.

## Code Examples

Verified patterns:

### Example 1: Sequential FP32 Accumulator — Bit-Exact Verified

```python
# Source: empirical test 2026-05-08; matches gemm_core.py line 76-82 + vec_core.py line 90-93
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def vsum_fp32_seq(arr_f32):
    """FP32 sequential accumulator, single-thread, no SIMD reorder.
    BIT-EXACT vs Python `for x in arr: s += np.float32(x)` (verified 2026-05-08).
    """
    s = np.float32(0.0)
    for i in range(arr_f32.shape[0]):
        s += arr_f32[i]
    return s

# Empirical: 1000-element random — bit-exact match in 0x419aa810
```

### Example 2: GELU with objmode Transcendental Escape (RECOMMENDED for D-12)

```python
# Source: RESEARCH §"Transcendental ULP-0 Drift" — verified 0/1024 FP16 mismatches
import numpy as np
from numba import njit, objmode

@njit(cache=True, fastmath=False)
def _gelu_impl(arr_f32):
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    inner = sqrt_2_over_pi * (arr_f32 + np.float32(0.044715) * arr_f32 * arr_f32 * arr_f32)
    out = np.empty_like(inner)
    with objmode(t='float32[:]'):
        t = np.tanh(inner)  # libm path — bit-exact
    return np.float32(0.5) * arr_f32 * (np.float32(1.0) + t)
```

### Example 3: Module-Level LUT Capture (FP8 codec — verified compiles + works)

```python
# Source: verified pattern; matches act_core.cvt_hq usage of FP8_TO_FP16_LUT
import numpy as np
from numba import njit

LUT_FP8_TO_FP32 = np.zeros(256, dtype=np.float32)  # populated by import-time builder
# ... populate ...

@njit(cache=True)
def decode_fp8(arr_uint8):
    """Fancy index into module-level LUT — VERIFIED 2026-05-08 it compiles + runs."""
    return LUT_FP8_TO_FP32[arr_uint8]
```

### Example 4: Lazy Import + Fallback Shim (`_jit.py`)

```python
# Source: RESEARCH §"Pattern 1: Lazy Import + Auto Fallback"
# This shim should be the SINGLE source-of-truth for HAS_NUMBA detection.
from __future__ import annotations
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

try:
    from numba import njit as _real_njit
    HAS_NUMBA: bool = True

    def njit(*args, **kwargs):
        return _real_njit(*args, **kwargs)

except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def decorator(fn): return fn
        return decorator
```

### Example 5: pytest-benchmark with JIT Warmup Fixture

```python
# Source: RESEARCH §"Pitfall 5"
import numpy as np
import pytest

@pytest.fixture(scope='session')
def warmed_kernels():
    from riscv.gtx.gemm_core import _gemm_core_njit
    # Touch every JIT kernel once with a tiny input — triggers compile
    A = np.zeros((4, 4), dtype=np.float32)
    B = np.zeros((4, 4), dtype=np.float32)
    bias = np.zeros((4, 4), dtype=np.float32)
    _gemm_core_njit(A, B, False, bias)
    # ... 24 more kernels ...
    return None

@pytest.mark.benchmark(group='gemm', warmup_iterations=2)
def test_gemm_njit_perf(benchmark, warmed_kernels):
    A = np.random.randn(16, 16).astype(np.float32)
    B = np.random.randn(16, 16).astype(np.float32)
    bias = np.zeros((16, 16), dtype=np.float32)
    result = benchmark(_gemm_core_njit, A, B, False, bias)
    assert result.shape == (16, 16)

@pytest.fixture(scope='session')
def baseline_walltime():
    """P6 NumPy baseline measured at start of P7 plan-stage."""
    return 30.0  # seconds, locked from P6 sweep run

def test_full_sweep_5x_speedup(benchmark, baseline_walltime, warmed_kernels):
    # Run vendor 84-op sweep with @njit kernels active
    walltime = benchmark.pedantic(
        run_full_sweep, iterations=1, rounds=3, warmup_rounds=1
    )
    # Acceptance: 5× speedup
    assert walltime['mean'] * 5 <= baseline_walltime, \
        f"5× speedup miss: {walltime['mean']:.2f}s vs baseline {baseline_walltime:.2f}s"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| numba 0.x → 0.59 with numpy 1.x | numba 0.61.2+ with numpy 2.x | mid-2024 (numpy 2.0 release) | CONTEXT D-05 floor (`>=0.59`) is now known to be incompatible with numpy 2.x; lift to `>=0.61.2,<0.66`. |
| `@jit(nopython=True)` (deprecated) | `@njit` (alias for `@jit(nopython=True)`) since 0.39 | 2018 | Use `@njit` only. CONTEXT D-07 already correct. |
| numba `pycc` AOT compile to `.so` | `@njit(cache=True)` on-disk LLVM bitcode | numba 0.46+ (cache stable); pycc deprecated 2023 | CONTEXT D-08 already chose cache=True; deferred-ideas already rejects pycc. |
| `numpy.float16` first-class | NOT YET (CPU target) — open issue [#4402](https://github.com/numba/numba/issues/4402) since 2019 | Never landed for CPU as of 2026-05 | Workaround: pre-cast in caller. RESEARCH Pitfall 1. |
| `with objmode` syntax | Stable since 0.40 | 2019 | Recommended for transcendental ULP-0 escape. Pattern 4 above. |
| pytest-benchmark 4.x | pytest-benchmark 5.x | 2024 | API stable; no breaking changes. CONTEXT D-14 lock. |
| cibuildwheel 1.x | cibuildwheel 3.x (April 2026) | 2026-04-02 | `test-extras` syntax stable since 2.x; same pattern works. |

**Deprecated/outdated:**
- **numba `pycc` AOT** — deprecated 2023; do NOT use for new code (CONTEXT deferred-ideas already lists this).
- **`@jit(nopython=True)`** — write `@njit` instead; alias.
- **`numba.types.float16`** — exists but only in CUDA target; NotImplementedError on CPU.
- **Hand-rolled `_threading_layer` selection** — `NUMBA_THREADING_LAYER` env var has sensible defaults; don't override unless profiling demands.

## Vendor Op Directory Inventory

**Empirical measurement 2026-05-08 (`find vendor/gtx_cpp_reference/test -mindepth 1 -maxdepth 1 -type d ! -name __pycache__`):**

**Total op directories: 84** (NOT 98 as P6 CONTEXT estimated; NOT 103 as user mentioned in deferred-ideas; ACTUAL: **84** ops)

**Asset distribution:**
- 84/84 directories contain at least one `*_ref.txt` file under `<OP>/n1s16/data/`. **Zero ops have missing refs.**
- 81/84 ops have exactly 1 `_ref.txt`; 3 ops (CONCAT, GET_REL_POS, ...) have 2 (multi-output).
- File format: `@<addr>` line followed by 32-byte hex lines, ~16384 lines per file. Spot check (RELU): 1MB ref file, 16384 data rows × 32 bytes.

**Project bundled .elf inventory (`src/main/python/riscv/gtx/data/firmware/`):**
- 12 .elf files: abs, activation_relu_gelu, add_vv, leaky_relu, mm_basic, mul_vv, nop_wjoin, relu, sigmoid, softmax, sum, tanh
- Coverage: ~14% of vendor 84-op set
- Per Pitfall 8, plan-stage Plan 03 must build the missing 72 from `vendor/.../test/<OP>/n1s16/n1s16_<op>.c` using vendor's `run_tests_n1s16.sh` flow with `/opt/riscv/` toolchain.

**Per-op asset audit (all 84):** ABS, ACC, ADD, ADD1, ADD_ID, ADD_REL_POS, ARANGE, CLAMP, CONCAT, CONV_2D, CONV_TRANSPOSE_1D, CONV_TRANSPOSE_2D, COS, CPY, CUMSUM, DIAG, DIAG_MASK_INF, DIAG_MASK_ZERO, DIV, DUP, ELU, EXP, EXPM1, FILL, FLOOR, GATED_LINEAR_ATTN, GEGLU, GEGLU_ERF, GEGLU_QUICK, GELU, GELU_ERF, GELU_QUICK, GET_REL_POS, GET_ROWS, GROUP_NORM, HARDSIGMOID, HARDSWISH, IM2COL, IM2COL_3D, L2_NORM, LEAKY_RELU, LOG, MEAN, MUL, MUL_MAT, MUL_MAT_ID, NEG, NORM, OUT_PROD, PAD, PAD_REFLECT_1D, POOL_1D, POOL_2D, REGLU, RELU, REPEAT, RMS_NORM, ROLL, ROPE, ROUND, RWKV_WKV6, RWKV_WKV7, SCALE, SET, SET_ROWS, SGN, SIGMOID, SILU, SIN, SOFTPLUS, SOFT_MAX, SOLVE_TRI, SQR, STEP, SUB, SUM, SWIGLU_OAI, TANH, TIMESTEP_EMBEDDING, TRI, TRUNC, WIN_PART, WIN_UNPART, XIELU.

**plan-stage gating:** D-10 acceptance must reflect actual numbers — "84 op directories, all with refs, 12 currently bundled, target 84 .elf bundled by Plan 03 GREEN."

## Exact Kernel Inventory

**Empirical count via `grep -E "^def " src/main/python/riscv/gtx/{gemm_core,vec_core,act_core}.py`:**

### gemm_core.py (3 kernels)
1. `gemm_core(A, B, *, has_bias, bias_fp32) -> ndarray[float16]` — line 39
2. `gemm_reduce_sum_a(A, *, prior_accum) -> float` — line 100
3. `gemm_dot(A, B, *, prior_accum) -> float` — line 122

### vec_core.py (7 kernels)
1. `sasmd_kernel(a, b, op) -> ndarray[float16]` — line 34 — has scalar broadcast handling
2. `dot_kernel(a, b) -> float16` — line 66 — explicit 1D for-loop
3. `vsum_kernel(view) -> float16` — line 82 — explicit for-loop sum
4. `clamp_min_kernel(a, scalar) -> ndarray[float16]` — line 96
5. `clamp_max_kernel(a, scalar) -> ndarray[float16]` — line 107
6. `accum_kernel(a) -> ndarray[float16]` — line 118
7. `arange_kernel(n, start, step) -> ndarray[float16]` — line 131

### act_core.py (15 JIT-target kernels + 2 LUT-builders not in JIT scope)
**7 activation kernels:**
1. `relu` — line 40
2. `prelu` — line 49
3. `gelu` — line 59 ⚠️ TRANSCENDENTAL (use objmode)
4. `tanh_act` — line 71 ⚠️ TRANSCENDENTAL (use objmode)
5. `sigmoid` — line 77 ⚠️ TRANSCENDENTAL (use objmode)
6. `softmax` — line 84 ⚠️ TRANSCENDENTAL (use objmode)
7. `esum` — line 109 ⚠️ TRANSCENDENTAL (use objmode)

**2 pool kernels:**
8. `pool_max` — line 130
9. `pool_avg` — line 152

**9 cvt kernels (NOTE: 9 not 7 — recount):**
10. `cvt_qh` — line 284 (FP16→FP8, fancy LUT index)
11. `cvt_hq` — line 295 (FP8→FP16, fancy LUT index)
12. `cvt_ih` — line 305 (FP16→INT8)
13. `cvt_hi` — line 316 (INT8→FP16)
14. `cvt_hn` — line 325 (INT32→FP16)
15. `cvt_sh` — line 333 (FP32→FP16)
16. `cvt_hs` — line 339 (FP16→FP32)
17. `cvt_dh` — line 344 (FP64→FP16)
18. `cvt_hd` — line 350 (FP16→FP64)

**Excluded from JIT scope:**
- `_build_fp8_to_fp16_lut` — line 179, runs ONCE at import, not on hot path.
- `_build_fp16_to_fp8_lut` — line 215, runs ONCE at import.

**Kernel total: 3 + 7 + 18 = 28 kernels.**

⚠️ **CONTEXT D-06 estimate was "약 25" (about 25). Empirical measurement shows 28.** If the 9 cvt kernels are pruned (per Pitfall §"Pattern 3 OPEN QUESTION" — they may not benefit from @njit due to fancy-index dominance), count drops to 19. If only the 7 act + 2 pool are added to gemm + vec, count is 19. Plan-stage decision per CONTEXT Claude's Discretion.

**For NJIT-02 acceptance:** plan-stage Plan 02 (gemm) + Plan 03 (vec) + Plan 04 (act+pool+cvt) lock in exact final count. Recommend planning for **all 28 kernels** initially, then dropping cvt kernels in Plan 04 if benchmark shows no speedup.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x (already in dev extras) + pytest-benchmark 5.x (NEW for P7) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` — already configured with `testpaths = ["tests"]`, `pythonpath = ["src/main/python", "examples"]`. ADD `addopts` adjustment to enable benchmark group filtering. |
| Quick run command | `pytest tests/gtx/test_njit_parity.py -x --no-cov` |
| Full suite command | `pytest tests/gtx -m 'not slow' --no-cov` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NJIT-01 | Lazy import + fallback works regardless of HAS_NUMBA | unit | `pytest tests/gtx/test_jit_shim.py -x --no-cov` | ❌ Wave 0 |
| NJIT-02 | All N kernels have JIT version available | unit | `pytest tests/gtx/test_jit_shim.py::test_njit_kernels_registered -x` | ❌ Wave 0 |
| NJIT-03 | Caller pre-casts FP16→FP32 — JIT signature accepts FP32 only | unit | `pytest tests/gtx/test_njit_parity.py::test_kernel_signatures_fp32_only -x` | ❌ Wave 0 |
| NJIT-04 | vendor 84-op sweep PASS with skip-on-missing-elf | integration | `pytest tests/gtx/test_regression_fw_full_sweep.py --no-cov` | ❌ Wave 0 |
| NJIT-05 | Per-kernel ULP-0 parity (Tier 1) | unit | `pytest tests/gtx/test_njit_parity.py -x --no-cov` | ❌ Wave 0 |
| NJIT-06 | Wall-clock 5× speedup (Tier 3) | benchmark | `pytest tests/gtx/test_njit_perf.py -x --benchmark-only --no-cov` | ❌ Wave 0 |
| NJIT-07 | extras install works (CI verifies via cibuildwheel test-extras=["fast"]) | smoke | `pytest tests/gtx/test_extras_install.py -x --no-cov` (or cibuildwheel CI logs) | ❌ Wave 0 |
| NJIT-08 | Doc sync — REQ + PROJECT + ROADMAP + README all reflect P7 | meta | manual review + grep `pip install spike\[fast\]` README.md | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/gtx/test_njit_parity.py -x --no-cov` (Tier 1 — fast, < 30s for 25-28 kernels)
- **Per wave merge:** `pytest tests/gtx -m 'not slow' --no-cov` (full Tier 1 + Tier 2 + Tier 3 except long-running benchmark cycles)
- **Phase gate:** `pytest tests/gtx --no-cov` (full strict regression including all 84-op sweep + benchmarks; ~5-15 min)

### Wave 0 Gaps

- [ ] `tests/gtx/test_jit_shim.py` — covers NJIT-01, NJIT-02 (HAS_NUMBA gate, kernel registry)
- [ ] `tests/gtx/test_njit_parity.py` — covers NJIT-03, NJIT-05 (28 kernel × ULP-0 parametrize)
- [ ] `tests/gtx/test_njit_perf.py` — covers NJIT-06 (pytest-benchmark + 5× walltime gate + warmup fixture)
- [ ] `tests/gtx/test_regression_fw_full_sweep.py` — covers NJIT-04 (84-op vendor sweep parametrize + skip-on-missing)
- [ ] `tests/gtx/test_extras_install.py` — covers NJIT-07 (smoke import test for `numba.__version__ >= 0.61.2`)
- [ ] `src/main/python/riscv/gtx/_jit.py` — supports NJIT-01 (the shim itself)
- [ ] Framework install: `pip install pytest-benchmark>=4.0` — add to dev extras + ensure dev install picks it up
- [ ] Optional Wave 0: pre-built 72 missing .elf in `tests/gtx/data/elf/` (if Pitfall 8 Path A chosen)
- [ ] Wave 0 fixture: `tests/gtx/conftest.py` — add session-scope `warmed_kernels` for Tier 3

## Plan Structure Recommendation

Based on prior phase precedent (P5 / P6 wave structure) and the natural decomposition of P7 into shim + 3 core wraps + sweep + perf + packaging, **5 plans across 3 waves** is recommended:

| Wave | Plan | Description | Dependencies | Parallel? |
|------|------|-------------|--------------|-----------|
| **Wave 0 (sequential, 1 plan)** | **Plan 01** | Scaffold + shim — `_jit.py`, RED test scaffolds for all P7 tests, REQ/PROJECT/ROADMAP doc sync (NJIT-08), pyproject.toml extras + numba floor (NJIT-07), benchmark fixture, `.gitignore` for `.nbi`/`.nbc`. **First task: lock numba floor at 0.61.2 + run baseline P6 walltime measurement and lock as `BASELINE_WALLTIME = ...` in test_njit_perf.py constant.** | None | Sequential |
| **Wave 1a (parallel, 3 plans)** | **Plan 02** | gemm_core wrapping (3 kernels). MM domain. Includes Tier 1 parity tests for gemm, gemm_reduce_sum_a, gemm_dot. | Plan 01 | Parallel A |
| **Wave 1a (parallel, 3 plans)** | **Plan 03** | vec_core wrapping (7 kernels). VEC domain. Includes Tier 1 parity tests for sasmd_kernel, dot_kernel, vsum_kernel, clamp_min_kernel, clamp_max_kernel, accum_kernel, arange_kernel. | Plan 01 | Parallel B |
| **Wave 1a (parallel, 3 plans)** | **Plan 04** | act_core wrapping. **DUE TO TRANSCENDENTAL OPEN QUESTION:** this plan must FIRST decide Option A/B/C from §"Transcendental ULP-0 Drift". RECOMMENDED: Option B (objmode for gelu/tanh/sigmoid/softmax/esum). Wraps 7 act + 2 pool + 9 cvt = 18 kernels (or 9 if cvt-skip per benchmark). | Plan 01 | Parallel C |
| **Wave 1b (sequential, 1 plan)** | **Plan 05** | Vendor 84-op sweep + Tier 3 perf gate + cibuildwheel integration. Builds the 72 missing .elf (Pitfall 8 Path A). Locks Tier 2 sweep as PASS, Tier 3 5× walltime as PASS. cibuildwheel `test-extras=["fast"]`. | Plans 02, 03, 04 GREEN | Sequential |

**Plan size estimates (lines of code; from P5/P6 lineage):**
- Plan 01: ~250 LOC scaffold + 5 doc updates
- Plan 02: ~80 LOC gemm_core mod + ~120 LOC test_njit_parity[gemm] GREEN
- Plan 03: ~150 LOC vec_core mod + ~200 LOC test_njit_parity[vec*] GREEN
- Plan 04: ~250 LOC act_core mod (objmode for 5 transcendental) + ~300 LOC test_njit_parity[act*] GREEN
- Plan 05: ~300 LOC sweep test + 72 .elf binaries + ~80 LOC test_njit_perf GREEN + cibuildwheel patch

**Parallelization safety:** Plans 02/03/04 do not edit shared files (each touches its own `*_core.py`). Plan 01 may have edited `pyproject.toml` (extras) — Plan 05 also edits `pyproject.toml` (cibuildwheel test-extras), but in a different section — manageable with `--no-verify` like P5/P6.

**Why not 4 plans (collapse 02/03/04):** P5 lineage shows that 6-plan + 4-wave structure is sustainable; P7's natural domain split (mm/vec/act) maps cleanly to 3 parallel plans. Collapsing would force sequential execution of 28-kernel changes, doubling phase walltime estimate.

**Why not 6 plans (separate Tier 1 / Tier 2 / Tier 3):** That would mean Plan 06 just runs `pytest test_njit_perf.py`, which is integrated cleanly into Plan 05 — separation is overhead.

**Risk allocation:** Plan 04 is the highest-risk plan (transcendental drift open question). RESEARCH-recommended Option B (objmode) eliminates the risk; if user prefers Option A (ULP-1 acceptance), Plan 04 simplifies but Plan 01 must update D-12 acceptance to reflect the relaxation.

## Open Questions

1. **Transcendental ULP-0 parity strategy** (PHASE-CRITICAL)
   - **What we know:** numba @njit's `np.tanh` / `np.exp` differ from libm by 1 ULP in FP32, propagating to ~1% FP16 ULP-0 mismatch for GELU. Quantified empirically. `with objmode` escape eliminates the drift.
   - **What's unclear:** Does the user PREFER strict D-12 ULP-0 (use objmode escape) or accept ULP-1 on 5/25 kernels (relax D-12)?
   - **Recommendation:** Plan 01 task 1 — present options A/B/C in the plan body and let user pick. RESEARCH defaults to Option B (objmode). Documented escape paths in Plan 04.

2. **9 cvt kernels — skip @njit due to fancy-index dominance?**
   - **What we know:** cvt_qh / cvt_hq use `LUT[arr.view(np.uint16).astype(np.intp)]` — vectorized fancy-index already at NumPy speed. cvt_ih / cvt_hi / cvt_hn / cvt_sh / cvt_hs / cvt_dh / cvt_hd use `arr.astype(...)` + arithmetic — primarily memory-bandwidth bound.
   - **What's unclear:** Does @njit give >2× speedup on these 9 kernels?
   - **Recommendation:** Plan 04 task 1 — benchmark each cvt kernel under @njit. Skip @njit on any with <2× speedup. Plan-stage discretion. Do not promise 9 cvt kernels in NJIT-02 acceptance until measured.

3. **CONTEXT D-06 estimate (25 kernels) vs measured (28 kernels)**
   - **What we know:** measured exactly: gemm 3 + vec 7 + act (7+2+9) = 28 kernels.
   - **What's unclear:** The user's CONTEXT D-06 said "~25". User intent — exclude any from JIT?
   - **Recommendation:** plan-stage Plan 02 reports measured 28 to user; if user wants to scope down (e.g., skip cvt = 19 kernels), document in Plan 04 explicitly.

4. **First-call compile time aggregate (~16s estimated, not measured)**
   - **What we know:** gemm_core single-kernel cold compile = 640ms (measured 2026-05-08). 28 × 640ms ≈ 17.9s.
   - **What's unclear:** Do users find a 17-second first-import delay acceptable? If yes, no action. If no, eager warmup at import time.
   - **Recommendation:** Plan 04 first-task — measure aggregate cold compile time. If > 30s, add eager warmup. Otherwise leave as lazy.

5. **Vendor 84 op directory count (P6 said 98, user said 103, RESEARCH measured 84)**
   - **What we know:** measured `find vendor/gtx_cpp_reference/test -mindepth 1 -maxdepth 1 -type d ! -name __pycache__ | wc -l` = 84.
   - **What's unclear:** Did the user think of submodule revisions? Are there `__pycache__` exclusions or symlinks the find missed?
   - **Recommendation:** Plan 03 task 0: re-run the find audit and document 84 in the test docstring with the precise find command for reproducibility.

6. **Building the 72 missing .elf — toolchain availability on dev machines**
   - **What we know:** vendor `run_tests_n1s16.sh` requires RISC-V GCC at `/opt/riscv/`. Same toolchain as P4 mm_basic.elf build.
   - **What's unclear:** Does the user agree to bulk-build all 72 in one dev-stage operation and lock them to git (Pitfall 8 Path A), or should P7 gate only the 12 already-bundled?
   - **Recommendation:** Plan 05 first task — verify toolchain exists; if yes, build all 72. If no (user CI-only), document Path B fallback (smaller acceptance criterion).

7. **CONTEXT D-15 50MB cap impact**
   - **What we know:** Adding 72 .elf × ~1.3KB = 94KB. Wheel base size impact negligible. extras (numba+llvmlite+deps) adds ~50-80MB to user's `pip install spike[fast]` env, but CONTEXT D-15 explicitly excludes extras transitive size.
   - **What's unclear:** Does PROJECT.md need the "base wheel ≤50MB" clarification text added (NJIT-08)?
   - **Recommendation:** Yes — Plan 01 task 4 adds the clarification.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All P7 work | ✓ | 3.10.12 | — (cp310-cp312 supported) |
| numpy | All P7 work | ✓ | 2.2.6 | — |
| numba | NJIT-01..06, plan execution | ✓ | 0.63.1 | Pure-NumPy fallback (NJIT-01 D-02). |
| llvmlite | numba transitive | ✓ | 0.46.0 | — (auto with numba) |
| pytest-benchmark | NJIT-06 perf gate | ✓ | 5.1.0 | None — REQUIRED for Tier 3 (test_njit_perf would skip without it). |
| pytest | All test work | ✓ | (via dev extras) | — |
| RISC-V cross-toolchain (`/opt/riscv/`) | Plan 05 .elf build (Pitfall 8 Path A) | ✗ | — | Path B: skip 72 ops, gate only 12 bundled .elf. |
| `dtc` (device-tree-compiler) | cibuildwheel before-all | (CI ✓) | (manylinux2014 yum-installed) | — |
| numba/llvmlite manylinux2014 wheels for cp310-cp312 | NJIT-07 cibuildwheel | ✓ (PyPI) | 0.61.2+ | — |

**Missing dependencies with no fallback:**
- (none for code work — all required tools are available)

**Missing dependencies with fallback:**
- **RISC-V cross-toolchain** for building 72 missing .elf — fallback is to gate only the 12 currently bundled .elf in Plan 05 (weakens NJIT-04 acceptance from "84-op" to "12-op", deferring 72-op coverage to v1.x). plan-stage Plan 05 first task verifies toolchain presence before committing to Path A vs B.

## Sources

### Primary (HIGH confidence)
- Empirical numba 0.63.1 + numpy 2.2.6 testing 2026-05-08 (cp310, manylinux2014_x86_64): bit-exact preservation for FP32 sequential accumulator, np.float16 NotImplementedError, transcendental drift quantification, FP8 LUT module-level capture validation, gemm_core 455× speedup measurement.
- PyPI metadata (`pip download --no-deps numba==X` + unzip METADATA): exact numpy / llvmlite version constraints for numba 0.59.0, 0.59.1, 0.60.0, 0.61.2, 0.62.1, 0.63.1, 0.65.1.
- `pyproject.toml` (this project) — Phase 1 D-08 cibuildwheel matrix lock; numpy>=2.0,<3 floor.
- `vendor/gtx_cpp_reference/test/` directory walk — 84 op directories empirically counted (NOT 98, NOT 103).
- `src/main/python/riscv/gtx/{gemm_core,vec_core,act_core}.py` — kernel inventory (28 kernels) verified by line-by-line read.
- CONTEXT.md decisions D-01..D-16 — verbatim copied into User Constraints section.

### Secondary (MEDIUM confidence)
- [numba 0.65.1 PyPI page](https://pypi.org/project/numba/) - latest version, supported python+numpy ranges
- [numba caching docs](https://numba.readthedocs.io/en/stable/developer/caching.html) — cache invalidation behavior, global capture pitfall
- [numba release notes 0.59.0](https://numba.readthedocs.io/en/stable/release/0.59.0-notes.html) — Python 3.9+ floor, 3.12 official support
- [numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html) — SVML behavior, fastmath semantics
- [numba supported numpy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html) — np.float16 NOT supported on CPU, np.tanh/np.exp/np.maximum/np.where/np.minimum supported on FP32
- [numba types reference](https://numba.readthedocs.io/en/stable/reference/types.html) — confirms float16 not in CPU type set
- [pytest-benchmark usage docs](https://pytest-benchmark.readthedocs.io/en/latest/usage.html) — pedantic mode, warmup_iterations, baseline comparison
- [cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/) — test-extras, test-command syntax verified
- [numba github issue #4402 (fp16 meta)](https://github.com/numba/numba/issues/4402) — open since 2019, no CPU support
- [numba github issue #8138 (fp16 cast bug)](https://github.com/numba/numba/issues/8138) — confirms CPU lack 2022; verified empirically 2026-05-08

### Tertiary (LOW confidence)
- [WebSearch numba np.tanh ULP drift](https://numba.pydata.org/numba-doc/dev/reference/fpsemantics.html) — confirms LLVM intrinsic 1 ULP claim but exact glibc vs LLVM tanhf comparison was DIRECTLY verified empirically (so the claim is HIGH confidence by direct measurement, not by source).
- [numba release notes overview](https://numba.readthedocs.io/en/stable/release-notes-overview.html) — only TOC visible via WebFetch; can't verify all release notes for fp16 mention. Compensated by verifying numba 0.63.1 + 0.65.1 metadata directly.
- [numba#9237 cache args bug](https://github.com/numba/numba/issues/9237) — referenced but not directly relevant to D-08.

## Metadata

**Confidence breakdown:**
- Standard stack (numba 0.61.2+, numpy 2.x compatibility): HIGH — verified PyPI metadata + empirical install
- numba `@njit` cache, fastmath=False bit-exactness for arithmetic: HIGH — verified empirically
- numba np.float16 NotImplementedError: HIGH — verified empirically + multiple GitHub issues
- Transcendental ULP-0 drift: HIGH — verified empirically (803/2048 tanh, 9/1024 GELU, 0/2048 with objmode)
- objmode escape works for tanh: HIGH — verified empirically
- pytest-benchmark warmup pattern: MEDIUM — documented but specific JIT-warmup integration not directly tested
- cibuildwheel test-extras: HIGH — official docs verbatim
- Vendor op directory count: HIGH — counted with `find`
- 28-kernel inventory: HIGH — counted with grep
- Plan structure recommendation (5 plans / 3 waves): MEDIUM — based on P5/P6 precedent, exact wave allocation may shift in plan-stage
- 5× walltime achievability: MEDIUM-HIGH — gemm_core 455× speedup measured; aggregate sweep has IO + dispatch overhead so net ≈ 5-50× is plausible

**Research date:** 2026-05-08

**Valid until:** 2026-06-08 (numba minor versions ship every ~3 months; numpy 2.5 expected late 2026 may shift the upper bound). Re-verify numba latest version at plan-stage (Plan 01 task 1).
