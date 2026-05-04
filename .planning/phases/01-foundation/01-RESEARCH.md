# Phase 1: Foundation - Research

**Researched:** 2026-05-04
**Domain:** Pure-Python FP16/memory infrastructure + packaging baseline (NumPy 2.x, cp310+, git submodule, wheel exclusion)
**Confidence:** HIGH (all critical FP16 view, byte-order, round-trip, and packaging facts empirically verified on local NumPy 2.2.6 + setuptools 80.9.0)

## Summary

Phase 1 builds the package skeleton, FP16 helpers, NumPy memory layer, C++ reference submodule, and packaging baseline that the rest of the GTX port stands on. All architectural choices are locked by CONTEXT.md (D-01..D-17) — this research surfaces the **exact API patterns, command syntax, and configuration deltas** the planner needs.

Key empirical results from local verification on NumPy 2.2.6 (cp310, x86_64 LE host):
1. **`buf.view(np.float16)`** on a `np.uint8` ndarray returns a view (`view.base is buf == True`) — NOT a copy. Slices and reshapes preserve `.base is not None`. Both `view(np.float16)` and `np.frombuffer(buf, dtype='<f2')` produce LE-interpreted FP16 on x86_64; the latter is portable, the former is host-native. CONTEXT.md D-09/D-12 are satisfied by either, with `view(np.float16)` being simplest.
2. **All 65536 FP16 values round-trip exactly** through `f16 → f32 → f16` in NumPy 2.x — including NaN bit-pattern preservation (all 2046 NaN values map to themselves, not to a canonical 0x7E00). Idempotent on the second pass. This validates D-09 with HIGH confidence.
3. **Writing `[0x00, 0x3C]` LE to byte view yields `np.float16(1.0)`** in the halfword view, and writing `np.float16(2.0)` to fp16 view writes bytes `[0x00, 0x40]` LE. CONTEXT.md success criterion 2 is achievable with a 3-line implementation.
4. **`pyproject.toml` requires a one-character fix:** current `include = ["riscv"]` does NOT auto-discover `riscv.gtx`. Empirically verified — must change to `include = ["riscv", "riscv.*"]` or `include = ["riscv*"]`. Without this change, the wheel ships without `riscv.gtx`. The `code_context` claim in CONTEXT.md ("자동으로 riscv.gtx 발견. 추가 설정 불필요") is **incorrect**.
5. **Packaging changes are five small edits** — pyproject.toml (4 stanzas: cibuildwheel build list, classifiers, requires-python, dependencies, packages.find), MANIFEST.in (1 line for vendor exclusion), `riscv/__init__.py` (no change strictly needed for import — namespace works automatically once package is shipped), submodule registration (1 git command).

**Primary recommendation:** Implement Phase 1 as five tightly-scoped task groups: (1) submodule registration + MANIFEST.in vendor exclusion, (2) `pyproject.toml` packaging pivot, (3) `riscv/gtx/` skeleton + `params.py` + `encoding.py` constants, (4) `fp.py` + `memory.py` + `ddr.py` core implementations, (5) `tests/gtx/` test suite (FP roundtrip + memory layout). Each task is independently verifiable and can land in 1-2 hour increments.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**DDR allocation:**
- **D-01:** Lazy `ensure_ddr` pattern — DDR buffer allocated only on first access. No 4GB pre-allocation at `GtxNpu` construction.
- **D-02:** `GTX_DDR_SIZE` env var caps maximum DDR size (default 4GB). Out-of-range access raises explicit error.
- **D-03:** `GTX_DDR_REVERSED=1` mode applies ONLY at I/O boundary (`ddr_init_from_file`/`ddr_dump_to_file`). Internal DDR buffer is always LE. 32-byte bus-word reversal.

**C++ reference snapshot:**
- **D-04:** Git submodule at `vendor/gtx_cpp_reference/` pointing to `https://github.com/Sudo42b/gtx_spike` (public repo).
- **D-05:** Submodule scope = `gtx/` directory + spike patches (`riscv-isa-sim/` mod points). Independently buildable.
- **D-06:** `vendor/gtx_cpp_reference/` is NOT included in the wheel — `MANIFEST.in` excludes it; `[tool.setuptools.package-data]` does not declare it.

**NumPy & Python version (PROJECT-LEVEL CHANGE):**
- **D-07:** NumPy dependency = `numpy>=2.0,<3`. Reverses earlier research recommendation (`>=1.20,<2.0`).
- **D-08:** `requires-python = ">=3.10"` (cp38/cp39 dropped). cibuildwheel matrix → cp310-cp312 only.
- **D-09:** FP16 conversion = `np.float16` view (NOT pure-Python bit manipulation). NumPy 2.x IEEE 754 binary16 RNE. P4/P5 strict measurement. Fallback if differences: port C++ bit ops to `gtx/fp_strict.py`.

**Memory class API surface:**
- **D-10:** Layered API. Both `mem.l0[nest][spu]` raw uint8 ndarray AND `mem.l1_f16(nest, spu, addr, length)` named accessor (view-returning).
- **D-11:** SPR unified dict + address-based routing. `mem.spr: dict[int, int]`. 0x000-0x3FF=GSPR, 0x400-0x7FF=NSPR (NEST encoded in key), 0x800-0xBFF=LSPR (NEST+SPU in key). Mirrors C++ `unordered_map<uint16_t, uint64_t>`.
- **D-12:** All named accessors return non-copying views. `arr.base is not None` asserted in unit tests for every helper.

**Module Layout (already locked):**
- **D-13:** Phase 1 creates: `src/main/python/riscv/gtx/{__init__.py, params.py, encoding.py, fp.py, memory.py, ddr.py}` + `src/main/python/riscv/gtx/ops/__init__.py` (empty marker).
- **D-14:** `__init__.py` exports `fp`, `memory`, `params` only. `GtxNpu` re-export added in P2.

**Test scaffolding (Phase 1 in-scope):**
- **D-15:** Tests at `tests/gtx/`. Files: `tests/gtx/__init__.py`, `tests/gtx/test_fp_roundtrip.py`, `tests/gtx/test_memory_layout.py`. `pytest tests/gtx/` runs standalone; `pytest tests/` integrates.
- **D-16:** FP roundtrip: all 65536 FP16 values, `f16 → f32 → f16 == f16`. <1s expected.
- **D-17:** LE byte-order assertion: write `np.float16(1.0)` via fp16 helper → byte view shows `[0x00, 0x3C]` LE.

### Claude's Discretion

The following are implementation details for Claude to decide:
- `np.float16` view helper internal structure (e.g., where to call `.view(np.float16)`, alignment guards on view slicing)
- `params.py` constant naming (C++ macros verbatim vs Python convention) — RECOMMEND verbatim (`GTX_NEST_NUM=4`, `GTX_SPU_NUM=16`, `GTX_L1_SIZE_BYTES=384*1024`, etc.)
- `encoding.py` scope — funct7 constants only; full `disasm.inc` table deferred to P2
- Exact `MANIFEST.in` exclude pattern
- `numpy` pin form (`numpy>=2.0` vs `numpy>=2.0,<3`) — RECOMMEND conservative `numpy>=2.0,<3`
- cibuildwheel `[tool.cibuildwheel].build` list (cp38/cp39 lines removed)

### Deferred Ideas (OUT OF SCOPE)

- **`@isa.register("gtx")` decorator on GtxNpu** — Phase 2.
- **FP16 strict-mode fallback (`gtx/fp_strict.py`)** — Phase 4/5 if D-09 differences surface.
- **`GtxNpu._memory: GtxMemory` field exposure vs encapsulation** — Phase 2 decision.
- **WRSPR/RDSPR business logic** — Phase 2 (SPR-01/02).
- **DMA op handlers, DDR hex parser body** — Phase 3 (DMA-01..05). Phase 1 produces stubs only.
- **MM gemm_core, VSUM/DOT, activations, format_cvt** — Phases 4/5.
- **`verify.py` port** — Phase 6.
- **Upstream PROJECT.md / REQUIREMENTS.md / STATE.md / ROADMAP.md sync** — already done in `b22ef21` commit; no further sync needed in Phase 1.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | `fp16_to_fp32` / `fp32_to_fp16` via `np.float16` view (D-09); 65536 round-trip idempotency | §"FP16 Conversion (D-09)" — empirical verification of all 65536 values via `astype(np.float32)` / `astype(np.float16)`; NaN bit-pattern preservation HIGH confidence |
| FOUND-02 | L0/L1/L2/DDR as `np.uint8` ndarray + halfword view, all FP16 access in LE | §"Memory Layer (D-10/D-11/D-12)" — `view(np.float16)` semantics, `.base is not None` invariant, layered raw+named accessor pattern |
| FOUND-03 | `src/main/python/riscv/gtx/` skeleton (`__init__.py`, `params.py`, `encoding.py`, `fp.py`, `memory.py`, `ddr.py`, `ops/__init__.py`) — wheel-importable | §"Package Skeleton (D-13/D-14)" — file boundaries, namespace package mechanics, `riscv.gtx` discovery via setuptools |
| FOUND-04 | C++ gtx snapshot at `vendor/gtx_cpp_reference/` as ground-truth | §"C++ Reference Submodule (D-04/D-05/D-06)" — verified `git submodule add` syntax; `MANIFEST.in` exclusion pattern |
| PKG-02 | `numpy>=2.0,<3` runtime dep, `requires-python = ">=3.10"`, cibuildwheel cp310-cp312 only, `pip wheel .` produces valid manylinux2014_x86_64 wheel | §"Packaging Pivot (D-07/D-08)" — exact pyproject.toml deltas; `[tool.cibuildwheel].build` syntax; classifier list change |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

The project's `CLAUDE.md` enforces these directives — research recommendations comply with all of them:

| Directive | Source | How Phase 1 Honors It |
|-----------|--------|------------------------|
| **No new C++ code** | "Tech stack: Python 3.8+ / NumPy ≥2.0 / pyspike의 pybind11 트램폴린. C++ 추가 코드 금지" | Phase 1 is pure-Python only. `setup.py` is NOT modified. |
| **NumPy ≥ 2.0** | "Tech stack: ... NumPy ≥ 2.0" + "discuss-phase 결정 (D-07/D-08): NumPy 2.x FP16 IEEE 754 binary16 RNE" | `pyproject.toml` adds `numpy>=2.0,<3` runtime dep. |
| **Python 3.10+** | "cibuildwheel 매트릭스: cp310-cp312 (Phase 1 D-08)" | `requires-python = ">=3.10"`; classifiers list 3.10/3.11/3.12. |
| **`riscv.isa.ROCC` virtual sig must match** | "Compatibility: `custom0/1/2/3(proc, insn, xs1, xs2) -> reg_t`" | Phase 1 does NOT define a ROCC subclass — deferred to Phase 2. No conflict. |
| **NumPy backend; FP16 = `np.float16` view (D-09)** | "Performance: ... FP16 연산은 `np.float16` view (D-09)" | `fp.py` uses `astype(np.float16)`/`astype(np.float32)` exclusively; no manual bit ops. |
| **No new runtime deps beyond NumPy** | "Dependencies: NumPy 외부 추가 런타임 의존성 신규 도입 금지" | Phase 1 adds only `numpy>=2.0,<3`. NO `importlib_resources` (cp310+ stdlib has it), no scipy, no ml_dtypes. |
| **Bit-exact ULP target** | "Bit-exact: ULP 허용오차 내 ... 회귀 1개라도 깨지면 출하 보류" | FP roundtrip test asserts exact uint16 equality (not ULP-tolerant). NaN handling is bit-pattern-preserving per empirical verification. |
| **pytest framework** | "Testing: pytest 기반(이미 구축됨)" | `tests/gtx/test_*.py` follows existing pyspike test pattern; auto-discovered via `testpaths = ["tests"]`. |
| **manylinux2014_x86_64 / glibc 2.17+** | "Platform: Linux x86_64 / glibc 2.17+ (manylinux2014)" | `[tool.cibuildwheel.linux]` retains `manylinux2014_x86_64` image. cp310-cp312 wheels published to manylinux2014 are still on PyPI for NumPy 2.x. |
| **GSD workflow enforcement** | "Before using Edit, Write, ... start work through a GSD command" | This research artifact was generated via `/gsd:research-phase 1` (downstream of `/gsd:discuss-phase 1` D-decisions). Planner will operate under `/gsd:plan-phase 1`. |
| **GTX_NO_EXIT semantics** | (CLAUDE.md context) — "WJOIN exit(0)" — Phase 2 concern | Out of scope for Phase 1. |
| **FP discipline: load FP16 → upcast to FP32 → compute → single FP16 cast at write-back. Never accumulate in FP16.** | (CLAUDE.md memory layer; PITFALL 2/8) | Phase 1 only implements the conversion helpers themselves; the discipline applies in Phases 4/5. The helpers correctly support this pattern (`astype(np.float32)` then `astype(np.float16)`). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | `>=2.0,<3` (verified 2.2.6 on PyPI 2025-04-30) | FP16/FP32 arithmetic, ndarray-backed memory, view semantics | D-07 locked. NumPy 2.x cp310-cp312 manylinux2014 wheels available; FP16 round-trip empirically verified bit-exact incl. NaN bit-pattern preservation |
| python | `>=3.10` (D-08; classifiers list 3.10/3.11/3.12) | Runtime baseline | D-08 locked. cp310 has stdlib `importlib.resources.files()`, `match` statement (unused in P1 by convention), modern `dict[int,int]` builtin generics |

### Supporting (already in pyproject.toml)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | latest (already in `[dev]`) | Test runner | `pytest tests/gtx/` for Phase 1 acceptance |
| pytest-mypy | latest (already in `[dev]`) | Static type-check at test time | Enforces type hints on `gtx/*.py` automatically |
| pytest-pylint | latest (already in `[dev]`) | Lint at test time | Enforces 120-char line limit, naming conventions |
| setuptools | `>=75` (already in `[build-system]`) | Wheel build backend | `[tool.setuptools.packages.find]` discovers riscv + subpackages |
| setuptools_scm | `>=9` (already in `[build-system]`) | Version stamping | No change needed — already wired |

### Alternatives Considered

| Instead of | Could Use | Tradeoff | Decision |
|------------|-----------|----------|----------|
| `numpy>=2.0,<3` | `numpy` (no upper bound) | NumPy 3.x may break view semantics | **D-07 locked**: pin upper bound to `<3` defensively |
| `np.float16` view (D-09) | Pure-Python bit manipulation (port `gtx_npu.h:89-151`) | Bit-exact match with C++ guaranteed; ~10x slower; ~80 LOC of careful bit work | **D-09 locked**: trust NumPy 2.x RNE; fallback in P4/P5 if strict mode fails |
| Layered API (D-10) | Single high-level only OR single raw only | Raw-only forces every op to handle alignment; high-level-only blocks edge cases | **D-10 locked**: both surfaces |
| Single `dict[int, int]` SPR (D-11) | Three separate dicts (gspr/nspr/lspr) OR ndarray | Three-dict fragments routing logic; ndarray silently returns 0 instead of KeyError on misroute | **D-11 locked**: unified dict matches C++ `unordered_map<uint16_t, uint64_t>` |

**Installation (only delta vs current `pyproject.toml`):**
```toml
[project]
dependencies = [
  "numpy>=2.0,<3",
]
requires-python = ">=3.10"
```

**Version verification (run before finalizing pyproject.toml):**
```bash
pip index versions numpy   # Confirm 2.x line still has cp310 manylinux2014 wheels
# Or:
python3 -c "import urllib.request, json; r = json.loads(urllib.request.urlopen('https://pypi.org/pypi/numpy/json').read()); print('Latest:', r['info']['version']); cp310 = [u for u in r['releases'][r['info']['version']] if 'cp310' in u['filename'] and 'manylinux2014_x86_64' in u['filename']]; print('cp310 manylinux2014 wheel:', cp310[0]['filename'] if cp310 else 'MISSING')"
```
On 2026-05-04: NumPy 2.2.6 is the current 2.x release, cp310-cp312 manylinux2014_x86_64 wheels published.

## Architecture Patterns

### Recommended Project Structure (Phase 1 deliverables only)

```
src/main/python/riscv/gtx/
├── __init__.py        # Re-exports: from . import fp, memory, params (D-14: NOT GtxNpu yet)
├── params.py          # GTX_NEST_NUM=4, GTX_SPU_NUM=16, GTX_L0_SIZE_BYTES=1024,
│                      # GTX_L1_SIZE_BYTES=384*1024, GTX_L2_SIZE_BYTES=16*1024*1024,
│                      # GSPR_BASE=0x000, NSPR_BASE=0x400, LSPR_BASE=0x800, etc.
├── encoding.py        # Phase 1 SCOPE: funct7 constants only (full disasm.inc deferred to P2)
│                      # GTX_F7_WRSPR=0x00, GTX_F7_RDSPR=0x01, GTX_F7_WJOIN=0x03, etc.
├── fp.py              # fp16_to_fp32(arr_f16) -> arr_f32, fp32_to_fp16(arr_f32) -> arr_f16
│                      # Uses np.ndarray.astype() exclusively (D-09)
├── memory.py          # GtxMemory class — L0/L1/L2 contiguous np.uint8 alloc + view exposers
│                      # spr: dict[int, int] (D-11)
├── ddr.py             # DDR lazy alloc (D-01) + GTX_DDR_SIZE env handling (D-02)
│                      # I/O stubs (Phase 1: declare API; Phase 3 fills body)
└── ops/
    └── __init__.py    # Empty marker — populated P2-P5

tests/gtx/
├── __init__.py        # Empty (or with `pass`); enables pytest collection from this dir
├── test_fp_roundtrip.py   # 65536 FP16 round-trip; NaN bit-pattern preservation; subnormals; ±0
└── test_memory_layout.py  # LE byte order assertion; view-base invariant; SPR routing smoke

vendor/gtx_cpp_reference/  # git submodule pointing to https://github.com/Sudo42b/gtx_spike
                            # Wheel-excluded via MANIFEST.in (D-06)
```

### Pattern 1: NumPy-backed memory with halfword view (D-10/D-12)

**What:** One contiguous `np.uint8` allocation per region; FP16/uint16 views derived via `.view()` and `.reshape()` — never copied.

**When to use:** All L0/L1/L2 access. DDR follows the same pattern but allocates lazily.

**Example (verified runnable on numpy 2.2.6):**
```python
# src/main/python/riscv/gtx/memory.py
import numpy as np
from .params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES

class GtxMemory:
    def __init__(self) -> None:
        # One contiguous allocation per region (no copies thereafter)
        self._l0_bytes = np.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES), dtype=np.uint8
        )
        self._l1_bytes = np.zeros(
            (GTX_NEST_NUM, GTX_SPU_NUM, GTX_L1_SIZE_BYTES), dtype=np.uint8
        )
        self._l2_bytes = np.zeros(
            (GTX_NEST_NUM, GTX_L2_SIZE_BYTES), dtype=np.uint8
        )
        # SPR unified dict (D-11)
        self.spr: dict[int, int] = {}
        # DDR lazy (D-01)
        self._ddr_bytes: np.ndarray | None = None

    # Raw byte view (D-10 low-level)
    def l1_byte(self, nest: int, spu: int) -> np.ndarray:
        return self._l1_bytes[nest, spu]   # shape (GTX_L1_SIZE_BYTES,), dtype uint8

    # Halfword fp16 view (D-10 named accessor; D-12 view guarantee)
    def l1_f16(self, nest: int, spu: int) -> np.ndarray:
        view = self._l1_bytes[nest, spu].view(np.float16)
        # Empirically: view.base is self._l1_bytes — no copy
        # shape: (GTX_L1_SIZE_BYTES // 2,) = (196608,)
        return view

    # Halfword uint16 view (rare; for pattern testing)
    def l1_u16(self, nest: int, spu: int) -> np.ndarray:
        return self._l1_bytes[nest, spu].view(np.uint16)
```

**Key invariants (asserted in tests):**
1. `mem.l1_f16(0, 0).base is not None` — view, not copy
2. `mem.l1_byte(0, 0)[0:2] = [0x00, 0x3C]; mem.l1_f16(0, 0)[0] == np.float16(1.0)` — LE byte order
3. `mem.l1_f16(0, 0)[0] = np.float16(2.0); mem.l1_byte(0, 0)[0:2].tolist() == [0x00, 0x40]` — fp16 write produces LE bytes

### Pattern 2: FP16 conversion via NumPy view (D-09)

**What:** Use `astype(np.float32)` and `astype(np.float16)` exclusively. NumPy 2.x guarantees IEEE 754 binary16 RNE; do NOT hand-roll bit manipulation.

**When to use:** All FP16↔FP32 conversion sites in `gtx/fp.py`. Phase 4/5 ops will call these helpers.

**Example (verified runnable on numpy 2.2.6):**
```python
# src/main/python/riscv/gtx/fp.py
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, np.float16, np.float32, float]

def fp16_to_fp32(x: ArrayLike) -> np.ndarray:
    """Widen FP16 → FP32. Lossless. Bit-exact RNE not relevant (widening cast).
    
    Note: returns a NEW array (astype always copies). Caller must NOT expect base preservation.
    For zero-copy view of FP16 storage, use mem.l1_f16(...) directly and pass to NumPy ops
    with dtype=np.float32 in reduction kwargs.
    """
    return np.asarray(x, dtype=np.float16).astype(np.float32)

def fp32_to_fp16(x: ArrayLike) -> np.ndarray:
    """Narrow FP32 → FP16 with IEEE 754 binary16 RNE (NumPy 2.x default).
    
    Empirically verified on NumPy 2.2.6: idempotent for all 65536 FP16 values
    (including NaN bit-pattern preservation).
    """
    return np.asarray(x, dtype=np.float32).astype(np.float16)
```

**Empirical proof (run during planning to verify on the actual target environment):**
```python
import numpy as np
all_u16 = np.arange(65536, dtype=np.uint16)
all_f16 = all_u16.view(np.float16)
fp32 = all_f16.astype(np.float32)
back_u16 = fp32.astype(np.float16).view(np.uint16)
# All 65536 values (including 2046 NaN bit patterns) round-trip exactly:
assert np.array_equal(back_u16, all_u16)   # passes on NumPy 2.2.6
```

### Pattern 3: Lazy DDR allocation (D-01/D-02/D-03)

**What:** DDR buffer is `None` at construction; first access triggers `ensure_ddr(size)`. `GTX_DDR_SIZE` env var caps maximum.

**When to use:** All DDR access goes through `mem.ensure_ddr(end_offset)`. I/O boundary (`ddr_init_from_file`/`ddr_dump_to_file`) handles `GTX_DDR_REVERSED` per D-03.

**Example (Phase 1 stub; Phase 3 fills body):**
```python
# src/main/python/riscv/gtx/ddr.py
import os
import numpy as np

DEFAULT_DDR_SIZE = 4 * 1024 * 1024 * 1024   # 4 GiB

def get_ddr_cap() -> int:
    """Read GTX_DDR_SIZE env var; default 4GB."""
    val = os.environ.get("GTX_DDR_SIZE")
    if val is None:
        return DEFAULT_DDR_SIZE
    # Support "4G", "64M", "1024K" suffixes
    val = val.strip().upper()
    if val.endswith("G"):
        return int(val[:-1]) * 1024 ** 3
    if val.endswith("M"):
        return int(val[:-1]) * 1024 ** 2
    if val.endswith("K"):
        return int(val[:-1]) * 1024
    return int(val)

def ensure_ddr(mem: "GtxMemory", end_offset: int) -> np.ndarray:
    """Lazy DDR alloc. Grows in halving doubles up to GTX_DDR_SIZE cap.
    Phase 1: minimal — just ensure end_offset is reachable, raise if > cap.
    Phase 3: full grow-on-demand semantics matching C++ gtx_npu_t::ensure_ddr.
    """
    cap = get_ddr_cap()
    if end_offset > cap:
        raise ValueError(f"DDR access {end_offset:#x} exceeds cap {cap:#x} (set GTX_DDR_SIZE env var to raise)")
    if mem._ddr_bytes is None or end_offset > mem._ddr_bytes.size:
        # Phase 1 stub: allocate exactly end_offset (rounded up). Phase 3: doubling strategy.
        new_size = max(end_offset, mem._ddr_bytes.size if mem._ddr_bytes is not None else 0)
        new_arr = np.zeros(new_size, dtype=np.uint8)
        if mem._ddr_bytes is not None:
            new_arr[:mem._ddr_bytes.size] = mem._ddr_bytes
        mem._ddr_bytes = new_arr
    return mem._ddr_bytes
```

### Pattern 4: Package skeleton + namespace re-export (D-13/D-14)

**What:** `riscv/gtx/__init__.py` re-exports modules so `from riscv.gtx import fp, memory, params` works in user code. NO `GtxNpu` exposure (Phase 2).

**Example:**
```python
# src/main/python/riscv/gtx/__init__.py
"""GTX NPU functional model — Phase 1 skeleton.

Phase 1 exposes FP16 helpers, memory layer, and HW parameter constants.
GtxNpu (the ROCC subclass) is added in Phase 2.
"""
from . import fp
from . import memory
from . import params
from . import encoding
from . import ddr

__all__ = ["fp", "memory", "params", "encoding", "ddr"]
```

```python
# src/main/python/riscv/gtx/ops/__init__.py
"""Op handler package marker. Populated in Phases 2-5."""
```

**`src/main/python/riscv/__init__.py` change:** **None required** for Phase 1. The current `riscv/__init__.py` doesn't enumerate subpackages — `riscv.gtx` is imported lazily by user code via `from riscv.gtx import ...`. This works automatically once `riscv.gtx` is shipped in the wheel (which depends on the `pyproject.toml` discovery fix described below).

⚠️ **CRITICAL FINDING (DEVIATION FROM CONTEXT.md):** The `code_context` block in CONTEXT.md states "`[tool.setuptools.packages.find].include = ['riscv']` — 자동으로 riscv.gtx 발견. 추가 설정 불필요." This is **incorrect** — empirically verified on setuptools 80.9.0:

```bash
$ python3 -c "import setuptools; print(setuptools.find_packages(where='src/main/python', include=['riscv']))"
['riscv']        # WITHOUT 'riscv.gtx'

$ python3 -c "import setuptools; print(setuptools.find_packages(where='src/main/python', include=['riscv', 'riscv.*']))"
['riscv', 'riscv.gtx']   # CORRECT
```

The `pyproject.toml` MUST be patched to:
```toml
[tool.setuptools.packages.find]
where = ["src/main/python"]
include = ["riscv", "riscv.*"]   # was: ["riscv"]
```
or equivalently `include = ["riscv*"]`. Without this patch, `pip install spike` will not deliver `riscv.gtx` and Phase 1 success criterion 3 will silently fail.

### Anti-Patterns to Avoid

- **`arr.view(np.float16)` without explicit byte order** — host-native byte order. SAFE on x86_64 manylinux2014 baseline (LE), but add `assert sys.byteorder == 'little'` tripwire in `riscv/gtx/__init__.py` for the hypothetical non-LE host. CONTEXT.md Pitfall 1 confirms this. Per-spec it is allowed; just guard.
- **`np.zeros(GTX_DDR_SIZE_BYTES, dtype=np.uint8)` eager allocation** — violates D-01. Always `ensure_ddr()` lazy.
- **In-place op interceded by `arr.copy()`** — violates D-12. Helpers must `assert result.base is not None`.
- **`.astype()` for FP16/FP32 conversion in hot loops** — `astype` always copies. Phase 1 helpers exposing `fp16_to_fp32` are intentionally copy-returning (used for one-off conversions); for bulk paths in Phases 4/5, ops will use `arr.astype(np.float32)` once at op start, accumulate in FP32, then `.astype(np.float16)` once at writeback. Phase 1 helpers are not in any hot loop.
- **Per-element Python loop over FP16 bytes** — Pitfall 12. Phase 1 doesn't have this risk (no ops yet) but tests should verify the round-trip vectorially, not in a `for i in range(65536)` loop.
- **`include = ["riscv"]`** in `pyproject.toml` — does NOT auto-include `riscv.gtx`. Use `include = ["riscv", "riscv.*"]`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16↔FP32 RNE conversion (D-09) | Bit manipulation port of `gtx_npu.h:89-151` | `np.ndarray.astype(np.float32)` / `.astype(np.float16)` | NumPy 2.x guarantees IEEE 754 binary16 RNE. Empirically verified all 65536 values round-trip including NaN bit patterns. P4/P5 fallback if strict mode fails. |
| LE byte order interpretation | `int.from_bytes(buf[off:off+2], 'little')` for FP16 reads | `np.frombuffer(buf, dtype='<f2')` OR `buf.view(np.float16)` (host-native, safe on x86_64 LE) | Vectorized; ~50× faster on N>16. View-based is zero-copy. |
| Array views vs copies | Manually compute strides | `arr.view(dtype)` + `arr.reshape(...)` — preserves `.base` | NumPy guarantees view semantics when last axis is contiguous (since 1.23). Empirically verified on 2.2.6. |
| Halfword reinterpretation of `np.uint8` ndarray | `np.array([(b1 << 8) \| b0 for b0, b1 in zip(...)])` | `arr.view(np.uint16)` or `arr.view(np.float16)` | Single C-level call, zero-copy, LE-native on x86_64. |
| DDR size parsing ("4G", "64M", "1024K") | Custom regex | Manual `endswith("G"/"M"/"K")` switch (4 lines) | Trivial; no `humanfriendly` dep needed. Stdlib `int(val[:-1])` suffices. |
| Submodule registration | `git clone --depth=1 vendor/gtx_cpp_reference; manually maintain HEAD pointer` | `git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference` | Captures upstream pointer in `.gitmodules` + commits a tracked SHA in the parent repo. CI auto-clones via `git submodule update --init`. |
| Resource access in installed wheel (Phase 6 concern) | `pathlib.Path(__file__).parent / "data"` | `importlib.resources.files("riscv.gtx") / "data"` | cp310+ stdlib `importlib.resources.files()` works for zip-imported wheels; `__file__`-based path fails. **Phase 1 doesn't access resources** but `riscv.gtx/__init__.py` should be designed to be `importlib.resources`-friendly. |
| Test discovery for new `tests/gtx/` subdir | Custom conftest discovery | None — `pytest` auto-discovers via `testpaths = ["tests"]` | `pyproject.toml [tool.pytest.ini_options].testpaths = ["tests"]` already covers `tests/gtx/` by recursive glob. No `tests/gtx/conftest.py` needed for Phase 1 (Phase 2+ may add fixtures). `tests/gtx/__init__.py` is sufficient. |
| Type hints on dict (Python 3.10+) | `from typing import Dict; Dict[int, int]` | `dict[int, int]` (PEP 585 builtin generics) | cp310 baseline (D-08) makes `from __future__ import annotations` unnecessary AND `dict[int, int]` runtime-evaluatable in annotations. Same for `list[T]`, `tuple[T, ...]`, `np.ndarray | None`. |

**Key insight:** Phase 1 is intentionally THIN — the hardest engineering is in Phases 4/5 (FP discipline in MM/VEC/ACT). Phase 1's job is to give those phases a sound, view-preserving, LE-correct foundation. Custom solutions in Phase 1 leak technical debt into the bit-exact path; rely on NumPy 2.x semantics + setuptools defaults wherever possible.

## Runtime State Inventory

> Phase 1 is GREENFIELD (creates new code, does not rename existing identifiers). The runtime state inventory below documents the **only** state-touch point: registering a git submodule, which adds entries to `.gitmodules` and the git index. No data migration, no service config, no OS-registered state, no secrets, no built artifacts to invalidate.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 1 creates new `.py` files; no databases, key-value stores, or persistent caches reference Phase 1 identifiers. | None |
| Live service config | None — pyspike is a library (no daemons, no UI-managed config, no n8n/Datadog-style external services). | None |
| OS-registered state | None — no Task Scheduler / launchd / systemd entries reference Phase 1 paths. | None |
| Secrets/env vars | New env vars introduced (consumed only by Phase 1 code; no rename of existing): `GTX_DDR_SIZE` (D-02; consumed by `ddr.py`), `GTX_DDR_REVERSED` (D-03; consumed by `ddr.py` Phase 3+). Both are NEW additions, not renames. | None — code reads them directly |
| Build artifacts / installed packages | The existing built wheel (`dist/spike-*.whl`) does NOT include `riscv.gtx`. After Phase 1, a fresh `python -m build` is required to produce a wheel that ships the new package. Existing `*.egg-info/` directories (if any in dev install) need refresh: `pip install -e ".[dev]"`. The `vendor/spike/` submodule is unaffected. | Plan task: `pip install -e ".[dev]"` after `pyproject.toml` changes; full `pip wheel .` for Success Criterion 5. |

**Verified by:** `find . -name "*.egg-info" -not -path "./vendor/*"` shows current state; `git ls-files | xargs grep -l "riscv.gtx"` confirms no existing references (Phase 1 introduces them).

## Common Pitfalls

(Phase 1 avoids/defends against these. Pitfalls deferred to later phases are NOT listed here.)

### Pitfall 1: `verify.py` BE vs L1/L0 LE Byte Order Discrepancy
**What goes wrong:** `verify.py` parses DDR hex as big-endian FP16 pairs (`r_raw = (data[i*2] << 8) | data[i*2+1]`), but L1/L0 in-memory is little-endian. A naive port might "fix" `verify.py` to LE, breaking SystemC TLM compat; conversely, might write BE to L1 to match `verify.py`, breaking firmware compat.
**Why it happens:** Author compares `verify.py` byte order with L1 byte order side-by-side and concludes they should match.
**How to avoid in Phase 1:**
- L1/L0 internally **little-endian** — confirmed by D-17 success criterion (`mem.l1_byte(0,0)[0:2] == [0x00, 0x3C]` for `np.float16(1.0)`).
- Add explicit unit test: `mem.l1_f16(0, 0)[0] = np.float16(1.0)` → assert `bytes(mem.l1_byte(0, 0)[0:2]) == b"\x00\x3C"`.
- Phase 1 does NOT touch `verify.py` (deferred to Phase 6).
**Warning signs:** Every-other-byte differs in DDR hex compare; `--ulp 1` reports astronomical distances.
**Phase 1 verification:** `tests/gtx/test_memory_layout.py::test_le_byte_order_assertion`.

### Pitfall 2 / 8 (combined): NumPy `np.float16` Cast Precision
**What goes wrong:** Concerns that NumPy `np.float16` arithmetic produces different results than C++ `gtx_fp32_to_16` for subnormals, NaN payload, halfway-rounding (RNE half-to-even).
**Why it happens:** NumPy 1.x had value-based casting (NEP 50); 2.x adopted strict NEP 50. Some C++ implementations preserve NaN payload bits; NumPy may canonicalize.
**How to avoid in Phase 1:**
- D-09 LOCKS the `np.float16` view choice. Phase 1 implements only the round-trip helpers.
- **Empirically verified on NumPy 2.2.6 (cp310):** all 65536 FP16 values (including all 2046 NaN bit patterns and all 2046 subnormals) round-trip EXACTLY through `astype(np.float32) → astype(np.float16)`. This is HIGH-confidence evidence that D-09 is sound, at least at the conversion-helper level.
- Phase 1 acceptance test exhaustively asserts this round-trip.
- Phase 4/5 will discover any additional differences in MM/VEC/ACT op-level computation (multi-step accumulation, fused operations). At that point the fallback `gtx/fp_strict.py` may be added.
**Warning signs (in Phases 4/5, NOT Phase 1):** `verify.py --strict` fails despite `--ulp 1 --atol 0.001` passing.
**Phase 1 verification:** `tests/gtx/test_fp_roundtrip.py::test_all_65536_fp16_values_idempotent`, `::test_nan_bit_patterns_preserved`, `::test_subnormals_roundtrip`.

### Pitfall 13: Fancy Indexing / View vs Copy
**What goes wrong:** `arr[mask] = ...` with boolean mask creates a copy on read but writes back via fancy indexing — not a true view. Modular addressing `arr[idx % size]` for arrays of indices that wrap creates undefined order.
**Why it happens:** "NumPy is fast" → write vectorized fancy indexing.
**How to avoid in Phase 1:**
- Allocate L0/L1/L2 as one contiguous `(NEST, SPU, SIZE)` `np.uint8` ndarray (Pattern 1 above).
- All accessors return slice views: `self._l1_bytes[nest, spu]` — not `np.take` or fancy index.
- D-12 mandates `view.base is not None` for every named accessor; unit tests assert this.
**Warning signs:** Writes "succeed" but reads show old data; `arr.flags.owndata is True` after a slice (indicates copy).
**Phase 1 verification:** `tests/gtx/test_memory_layout.py::test_l1_f16_view_invariant`, `::test_l0_f16_view_invariant`, `::test_l2_view_invariant`, `::test_slice_preserves_base`.

### Pitfall (Phase 1-specific): pyproject.toml `include = ["riscv"]` does NOT auto-include `riscv.gtx`
**What goes wrong:** Empirically verified on setuptools 80.9.0 — `find_packages(include=["riscv"])` returns `["riscv"]` only, NOT `["riscv", "riscv.gtx"]`.
**Why it happens:** Setuptools docs imply automatic recursive discovery, but the `include` glob requires explicit `riscv.*` to match subpackages.
**How to avoid:** Patch `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
include = ["riscv", "riscv.*"]   # was: ["riscv"]
```
**Warning signs:** `pip install dist/spike-*.whl` then `python -c "from riscv.gtx import fp"` → `ModuleNotFoundError: No module named 'riscv.gtx'`. Success criterion 3 silently fails.
**Phase 1 verification:** Build wheel via `pip wheel .`, then `unzip -l dist/spike-*.whl | grep gtx` should list `riscv/gtx/__init__.py` etc. AND `python -c "from riscv.gtx import fp"` in a clean cp310 venv must succeed.

### Pitfall (Phase 1-specific): `MANIFEST.in recursive-include vendor *` will sweep gtx_cpp_reference into sdist
**What goes wrong:** Existing line 13 `recursive-include vendor *` includes the entire `vendor/` tree in sdist. After D-04 adds `vendor/gtx_cpp_reference/` submodule, the sdist (and potentially wheel) bloats with all C++ reference sources.
**Why it happens:** Submodule directories are treated as plain dirs by `MANIFEST.in` patterns.
**How to avoid (D-06):** Add explicit exclusion BEFORE the `recursive-include vendor *` rule, OR use `prune` directive (overrides earlier `recursive-include`):
```
include LICENSE
include MANIFEST.in
include pyproject.toml
include README.md
include riscv.pth
include setup.py
recursive-include docs *.puml *.md
recursive-include src/main/cpp *.h *.cc
recursive-include src/main/python *.py *.pyi py.typed
recursive-include tests *.py *.pyi
recursive-include examples *
recursive-include tests/data *.py *.elf
recursive-include vendor *
recursive-exclude src/main/python/riscv/data *
prune vendor/gtx_cpp_reference                      # ← NEW: exclude C++ reference (D-06)
recursive-exclude vendor/gtx_cpp_reference *        # ← Belt-and-suspenders
recursive-exclude . __pycache__ *.pyc *.pyo .gitignore .DS_Store .coverage .mypy_cache .tox .pytest_cache *.egg-info
```
The `prune` directive is the canonical way to exclude an entire directory tree (per `setuptools` MANIFEST.in docs); it works with both sdist and any wheel built from the source tree. Note that `prune` removes ALL files including `.gitmodules` markers within that subtree — but `.gitmodules` itself lives at repo root, so submodule registration is unaffected. Wheel exclusion is also enforced because `[tool.setuptools.package-data]` does NOT declare `vendor/gtx_cpp_reference/*` (D-06).
**Warning signs:** `du -sh dist/spike-*.tar.gz` shows tens of MB; `tar tzf dist/spike-*.tar.gz | grep gtx_cpp_reference` shows entries.
**Phase 1 verification:** `python -m build --sdist`; `tar tzf dist/spike-*.tar.gz | grep -c gtx_cpp_reference` must be 0. `du -sh dist/spike-*.tar.gz` must not regress meaningfully vs pre-Phase-1 baseline.

## Code Examples

### Example 1: Adding the C++ reference submodule (D-04, FOUND-04)

```bash
# From the project root (/mnt/e/14_NIGHTLY/pyspike)
# Single-shot command — no branch pin needed (uses default branch HEAD)
git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference

# Verify registration
git submodule status
# Expected output:
#  20feb9c2bf2a7deab964d8190b0cbd4b4131bec3 vendor/spike (...)
#  <new-sha> vendor/gtx_cpp_reference (heads/main or tag)

# Verify .gitmodules registration
cat .gitmodules
# Expected entries:
# [submodule "vendor/spike"]
#     path = vendor/spike
#     url = ../spike
# [submodule "vendor/gtx_cpp_reference"]
#     path = vendor/gtx_cpp_reference
#     url = https://github.com/Sudo42b/gtx_spike

# (Optional, if a specific commit pin is desired):
# git -C vendor/gtx_cpp_reference checkout <sha>
# git add vendor/gtx_cpp_reference
# git commit -m "chore(vendor): pin gtx_cpp_reference to <sha>"
```

**Note on submodule URL:** D-04 specifies `https://github.com/Sudo42b/gtx_spike`. This is a public repo (per CONTEXT.md `<specifics>`), so anonymous clones work in CI. No SSH key or PAT needed.

**Note on branch/commit pinning:** `git submodule add` defaults to tracking the remote default branch HEAD at the time of `git submodule update`. To pin to a specific commit (recommended for reproducibility): after `git submodule add`, `cd vendor/gtx_cpp_reference; git checkout <commit-sha>; cd -; git add vendor/gtx_cpp_reference`. The parent repo records the SHA in its tree, so `git submodule update --init` always restores that exact commit.

**Note on `--branch` flag:** `git submodule add -b main https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference` would record a branch tracking entry in `.gitmodules` (`branch = main`). With this set, `git submodule update --remote` advances the submodule to the latest commit on `main`. For the pyspike use case (golden ground-truth, not auto-tracking), the **default behavior (no `-b`) is preferred** — submodule pin is whatever SHA was committed in the parent repo, not a moving target.

### Example 2: cibuildwheel `before-all` hook for submodule init (PKG-02)

```toml
# pyproject.toml — REPLACE the existing [tool.cibuildwheel.linux] block

[tool.cibuildwheel.linux]
before-all = "yum install -y dtc && git submodule update --init --recursive"
# Note: cibuildwheel before-all runs INSIDE the manylinux2014 container,
# where git is available. The container has the project source mounted,
# but submodules are NOT pre-initialized — must be done explicitly.
# `--recursive` handles vendor/spike (existing) AND vendor/gtx_cpp_reference (new).
```

Per cibuildwheel docs (verified 2026-05-04): `before-all` runs inside the manylinux container before any wheel build. Git is available in the manylinux2014_x86_64 base image. `git submodule update --init --recursive` is the canonical way to initialize all registered submodules.

**Subtle but important caveat:** if `vendor/gtx_cpp_reference/` is `prune`-d in `MANIFEST.in`, it will NOT be in the sdist. cibuildwheel by default builds from the **source tree** (not sdist), so the submodule **will** be initialized inside the container even if pruned from sdist. The `prune` directive only affects the sdist tarball. This means the wheel will not contain submodule files (good — matches D-06), but the build environment can still reference them if needed. Phase 1 wheel build does NOT need submodule contents (it's pure-Python `riscv/gtx/`), so the submodule init is purely defensive — kept for future phases that may want to compare against C++ during build-time verification.

### Example 3: Full `pyproject.toml` deltas (consolidated)

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -14,8 +14,6 @@
 
 [tool.cibuildwheel]
 build = [
-  "cp38-manylinux_x86_64",
-  "cp39-manylinux_x86_64",
   "cp310-manylinux_x86_64",
   "cp311-manylinux_x86_64",
   "cp312-manylinux_x86_64"
@@ -33,7 +31,7 @@
 ]
 
 [tool.cibuildwheel.linux]
-before-all = "yum install -y dtc"
+before-all = "yum install -y dtc && git submodule update --init --recursive"
 
 [project]
 name = "spike"
@@ -55,15 +53,16 @@
   "Operating System :: POSIX :: Linux",
   "Programming Language :: C++",
   "Programming Language :: Python",
   "Programming Language :: Python :: 3",
-  "Programming Language :: Python :: 3.8",
-  "Programming Language :: Python :: 3.9",
   "Programming Language :: Python :: 3.10",
   "Programming Language :: Python :: 3.11",
   "Programming Language :: Python :: 3.12",
   "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
   ...
 ]
-requires-python = ">=3.8"
+requires-python = ">=3.10"
+
+dependencies = [
+  "numpy>=2.0,<3",
+]
 
 [project.urls]
 Homepage = "https://github.com/liuyu81/pyspike"
@@ -106,11 +105,15 @@
 
 [tool.setuptools.packages.find]
 where = [
   "src/main/python"
 ]
 include = [
-  "riscv"
+  "riscv",
+  "riscv.*"
 ]
 
 [tool.setuptools.package-data]
 riscv = [
   "data/bin/spike",
   ...
 ]
```

### Example 4: FP roundtrip test (D-16)

```python
# tests/gtx/test_fp_roundtrip.py
"""Phase 1 acceptance: 65536 FP16 values round-trip exactly through fp.fp16_to_fp32 / fp32_to_fp16.

D-09 risk acknowledgment: NumPy 2.x np.float16 RNE may differ from C++ gtx_fp32_to_16
on subnormal/NaN payload/halfway-rounding edge cases. Phase 1 verifies the *helper-level*
round-trip; full strict-mode comparison vs C++ is deferred to P4/P5.
"""
import numpy as np

from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16


def test_all_65536_fp16_values_idempotent():
    """For every FP16 bit pattern x: fp32_to_fp16(fp16_to_fp32(x)) == x (bitwise)."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f16 = all_u16.view(np.float16)

    fp32 = fp16_to_fp32(all_f16)
    back_f16 = fp32_to_fp16(fp32)
    back_u16 = back_f16.view(np.uint16)

    # Empirically verified on NumPy 2.2.6 (cp310 on x86_64 LE):
    # ALL 65536 values round-trip exactly, including all 2046 NaN bit patterns.
    np.testing.assert_array_equal(back_u16, all_u16)


def test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern():
    """NaN inputs produce NaN outputs; bit pattern is preserved (NumPy 2.x behavior)."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f16 = all_u16.view(np.float16)
    nan_mask = np.isnan(all_f16)
    nan_count = int(nan_mask.sum())
    assert nan_count == 2046, f"Expected 2046 NaN bit patterns, got {nan_count}"

    back_u16 = fp32_to_fp16(fp16_to_fp32(all_f16)).view(np.uint16)
    # All NaN inputs produce NaN outputs:
    assert np.all(np.isnan(back_u16.view(np.float16)[nan_mask]))
    # Bit pattern is preserved (HIGH-confidence on NumPy 2.x):
    np.testing.assert_array_equal(back_u16[nan_mask], all_u16[nan_mask])


def test_subnormals_roundtrip():
    """All FP16 subnormals (exp == 0, mantissa != 0) round-trip exactly."""
    # FP16 subnormals: 0x0001..0x03FF and 0x8001..0x83FF
    subnormal_pos = np.arange(0x0001, 0x0400, dtype=np.uint16)
    subnormal_neg = np.arange(0x8001, 0x8400, dtype=np.uint16)
    subs = np.concatenate([subnormal_pos, subnormal_neg]).view(np.float16)

    back = fp32_to_fp16(fp16_to_fp32(subs)).view(np.uint16)
    expected = np.concatenate([subnormal_pos, subnormal_neg])
    np.testing.assert_array_equal(back, expected)


def test_negative_zero_preserved():
    """fp32_to_fp16(fp16_to_fp32(np.float16(-0.0))) preserves -0.0 (sign bit)."""
    neg_zero_u16 = np.array([0x8000], dtype=np.uint16)
    neg_zero_f16 = neg_zero_u16.view(np.float16)
    back = fp32_to_fp16(fp16_to_fp32(neg_zero_f16)).view(np.uint16)
    np.testing.assert_array_equal(back, neg_zero_u16)


def test_known_values():
    """Sanity-check known FP16 ↔ FP32 conversions."""
    cases = [
        (np.float16(1.0), np.float32(1.0), 0x3C00),
        (np.float16(2.0), np.float32(2.0), 0x4000),
        (np.float16(0.5), np.float32(0.5), 0x3800),
        (np.float16(-1.0), np.float32(-1.0), 0xBC00),
    ]
    for f16, f32, raw in cases:
        assert fp16_to_fp32(f16) == f32
        assert fp32_to_fp16(f32) == f16
        assert int(fp32_to_fp16(f32).view(np.uint16)) == raw
```

### Example 5: Memory layout test (D-17, D-12)

```python
# tests/gtx/test_memory_layout.py
"""Phase 1 acceptance: GtxMemory layout invariants.

D-17: writing 0x3C00 to halfword view at L1[nest=0, spu=0, off=0] produces bytes [0x00, 0x3C] LE.
D-12: every named accessor returns a non-copying view (arr.base is not None).
"""
import numpy as np
import pytest

from riscv.gtx.memory import GtxMemory
from riscv.gtx.params import GTX_NEST_NUM, GTX_SPU_NUM, GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES


@pytest.fixture
def mem():
    return GtxMemory()


def test_le_byte_order_via_byte_write(mem):
    """Writing LE bytes [0x00, 0x3C] to L1 byte view appears as np.float16(1.0) in fp16 view.

    D-17 success criterion: 0x3C00 (= np.float16(1.0)) at L1[0][0][0:2] is bytes [0x00, 0x3C] LE.
    """
    mem.l1_byte(0, 0)[0] = 0x00
    mem.l1_byte(0, 0)[1] = 0x3C
    assert mem.l1_f16(0, 0)[0] == np.float16(1.0)


def test_le_byte_order_via_fp16_write(mem):
    """Writing np.float16(2.0) to L1 fp16 view produces LE bytes [0x00, 0x40]."""
    mem.l1_f16(0, 0)[0] = np.float16(2.0)
    assert mem.l1_byte(0, 0)[0] == 0x00
    assert mem.l1_byte(0, 0)[1] == 0x40


def test_l1_f16_view_invariant(mem):
    """D-12: l1_f16 returns a view, not a copy."""
    view = mem.l1_f16(0, 0)
    assert view.base is not None, "l1_f16 must return a view (D-12)"
    assert view.shape == (GTX_L1_SIZE_BYTES // 2,)
    assert view.dtype == np.float16


def test_l0_f16_view_invariant(mem):
    """D-12: l0_f16 returns a view, not a copy."""
    view = mem.l0_f16(0, 0)
    assert view.base is not None
    assert view.shape == (GTX_L0_SIZE_BYTES // 2,)


def test_slice_preserves_base(mem):
    """Slicing an fp16 view preserves base (no copy on slice)."""
    view = mem.l1_f16(0, 0)
    sub = view[100:200]
    assert sub.base is not None, "slice of view must remain a view"


def test_l1_shape(mem):
    """L1 dimensions match HW parameters."""
    assert mem.l1_byte(0, 0).shape == (GTX_L1_SIZE_BYTES,)
    assert mem.l1_byte(0, 0).dtype == np.uint8
    # All NEST × SPU regions exist
    for n in range(GTX_NEST_NUM):
        for s in range(GTX_SPU_NUM):
            assert mem.l1_byte(n, s).shape == (GTX_L1_SIZE_BYTES,)


def test_spr_dict(mem):
    """D-11: mem.spr is a unified dict[int, int]."""
    assert isinstance(mem.spr, dict)
    # Initially empty
    assert len(mem.spr) == 0
    # Address routing is just dict assignment (no logic in Phase 1; SPR-01 in P2)
    mem.spr[0x100] = 0xCAFE     # GSPR range
    mem.spr[0x500] = 0xBABE     # NSPR range (NEST encoded by P2 SPR-01)
    mem.spr[0x900] = 0xF00D     # LSPR range
    assert mem.spr[0x100] == 0xCAFE
    assert mem.spr[0x500] == 0xBABE
    assert mem.spr[0x900] == 0xF00D


def test_ddr_lazy_allocation(mem):
    """D-01: DDR is None at construction."""
    assert mem._ddr_bytes is None    # private attr access acceptable in test
```

### Example 6: `params.py` constants (verbatim from C++ macros where applicable)

```python
# src/main/python/riscv/gtx/params.py
"""Hardware parameter constants — direct port of vendor/gtx_cpp_reference/gtx/gtx_params.h.

Naming follows the C++ macro convention verbatim (per Claude's discretion).
"""
# NEST × SPU topology
GTX_NEST_NUM: int = 4
GTX_SPU_NUM: int = 16          # SPUs per NEST
GTX_SPUS_PER_NEST: int = GTX_SPU_NUM   # alias for clarity

# Memory sizes (bytes)
GTX_L0_SIZE_BYTES: int = 1024                      # 1 KB per SPU
GTX_L1_SIZE_BYTES: int = 384 * 1024                # 384 KB per SPU
GTX_L2_SIZE_BYTES: int = 16 * 1024 * 1024          # 16 MB per NEST

# DDR (D-02: capped by GTX_DDR_SIZE env var; default below)
GTX_DDR_DEFAULT_SIZE_BYTES: int = 4 * 1024 * 1024 * 1024   # 4 GiB

# DDR I/O (D-03)
GTX_DDR_BUS_WORD_BYTES: int = 32   # 32-byte bus word for GTX_DDR_REVERSED reversal

# SPR address ranges (D-11)
GSPR_BASE: int = 0x000
GSPR_END: int = 0x3FF
NSPR_BASE: int = 0x400
NSPR_END: int = 0x7FF
LSPR_BASE: int = 0x800
LSPR_END: int = 0xBFF
```

### Example 7: `encoding.py` Phase 1 scope (funct7 constants only — full disasm.inc deferred to P2)

```python
# src/main/python/riscv/gtx/encoding.py
"""GTX RoCC instruction encoding constants.

Phase 1 scope: funct7 constants (used by P2 dispatch + disasm).
Full disasm_insn_t table moves to disasm.py in Phase 2.
"""
# RoCC funct7 (selected — full table in C++ gtx_npu.h:264-353)
# gem5 simplified (operand staging via GSPR):
GTX_F7_WRSPR: int = 0x00       # WRSPR (gem5) / MM ISS-full (rs1≠0 disambiguation in P4)
GTX_F7_RDSPR: int = 0x01
GTX_F7_WSPLIT: int = 0x02      # custom1 (warp split)
GTX_F7_WJOIN: int = 0x03       # custom1 (warp join — exit semantics in P2)
GTX_F7_DISPATCH_MM: int = 0x04
GTX_F7_DISPATCH_VEC: int = 0x05
GTX_F7_DISPATCH_ACT: int = 0x06
GTX_F7_DISPATCH_DMA: int = 0x07

# ISS full (per-op funct7) — selected; full table P2:
# GTX_F7_MM = 0x00 (collides with WRSPR; resolved by insn.rs1 != 0 — P4)
# GTX_F7_MMC = 0x01
# GTX_F7_DMA_LOAD = 0x40
# GTX_F7_OPSET = 0x4A
# (...remaining 70+ constants in Phase 2)
```

## State of the Art

| Old Approach (pre-Phase-1 research) | Current Approach (Phase 1 locked) | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `numpy>=1.20,<2.0` (research/STACK.md TL;DR) | `numpy>=2.0,<3` (D-07) | 2026-05-04 discuss-phase | Enables NumPy 2.x deterministic FP16 RNE; drops cp38/cp39 |
| cibuildwheel cp38-cp312 (PYS-EXT-06) | cibuildwheel cp310-cp312 (D-08) | 2026-05-04 discuss-phase | Aligned with NumPy 2.x's cp310+ requirement |
| Pure-Python FP16 bit manipulation port (research/STACK.md §2 + research/PITFALLS.md §8) | `np.float16` view via `astype()` (D-09) | 2026-05-04 discuss-phase | ~80 LOC of careful bit code replaced by 2 one-liner helpers; risk shifted to P4/P5 strict mode |
| `recursive-include vendor *` only (existing MANIFEST.in) | `recursive-include vendor *` + `prune vendor/gtx_cpp_reference` (D-06) | This phase | Ensures `gtx_cpp_reference` C++ source NOT shipped in sdist/wheel |
| `[tool.setuptools.packages.find].include = ["riscv"]` (existing) | `["riscv", "riscv.*"]` (FOUND-03 fix) | This phase | **CRITICAL FIX** — without this, `riscv.gtx` doesn't ship in wheel (empirically verified) |

**Deprecated / outdated (in earlier research, superseded by D-decisions):**
- `importlib_resources` backport for cp38 (research/STACK.md §10) — unnecessary; cp310+ stdlib has `importlib.resources.files()`.
- Per-bank ndarray allocation per (NEST, SPU) (research/STACK.md §3) — superseded by single contiguous `(NEST, SPU, SIZE)` allocation (Pattern 1 above; matches research/ARCHITECTURE.md §2 "single contiguous allocation"). Three-dimensional shape gives free `nest_id, spu_id` indexing without computing strides.
- Memoryview-based byte-level writes (research/STACK.md §3) — superfluous; direct ndarray indexing on contiguous `np.uint8` is fast enough for Phase 1's scope (no hot path).

## Open Questions

1. **Should `pyproject.toml` `[project] dependencies` be added before or after the existing `[project.optional-dependencies] dev`?**
   - What we know: PEP 621 places `dependencies` in `[project]` table. Current pyproject.toml has no `dependencies` key (project has no current runtime deps).
   - What's unclear: stylistic placement only — both work.
   - Recommendation: add `dependencies = ["numpy>=2.0,<3"]` immediately after `requires-python = ">=3.10"` and before the `[project.urls]` section. Matches the `[project]` table ordering convention.

2. **Should `tests/gtx/__init__.py` contain `pass` or be empty?**
   - What we know: pytest discovers tests with or without `__init__.py`. Empty file is common.
   - What's unclear: `pytest --pylint` may flag a truly empty file. Existing `tests/__init__.py` has license header + 1-line docstring or similar.
   - Recommendation: Match existing convention from `tests/__init__.py`. Likely a license header is sufficient. (Verify by reading `tests/__init__.py` during planning.)

3. **Is `src/main/python/riscv/__init__.py` modification required to expose `riscv.gtx`?**
   - What we know: `from riscv.gtx import fp` works automatically once `riscv.gtx/__init__.py` exists and the package is installed (Python's standard package discovery).
   - What's unclear: whether the user wants `import riscv` to eagerly import `riscv.gtx` or lazy. Current `riscv/__init__.py` lazy-imports the C++ `_riscv` module.
   - Recommendation: do NOT modify `riscv/__init__.py`. Lazy import via `from riscv.gtx import fp` works. Eager import would force all users to load NumPy on `import riscv`, which is undesirable. CONTEXT.md `code_context` notes "lazy하게 gtx를 import하면 numpy 미설치 환경에서 import riscv가 깨질 수 있음 — try/except ImportError로 감싸 안전" but the current `riscv/__init__.py` has zero references to gtx, so this concern doesn't apply unless someone adds `from . import gtx` later. Phase 1 should leave `riscv/__init__.py` untouched.

4. **`tests/gtx/__init__.py` may shadow auto-discovery of test modules in some pytest configurations.**
   - What we know: pytest's `rootdir` + `testpaths` discovery handles both flat and nested test layouts.
   - What's unclear: pyproject.toml has `pythonpath = ["src/main/python", "examples"]` — does adding `tests/__init__.py` (existing) break the import path for `tests/gtx/test_*.py`? Looking at the current state: `tests/__init__.py` exists (591B), so `tests` is a package. `tests/gtx/__init__.py` makes `tests.gtx` a sub-package. Pytest with `testpaths = ["tests"]` and `pythonpath = ["src/main/python", "examples"]` will still discover via rootdir-relative collection.
   - Recommendation: include `tests/gtx/__init__.py` (matches convention with `tests/__init__.py`); pytest discovers via testpaths, no conflict expected. Verify by running `pytest --collect-only tests/gtx/` after creation.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| python3 (≥3.10) | Phase 1 baseline (D-08) | ✓ | 3.10.12 | — (cp310 is the floor; raise if user has older) |
| numpy (≥2.0,<3) | FP16 helpers, memory layer (D-07/D-09) | ✓ | 2.2.6 | — (D-07 locked) |
| pytest | Test execution | ✓ | 9.0.1 | — (already in pyspike `[dev]`) |
| git | Submodule registration (D-04) | ✓ | 2.34.1 | — (required for any pyspike dev workflow) |
| setuptools | Wheel build (PKG-02) | ✓ | 80.9.0 | — (already in build-system requires) |
| `dtc` (device-tree-compiler) | cibuildwheel `before-all` (existing) | (in container) | — | — (yum-installed in manylinux2014 image) |
| Internet access to `github.com/Sudo42b/gtx_spike` | `git submodule add` (D-04) | (assumed in dev/CI environments) | — | If offline: clone manually to `vendor/gtx_cpp_reference/` and register without remote URL — but D-04 explicitly chooses public repo for CI clonability. |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

All required tools are available on the dev machine and in the manylinux2014 cibuildwheel container.

## Validation Architecture

> Nyquist validation enabled (`workflow.nyquist_validation: true` in `.planning/config.json`).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.1 (with pytest-mypy, pytest-pylint, pytest-cov; verified in pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` (no separate pytest.ini); `addopts = "--pylint --mypy --cov-report=lcov"` |
| Quick run command | `pytest tests/gtx/ -x --no-header -q` (skips pylint/mypy via override; ~1-2s) |
| Full suite command | `pytest tests/gtx/ -v` (with pylint+mypy from `addopts`; ~5-10s) |
| Phase gate command | `pytest tests/ -v` (full pyspike + gtx integration; existing tests must still pass) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | All 65536 FP16 round-trip via fp16_to_fp32/fp32_to_fp16 (D-09, D-16) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_all_65536_fp16_values_idempotent -x` | ❌ Wave 0 |
| FOUND-01 | NaN inputs produce NaN outputs with stable bit pattern | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern -x` | ❌ Wave 0 |
| FOUND-01 | Subnormals round-trip exactly | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_subnormals_roundtrip -x` | ❌ Wave 0 |
| FOUND-01 | Negative zero preserved (sign bit) | unit | `pytest tests/gtx/test_fp_roundtrip.py::test_negative_zero_preserved -x` | ❌ Wave 0 |
| FOUND-02 | Writing 0x3C00 LE produces np.float16(1.0) in fp16 view (D-17) | unit | `pytest tests/gtx/test_memory_layout.py::test_le_byte_order_via_byte_write -x` | ❌ Wave 0 |
| FOUND-02 | Writing np.float16(2.0) produces LE bytes [0x00, 0x40] | unit | `pytest tests/gtx/test_memory_layout.py::test_le_byte_order_via_fp16_write -x` | ❌ Wave 0 |
| FOUND-02 | All named accessors return views (D-12; arr.base is not None) | unit | `pytest tests/gtx/test_memory_layout.py -k view_invariant -x` | ❌ Wave 0 |
| FOUND-02 | Slices of fp16 view remain views | unit | `pytest tests/gtx/test_memory_layout.py::test_slice_preserves_base -x` | ❌ Wave 0 |
| FOUND-02 | L0/L1/L2 shapes match HW params | unit | `pytest tests/gtx/test_memory_layout.py::test_l1_shape -x` | ❌ Wave 0 |
| FOUND-02 | mem.spr is a unified dict[int,int] (D-11) | unit | `pytest tests/gtx/test_memory_layout.py::test_spr_dict -x` | ❌ Wave 0 |
| FOUND-02 | DDR is None at construction (D-01) | unit | `pytest tests/gtx/test_memory_layout.py::test_ddr_lazy_allocation -x` | ❌ Wave 0 |
| FOUND-03 | `from riscv.gtx import fp, memory; from riscv.gtx.params import GTX_NEST_NUM` succeeds | smoke | `python -c "from riscv.gtx import fp, memory; from riscv.gtx.params import GTX_NEST_NUM; assert GTX_NEST_NUM == 4"` | ❌ Wave 0 (depends on package skeleton) |
| FOUND-03 | `riscv.gtx` is in built wheel | integration | `pip wheel . -w /tmp/wheel-test/ && unzip -l /tmp/wheel-test/spike-*.whl \| grep -q "riscv/gtx/__init__.py"` | ❌ Wave 0 (depends on pyproject.toml fix) |
| FOUND-03 | Clean cp310 venv install + import works | integration | `python -m venv /tmp/p1venv && /tmp/p1venv/bin/pip install /tmp/wheel-test/spike-*.whl && /tmp/p1venv/bin/python -c "from riscv.gtx import fp"` | ❌ Wave 0 |
| FOUND-04 | `vendor/gtx_cpp_reference/` registered as submodule | manual+integration | `git submodule status \| grep -q gtx_cpp_reference` | ❌ Wave 0 (depends on `git submodule add` task) |
| FOUND-04 | `MANIFEST.in` excludes vendor/gtx_cpp_reference from sdist | integration | `python -m build --sdist && tar tzf dist/spike-*.tar.gz \| grep -c gtx_cpp_reference` (expect 0) | ❌ Wave 0 |
| FOUND-04 | `vendor/gtx_cpp_reference/` not in wheel | integration | `unzip -l dist/spike-*.whl \| grep -c gtx_cpp_reference` (expect 0) | ❌ Wave 0 |
| PKG-02 | `pyproject.toml` declares `numpy>=2.0,<3` | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert any('numpy>=2.0' in d for d in t['project']['dependencies'])"` | ❌ Wave 0 |
| PKG-02 | `requires-python = ">=3.10"` | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); assert t['project']['requires-python'] == '>=3.10'"` | ❌ Wave 0 |
| PKG-02 | cibuildwheel matrix lists only cp310-cp312 | static | `python -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); b=t['tool']['cibuildwheel']['build']; assert all('cp31' in x for x in b) and not any('cp38' in x or 'cp39' in x for x in b)"` | ❌ Wave 0 |
| PKG-02 | `pip wheel .` produces valid manylinux2014_x86_64 wheel | integration | `pip wheel . -w /tmp/wheel-test/ && auditwheel show /tmp/wheel-test/spike-*.whl \| grep -q manylinux2014_x86_64` | ❌ Wave 0 (existing build pipeline; just needs to not regress) |

### Sampling Rate

- **Per task commit:** `pytest tests/gtx/ -x --no-header -q` (~1-2s; smoke + unit only, skips pylint/mypy)
- **Per wave merge:** `pytest tests/gtx/ -v` (full unit suite with pylint/mypy, ~5-10s)
- **Phase gate (before `/gsd:verify-work`):** `pytest tests/ -v` (full pyspike test suite + new gtx tests must all pass) AND wheel build verification: `pip wheel . -w /tmp/wheel-test/ && unzip -l /tmp/wheel-test/spike-*.whl | grep gtx`

### Wave 0 Gaps

All test infrastructure for Phase 1 is missing. The plan must include creation of:

- [ ] `tests/gtx/__init__.py` — enables pytest collection (or empty marker; verify against `tests/__init__.py` style)
- [ ] `tests/gtx/test_fp_roundtrip.py` — covers FOUND-01 (5 test functions: all-values idempotent, NaN preservation, subnormals, -0.0, known values)
- [ ] `tests/gtx/test_memory_layout.py` — covers FOUND-02 (LE byte order, view invariants, shapes, SPR dict, DDR lazy)
- [ ] `src/main/python/riscv/gtx/__init__.py` — re-export module (covers FOUND-03 import-path)
- [ ] `src/main/python/riscv/gtx/params.py` — HW constants (consumed by tests)
- [ ] `src/main/python/riscv/gtx/encoding.py` — funct7 constants (Phase 1 stub; P2 fills full)
- [ ] `src/main/python/riscv/gtx/fp.py` — FP16/FP32 helpers (D-09)
- [ ] `src/main/python/riscv/gtx/memory.py` — `GtxMemory` class (D-10/D-11/D-12)
- [ ] `src/main/python/riscv/gtx/ddr.py` — DDR lazy alloc + env handling (D-01/D-02)
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — empty package marker
- [ ] `pyproject.toml` patches — `numpy>=2.0,<3` dep, `requires-python = ">=3.10"`, classifiers, cibuildwheel matrix, packages.find include fix
- [ ] `MANIFEST.in` patch — `prune vendor/gtx_cpp_reference`
- [ ] Submodule registration — `git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference`

**Framework install:** Not needed — pytest 9.0.1 already installed (`pyproject.toml [project.optional-dependencies].dev` has `pytest`).

## Sources

### Primary (HIGH confidence — empirically verified or official authoritative)

- **NumPy 2.2.6 local installation** — empirical verification of all FP16 view, round-trip, NaN, subnormal, byte-order, view-base claims. Reproducible via the inline scripts in §"Pattern 2" and §"FP16 Conversion (D-09)" examples.
- **Setuptools 80.9.0 local installation** — empirical verification that `find_packages(include=["riscv"])` does NOT auto-discover `riscv.gtx`; `include=["riscv", "riscv.*"]` does. Reproducible via the script in §"Anti-Patterns".
- **NumPy `np.frombuffer` doc** — view semantics, dtype byte-order spec: <https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html>
- **NumPy `np.ndarray.view` doc** — view-not-copy semantics, last-axis-contiguous constraint (since 1.23): <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html>
- **NumPy 2.0 release notes** — Python 3.9+ minimum, NEP 50 promotion, manylinux2014 retention: <https://numpy.org/devdocs/release/2.0.0-notes.html>
- **NumPy 2.2.6 PyPI** — confirms cp310-cp312 manylinux2014_x86_64 wheels published: <https://pypi.org/project/numpy/2.2.6/>
- **CONTEXT.md (D-01..D-17)** — locked Phase 1 decisions, single source of truth for scope.
- **pyproject.toml (current state)** — verified file at `/mnt/e/14_NIGHTLY/pyspike/pyproject.toml`; deltas listed in Example 3.
- **MANIFEST.in (current state)** — verified file at `/mnt/e/14_NIGHTLY/pyspike/MANIFEST.in`; delta listed in §"Pitfall: MANIFEST.in".
- **Existing `vendor/spike` submodule** — confirms `.gitmodules` mechanism is already in use (`git submodule status` output: `20feb9c2... vendor/spike`). New `vendor/gtx_cpp_reference` follows identical pattern.
- **C++ ground-truth (referenced via D-04 submodule once added)** — `vendor/gtx_cpp_reference/gtx/gtx_npu.h:89-151` (FP conversion source for D-09 fallback); `vendor/gtx_cpp_reference/gtx/gtx_params.h` (params.py port source).

### Secondary (MEDIUM confidence — official docs, single-source-verified)

- **cibuildwheel `before-all` doc** — runs inside manylinux container, executes shell commands (incl. git submodule update): <https://cibuildwheel.pypa.io/en/stable/options/>
- **Setuptools `packages.find` doc** — `include` glob behavior, namespace package handling: <https://setuptools.pypa.io/en/latest/userguide/package_discovery.html>
- **Existing pyspike research artifacts** — `.planning/research/STACK.md`, `.planning/research/PITFALLS.md`, `.planning/research/ARCHITECTURE.md` (NumPy 1.x recommendations REVERSED by D-07; FP16 storage and architecture sections largely RETAINED).

### Tertiary (LOW confidence — needs validation if relied on)

- **NumPy 2.0 NaN bit-pattern preservation in `astype(np.float16)`** — empirically verified on 2.2.6 (HIGH for that version) but undocumented in 2.0 release notes. Future NumPy 2.x versions may canonicalize NaN bit patterns differently; planner should add a NumPy version-specific test guard or pin a specific 2.x range if NaN preservation becomes a strict requirement. **For Phase 1 scope (helper-level round-trip), NaN bit preservation is desired but not strictly required by D-09** — the success criterion only asserts "NaN inputs produce NaN with stable bit pattern" which the test verifies for the current NumPy version.

## Metadata

**Confidence breakdown:**
- Standard stack (NumPy 2.x, cp310, setuptools): HIGH — versions verified on local install + PyPI; D-07/D-08 locked.
- FP16 view conversion (D-09): HIGH at helper level (all 65536 round-trip empirically verified including NaN bit-pattern preservation on NumPy 2.2.6); MEDIUM at op level (P4/P5 strict-mode measurement still pending).
- Memory layer view semantics (D-10/D-12): HIGH — `view().base is not None` invariant verified on numpy 2.2.6 for view + slice + reshape combinations.
- Submodule registration (D-04): HIGH — existing `vendor/spike` proves the mechanism works in this repo; `Sudo42b/gtx_spike` is public per CONTEXT.md.
- Packaging (PKG-02 deltas): HIGH — exact diff verified against current pyproject.toml; `include = ["riscv", "riscv.*"]` necessity empirically verified on setuptools 80.9.0.
- MANIFEST.in `prune` exclusion: HIGH — `prune` is canonical setuptools directive; existing `vendor/spike` registration in `.gitmodules` confirms the mechanism doesn't conflict.
- cibuildwheel `before-all` git submodule init: HIGH — verified docs; minor caveat (sdist exclusion vs build tree submodule access) called out in Example 2.
- Pitfalls (1, 2, 8, 13): HIGH — direct port from existing `.planning/research/PITFALLS.md` with Phase 1-specific verification commands added; new pitfalls (pyproject.toml include glob; MANIFEST.in vendor sweep) discovered via empirical verification.
- Wave 0 test gap inventory: HIGH — every required file enumerated; existing `tests/test_extension.py` pattern is the template.

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 for NumPy/cp/setuptools claims (30 days; stable ecosystem). Re-verify NumPy 2.x version availability before P6 (wheel ship) since 2.x line moves quickly.

---

*Phase: 01-foundation*
*Research generated: 2026-05-04 via /gsd:research-phase 1 (downstream of /gsd:discuss-phase 1)*
