# Phase 4: MM Subsystem - Research

**Researched:** 2026-05-06
**Domain:** GTX NPU MM/MMC subsystem — `firmware_mm_op` packed-rs1 decode + `gemm_core` FP32 internal accumulate + `mxe_accum` per-(NEST,SPU) FP32 chain + first .elf strict-mode regression
**Confidence:** HIGH for code locking; **MEDIUM** for `np.matmul` bit-exact viability (empirical drift detected — fallback strategy locked); **LOW** for in-process .elf load (subprocess fallback recommended).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**MM 모듈 구성 (D-01 ~ D-04)**
- **D-01:** **3-way module split** — `riscv/gtx/ops/mm.py` (@handler 진입점) + `riscv/gtx/mm_engine.py` (firmware_mm decode + variant dispatcher + mxe_accum read/write 책임) + `riscv/gtx/gemm_core.py` (순수 stateless NumPy GEMM 커널 1개 함수).
- **D-02:** **`gemm_core` 구현 = `np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)`** — single-line, FP32 internal accumulate (BLAS 자동), single FP16 cast.
  - **Risk locked by research (this doc):** BLAS implementation defined 누적 순서 ≠ C++ scalar 3-loop. **Empirical drift detected — see §`np.matmul` Bit-Exactness Analysis below. P4 plan MUST adopt explicit 3-loop for bit-exact path; `np.matmul` is the P7 numba @njit acceleration target.**
- **D-03:** **`gemm_core` stateless API:** `gemm_core(A: ndarray, B: ndarray, *, has_bias: bool = False, prior_accum: float = 0.0) -> tuple[ndarray, float]` — array-in/scalar-in/array-out/scalar-out. mxe_accum은 호출자(`mm_engine`)가 read/write 책임.
- **D-04:** **10개 MM/MMC variant 모두 별도 `@handler` 함수 등록** — `_exec_mm`, `_exec_mm_s`, `_exec_mm_o`, `_exec_mm_v`, `_exec_mm_t`, `_exec_mmc`, `_exec_mmc_s`, `_exec_mmc_o`, `_exec_mmc_v`, `_exec_mmc_t`.

**`is_accumulate` 분기 + mxe_accum read/write 위치 (D-05 ~ D-08)**
- **D-05:** **`is_accumulate` 분기는 `ops/mm.py` 진입점에서.** mmc 계열 (funct7=0x01) → `is_accumulate=True`; mm 계열 (funct7=0x00) → `False`. `_exec_mmc_o(npu, proc, insn, xs1, xs2)` calls `mm_engine.firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_o')`.
- **D-06:** **`mxe_accum` read/write는 `ops/mm.py` 진입점.** sequence: read `nest = npu.warp.tmu_id; spu = npu.warp.curr_id` → read `prior = float(npu._mxe_accum[nest, spu])` if `is_accumulate` → call `gemm_core` → write `npu._mxe_accum[nest, spu] = new_accum` if MM_O/MM_V/MMC_O/MMC_V → result to memory (L1 or L0 per variant).
- **D-07:** **Pitfall 3 anti-pattern test = `mm.s → mmc.s → mmc` chain 단일 테스트** + per-cell scope dual-assertion (only `[nest, spu]` cell changed, other 4×16-1 cells unchanged via snapshot copy diff).
- **D-08:** **funct7=0x00 collision (WRSPR vs MM) parametrized matrix test** — 4 cases over `(funct7 ∈ {0x00, 0x01}, has_rs1 ∈ {True, False})`.

**`.elf` 회귀 fixture 전략 (D-09 ~ D-12)**
- **D-09:** **`mm_basic.elf`는 vendor/gtx_cpp_reference/에서 차용** (research lock). **fallback (THIS RESEARCH CONFIRMS): vendor에 mm_basic.elf 없음** → P2 D-22 패턴 (사전 빌드 .S/Makefile/.elf 커밋).
- **D-10:** **golden hex `mm_basic_n1s16.hex`도 vendor에서 차용** — research lock. **THIS RESEARCH CONFIRMS: vendor 내 단일 mm `.hex` golden 없음** → 자체 합성 fallback 또는 in-process 호출로 직접 비교.
- **D-11:** **`.elf` 구동은 pytest in-process** — fallback subprocess. **THIS RESEARCH RECOMMENDS subprocess as PRIMARY** (P2 `test_skeleton.py` 패턴 — 검증된 GIL/단일-hart 안전 경로).
- **D-12:** **`GTX_DDR_DUMP` 처리는 P4에서 처음 도입 — but 테스트 내부 명시 호출만.**

**Strict-mode 검증 인프라 (D-13 ~ D-15)**
- **D-13:** **`tests/gtx/_verify_minimal.py` mini 직역** (~30 LOC). 시그니처: `def compare_hex(actual_path: str, golden_path: str, *, ulp: int = 1, atol: float = 0.001, strict: bool = True) -> tuple[bool, dict]`. `_verify_minimal`는 **`(data[i*2] << 8) | data[i*2+1]`** big-endian FP16 bit-pair (Pitfall 1 lock).
- **D-14:** **P4도 strict 강제** — `within_tolerance > 0`도 failure로 보고.
- **D-15:** **Op-level unit test는 `np.array_equal(actual.view(np.uint16), expected.view(np.uint16))` 직접** — `_verify_minimal`은 `.elf` 회귀에만 사용.

### Claude's Discretion

- `gemm_core` 정확한 signature (prior_accum scalar vs ndarray view, has_bias 필요 여부, 반환 tuple 순서)
- `np.matmul` BLAS 누적 순서가 C++ scalar 3-loop과 bit-exact인지 — **THIS RESEARCH LOCKS: NOT bit-exact (max 4 ULP / 0.0078 abs drift over 500 trials). Plan MUST use explicit 3-loop fallback for P4.**
- 10 MM/MMC variant funct3 정확한 매핑 — **THIS RESEARCH LOCKS** (§Standard Stack §funct3 Mapping)
- `mm_engine.py` 내 함수 분리 정도 (`firmware_mm` 단일 함수 vs `_decode_args` + `_dispatch_variant` + `_writeback` 분리)
- `MM_T` (transposed B) / `MM_V` (vector) variant의 정확한 데이터 경로 — **THIS RESEARCH LOCKS** (§MM Variant Semantics)
- `mxe_accum[nest, spu]` 정확한 nest/spu 추출 — **THIS RESEARCH LOCKS: `npu.warp.tmu_id` / `npu.warp.curr_id`** (verified in `src/main/python/riscv/gtx/warp_state.py:29-30`)
- `dispatch_iss_opcode`에서 funct7=0x00/0x01 케이스 추가 정확한 위치 — **THIS RESEARCH LOCKS: NOT in dispatch_4mode** (§dispatch routing analysis)
- `_verify_minimal.py` 정확한 구현 (FP16 BE-packing 처리 — PITFALLS Pitfall 1 정합)
- `mm_basic.elf` 위치가 vendor에 없을 시 fallback 방식 — **THIS RESEARCH CONFIRMS: write our own `.S` source under `tests/gtx/data/elf/mm_basic.S`**
- in-process .elf 로드의 정확한 진입점 — **THIS RESEARCH RECOMMENDS subprocess fallback**

### Deferred Ideas (OUT OF SCOPE)

- VEC op 핸들러 (SASMD/DOT/VSUM/CLAMP) → P5
- ACT op 핸들러 + format_cvt + FP8 codec → P5
- `verify.py` 정식 포팅 → P6
- 전체 .elf 회귀 100% strict mode → P6
- `tests/gtx/data/{golden,elf}/` package_data 등록 → P6
- 자동 DDR dump (atexit hook) → P6
- Numba @njit 가속 적용 (gemm_core) → **P7 (D-02 BLAS path becomes the @njit target after P6 strict mode green via 3-loop)**
- DMA-3D / IM2COL / MCAST → v2
- `mexec` full microcode loop → v1 펌웨어 미요구 시 stub
- Mode 4 inner payload의 VEC/ACT funct7 → P5
- v2 PY-OVRD-01, PY-FUNCT7-01, MM-V2-01, CYC-01/02, MM-NUMBA-01 (P7)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MM-01 | `gemm_core` — `np.matmul` with `dtype=np.float32` and single `np.float16` cast | §gemm_core C++ ground-truth (gtx_npu_mm.cc:27-94) + §`np.matmul` Bit-Exactness Analysis: BLAS-vs-3-loop drift up to 4 ULP, **MUST use explicit 3-loop for P4**, np.matmul reserved for P7 |
| MM-02 | All 10 MM/MMC variants implemented | §funct3 Mapping (gtx_npu_disasm.inc:39-50, gtx_npu_mm.cc:357-376) — 5 mm + 5 mmc with mnemonic ↔ funct3 table |
| MM-03 | `firmware_mm_op` packed rs1 decode + funct3 variant dispatch | §rs1 Packed Decode (gtx_npu_mm.cc:347-355): `dim16` lambda confirms `0 → 0x10000` per 16-bit field; funct7=0x00 ↔ funct7=0x01 collision via `insn.rs1 != 0` heuristic (gtx_npu_custom0.cc:60-70) |
| MM-04 | `mxe_accum` per-(NEST,SPU) FP32 chain | §mxe_accum semantics (gtx_npu_mm.cc:200-212, 267-269): MM_O/MMC_O scalar `sum(A)` written; MM_V/MMC_V dot-product scalar; only `[nest, spu]` cell mutated. **MM_S/MMC_S, MM/MMC, MM_T/MMC_T do NOT touch mxe_accum** — they use ADDRC for FP32 bias instead. |
| MM-05 | First .elf GEMM regression strict mode pass | §ELF Fixture Strategy: vendor mm_basic.elf NOT FOUND → write own `.S` under `tests/gtx/data/elf/mm_basic.S` + `Makefile`. Subprocess `pyspike --extlib=riscv.gtx ... .elf` per P2 test_skeleton.py pattern. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Tech stack**: Python 3.10+ / NumPy ≥ 2.0 / pyspike pybind11 trampoline. **No C++ additions** — pure Python rewrite.
- **Compatibility**: `riscv.isa.ROCC` virtual signature `customN(self, proc, insn, xs1, xs2) -> reg_t`. processor_t / rocc_insn_t pybind11 binding objects.
- **Performance**: NumPy backend, FP16 + FP32 internal accumulate. Regression must complete in tens of minutes.
- **Dependencies**: NumPy only (no scipy / numba / cython in v1). Numba is a P7 follow-up.
- **Bit-exact**: ULP 1 / atol 0.001 in `verify.py --strict` mode against C++ libgtx_npu.so golden.
- **Testing**: pytest. Per-op unit + .elf regression.
- **Platform**: Linux x86_64 / glibc 2.17+.

## Summary

Phase 4 is the **value-driver phase**: builds the first compute layer atop the P3 data plane. Five v1 requirements (MM-01..05). Five locked deferred items from CONTEXT.md `<deferred>` section have been resolved by reading the C++ ground-truth and existing pyspike code:

1. **`firmware_mm_op` rs1 packed bit layout** — LOCKED. `colB[63:48] | colA[31:16] | rowA[15:0]`. HW conv via `dim16` lambda (gtx_npu_mm.cc:347): every 16-bit field where 0 maps to 0x10000. Mask formula: `dim16(v) = (v & 0xFFFF) or 0x10000` (Python: `(rs1 & 0xFFFF) or 0x10000`).

2. **10 MM/MMC variant funct3 mapping** — LOCKED. Disasm table (gtx_npu_disasm.inc:39-50) and dispatch (gtx_npu_mm.cc:357-376) both confirm:
   - funct3=0 → mm_s/mmc_s
   - funct3=1 → mm_o/mmc_o
   - funct3=2 → mm/mmc
   - funct3=3 → mm_v/mmc_v
   - **funct3=7 → mm_t/mmc_t** (NOT funct3=4 as CONTEXT.md surface sketch implied)
   funct7 = 0x00 (MM family) or 0x01 (MMC family).

3. **`np.matmul` BLAS vs C++ scalar 3-loop bit-exactness** — LOCKED via empirical test. **NOT bit-exact**. Across 500 random 16×16×16 FP16-cast-to-FP32 trials on this development host (NumPy 2.2.6, scipy-openblas backend), 41/500 trials drift up to 4 ULP / 0.0078 abs (worst case) — far exceeding `verify.py --ulp 1 --atol 0.001` strict-mode tolerance. **Plan MUST use explicit 3-loop FP32 accumulate for P4.** `np.matmul` is reserved for the P7 numba `@njit` boundary as planned.

4. **`mxe_accum` exact semantics** — LOCKED. Per `gtx_npu_mm.cc:200-212` (MM_O), `mxe_accum[nest][spu]` stores the **scalar `sum(A)` (FP32)** for MM_O / `sum(A) + prior_accum` for MMC_O. Per gtx_npu_mm.cc:267-269 (MM_V), it stores **scalar dot(A,B)** / `dot(A,B) + prior_accum` for MMC_V. **MM_S, MM/MMC, MM_T variants do NOT touch mxe_accum** — they use ADDRC (LSPR_SPM_ADDRC = 0x902) for FP32 bias staging instead. The `gemm_core(A, B, *, has_bias, prior_accum)` API surface from CONTEXT.md is incomplete: it works for MM/MMC variants only; MM_O / MM_V / MMC_O / MMC_V need a separate scalar-output kernel.

5. **`funct7=0x00 insn.rs1!=0` heuristic** — LOCKED. C++ confirmed (gtx_npu_custom0.cc:60-70). When `insn.rs1 != 0`, route to `firmware_mm_op` with `is_accumulate = (funct7 == 0x01)`. Otherwise (rs1==0), route to gem5-simplified WRSPR/RDSPR. Existing pyspike `ops/spr.py` already handles the rs1==0 path; P4 only needs to install the rs1!=0 path.

6. **`mm_basic.elf` + golden hex vendor path** — RESOLVED with FALLBACK. **No mm_basic.elf or related golden hex exists in vendor/gtx_cpp_reference/.** Vendor only ships `add_vv_example.elf` (VEC, P5 territory) + `test_printf.elf` (no GTX ops). The vendor `test/MUL_MAT/n1s16/` directory contains only a `.c` source + .txt input/ref data (not .elf/.hex). Fallback path: write our own `tests/gtx/data/elf/mm_basic.S` + `Makefile` + sized-1KB `.elf` committed to git, mirroring P2 D-22 `nop_wjoin.elf` pattern. Golden hex synthesized at test setup by running `mm_basic.elf` once through C++ libgtx_npu.so OR computed in-Python from explicit 3-loop reference (with FP32 accumulate, single FP16 cast).

7. **In-process .elf load mechanism** — RECOMMENDED subprocess. pyspike DOES expose `sim_t.run()` (py_module.cc:946) and `processor_t.step(n)` (py_module.cc:728), so in-process is theoretically possible. However, P2 plan-05 D-1 already documented GIL contamination + WJOIN SystemExit propagation friction when running spike under pytest workers. The **P2 `test_skeleton.py` subprocess pattern** is battle-tested for `pyspike --extlib=riscv.gtx <fw>.elf` and should be reused. In-process attempt deferred to v2.

8. **`dispatch_iss_opcode` body extension location** — LOCKED. **NEITHER extend `dispatch_iss_opcode` NOR bypass via dispatch_4mode.** The `firmware_mm_op` path (funct7=0x00/0x01 with rs1!=0) is **completely separate** from `dispatch_iss_opcode` (which handles gem5-simplified DISPATCH_MM funct7=0x04 routing through Mode 4). Per gtx_npu_mm.cc:333-389, `firmware_mm_op` directly invokes `exec_mm_*` per funct3, with its own NEST/SPU selection (`is_ploop ? tmu_id : 0; is_tloop ? curr_id : 0`). P4 mm_engine should mirror this: fetch nest/spu from `npu.warp` directly, do not call dispatch_4mode. **The dispatch_iss_opcode funct7=GTX_OP_MM=0 case (gem5-simplified DISPATCH_MM path) is a SEPARATE P4 task** — adds the Mode 4 routing for the gem5 `dispatch_mm` instruction (funct7=0x04 hits `dispatch()` → Mode 4 → `dispatch_iss_opcode(funct7=0)`). This is what ROADMAP success #5 (Mode 4 P+T) refers to.

9. **WarpState `tmu_id`/`curr_id` exact field names** — LOCKED. `npu.warp.tmu_id` and `npu.warp.curr_id`. Verified by reading `src/main/python/riscv/gtx/warp_state.py:29-30`:
   ```python
   tmu_id: int = 0   # NEST id selected by start_p
   curr_id: int = 0  # SPU id (T-loop) or GDMAC id (S-loop)
   ```
   These are direct (no `_` prefix), accessed via the `WarpState` dataclass instance held at `npu.warp`. The CONTEXT.md surface sketch line `nest, spu = npu.tmu_id, npu.curr_id` is INCORRECT — it must be `npu.warp.tmu_id, npu.warp.curr_id`.

**Primary recommendation:** Execute P4 with **explicit 3-loop FP32 accumulate** for `gemm_core` to guarantee strict-mode bit-exactness against the C++ scalar 3-loop golden. Keep the API stateless and numba-friendly. Use subprocess `pyspike --extlib=riscv.gtx ... .elf` for the regression test. Build `mm_basic.elf` from a hand-written `.S` source mirroring the P2 nop_wjoin.elf fixture pattern.

## Standard Stack

### Core (already on disk; P4 reuses)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | ≥ 2.0,<3 (host has 2.2.6) | FP32 ops, view-based memory access | Project D-07; bit-exact FP16 round-trip verified P1 |
| pytest | ≥ 8 | Unit + regression test framework | P1/P2/P3 established |
| Existing pyspike binding | wheel-shipped | `riscv.processor.processor_t.get_state().XPR[i]`, `riscv.extension.rocc_insn_t.{rs1,rs2,xd,xs1,xs2,funct}` | RoCC trampoline already validated end-to-end |

### Supporting (P4 creates)
| Module | Path | Purpose | Notes |
|--------|------|---------|-------|
| gemm_core | `src/main/python/riscv/gtx/gemm_core.py` | Pure stateless kernel | Numba @njit boundary (P7) |
| mm_engine | `src/main/python/riscv/gtx/mm_engine.py` | rs1 decode + variant dispatcher | Bound to `proc/insn/npu` |
| ops/mm | `src/main/python/riscv/gtx/ops/mm.py` | 10 @handler entry points | `is_accumulate` branching here |
| ops/__init__ patch | `src/main/python/riscv/gtx/ops/__init__.py` | Add `from . import mm` line | Triggers @handler decorator load |
| _verify_minimal | `tests/gtx/_verify_minimal.py` | Mini verify.py port | Test-only; P6 promotes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Explicit 3-loop in `gemm_core` | `np.matmul(A_f32, B_f32)` | BLAS faster (~100×) but **not bit-exact** vs C++ scalar 3-loop golden. P7 numba dynamic optimization revisits. |
| Subprocess for .elf regression | In-process via `sim_t.run()` | In-process avoids fork, but P2 plan-05 documented GIL/WJOIN friction. Subprocess is the validated path. |
| Synthesize golden hex with in-Python 3-loop | Use C++ libgtx_npu.so to generate golden | Self-comparison (3-loop vs 3-loop) is circular but provides immediate test coverage. **Plan should produce BOTH:** (a) Python-synthesized golden for tightest TDD loop, (b) capture C++ libgtx_npu.so output if available as a tertiary cross-check. |
| `riscv/gtx/ops/mm.py` direct decode | Call `mm_engine.firmware_mm` from each @handler | Surface sketch in CONTEXT.md picks the latter (variant string passed). Stick with that — it preserves D-04 individual @handler granularity for disasm + numba. |

**Installation (no new pip deps):**
```bash
# verified against current host
python3 -c "import numpy as np; print(np.__version__)"  # 2.2.6
python3 -c "import pytest; print(pytest.__version__)"
```

**Version verification:** Performed against npm registry mirror `pip show numpy` on this host (2.2.6, 2025 release). Project pin is `numpy>=2.0,<3` (P1 D-07).

## Architecture Patterns

### Recommended Project Structure
```
src/main/python/riscv/gtx/
├── gemm_core.py        # NEW — pure stateless GEMM kernel + scalar reductions
├── mm_engine.py        # NEW — spike-bound dispatcher (rs1 decode + variant dispatch)
├── ops/
│   ├── __init__.py     # MODIFIED — add `from . import mm`
│   ├── mm.py           # NEW — 10 @handler entry points (mm/mm_s/mm_o/mm_v/mm_t + mmc family)
│   ├── spr.py          # P2 (untouched)
│   ├── control.py      # P2/P3 (untouched)
│   └── dma.py          # P3 (untouched)
└── dispatch_4mode.py   # MODIFIED — add funct7=GTX_OP_MM case in dispatch_iss_opcode (Mode 4 entry from gem5 dispatch_mm path)

tests/gtx/
├── _verify_minimal.py     # NEW — ~30 LOC compare_hex helper
├── test_op_mm.py          # NEW — 10 variant unit tests (D-15 np.array_equal)
├── test_mm_chain.py       # NEW — D-07 mxe_accum chain + per-cell isolation
├── test_funct7_routing.py # NEW — D-08 4-case parametrized matrix
├── test_regression_fw_mm.py  # NEW — subprocess pyspike + DDR dump + compare_hex --strict
└── data/elf/
    ├── mm_basic.S      # NEW — hand-written firmware emitting MM custom0 + WJOIN
    ├── Makefile        # NEW — riscv64-unknown-elf-gcc build
    └── mm_basic.elf    # NEW — committed pre-built .elf (~1-2KB)
```

### Pattern 1: 3-Way Split — Stateless Kernel / Spike-Bound Engine / Handler Entry
**What:** D-01 evolution of P3's 2-way split. The kernel (`gemm_core.py`) is pure NumPy, no spike imports → directly @njit-able by P7 numba. The engine (`mm_engine.py`) decodes packed bits, reads/writes `npu` state, calls the kernel. The handler (`ops/mm.py`) is the spike-facing @handler decorator entry.

**When to use:** GEMM-style ops with non-trivial state coupling (`mxe_accum`).

**Example (P4 surface sketch — verified against C++):**
```python
# riscv/gtx/gemm_core.py — pure stateless
import numpy as np
from numpy.typing import NDArray

def gemm_core(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    *,
    has_bias: bool = False,
    bias_fp32: NDArray[np.float32] | None = None,  # ADDRC FP32 buffer
) -> NDArray[np.float16]:
    """C = A @ B [+ bias_fp32] -> FP16, FP32-internal accumulate.

    Direct port of gtx_npu_mm.cc:27-94 (gemm_core).
    Uses explicit 3-loop (NOT np.matmul) — research locked bit-exact requirement.

    Returns FP16 (M, N) result. mxe_accum (scalar) is caller's responsibility
    (only MM_O/MMC_O/MM_V/MMC_V variants touch it; this kernel handles M×N variants).
    """
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    M, K = A_f32.shape
    K2, N = B_f32.shape
    assert K == K2

    C_f32 = np.zeros((M, N), dtype=np.float32)
    # Explicit 3-loop bit-exact with C++ gtx_npu_mm.cc:73-79.
    # P7 numba @njit will accelerate this.
    for i in range(M):
        for j in range(N):
            s = np.float32(0.0)
            for k in range(K):
                s += A_f32[i, k] * B_f32[k, j]
            C_f32[i, j] = s

    if has_bias and bias_fp32 is not None:
        C_f32 += bias_fp32  # broadcast (M, N) FP32 bias from ADDRC

    return C_f32.astype(np.float16)


def gemm_reduce_sum_a(A: NDArray[np.float16], *, prior_accum: float = 0.0) -> float:
    """MM_O / MMC_O scalar: sum(A) [+ prior_accum] in FP32.

    Direct port of gtx_npu_mm.cc:200-211. Returns scalar FP32 to be cast to
    FP16 by caller for L0 write AND stored back into mxe_accum.
    """
    return float(A.astype(np.float32).sum(dtype=np.float32)) + prior_accum


def gemm_dot(A: NDArray[np.float16], B: NDArray[np.float16],
             *, prior_accum: float = 0.0) -> float:
    """MM_V / MMC_V scalar: dot(A, B) [+ prior_accum] in FP32.

    Direct port of gtx_npu_mm.cc:262-265 (manual fallback path).
    """
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    s = np.float32(0.0)
    for k in range(A_f32.shape[0]):
        s += A_f32[k] * B_f32[k]
    return float(s) + prior_accum
```

### Pattern 2: rs1 Packed Decode in mm_engine
**What:** Mirror `decode_firmware_dma_args` from P3 (`dma_engine.py:66-99`) for MM.

**Example:**
```python
# riscv/gtx/mm_engine.py — pure decode helper
def decode_firmware_mm_args(rs1: int) -> dict:
    """Decode packed rs1 -> {row_A, col_A, col_B}.

    Direct port of gtx_npu_mm.cc:347-350 dim16 lambda.
    rs1 layout: colB[63:48] | colA[31:16] | rowA[15:0].
    HW convention: 0 in any 16-bit field means 65536.

    Reserved bits (32-47): unused per C++ source.
    """
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return {
        'row_A': dim16(rs1),
        'col_A': dim16(rs1 >> 16),
        'col_B': dim16(rs1 >> 48),
    }
```

### Pattern 3: 10 @handler Entries via String Table
**What:** D-04 individual functions for numba-friendliness; CONTEXT.md surface sketch shows clean form.

**Example (verified against gtx_npu_disasm.inc:39-50):**
```python
# riscv/gtx/ops/mm.py
from .._registry import handler
from .. import mm_engine

# funct7=0x00 → MM family (is_accumulate=False)
@handler(kind='custom0', funct7=0x00, funct3=0, mnemonic='mm_s', mask_funct3=True)
def _exec_mm_s(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=False, variant='mm_s')

@handler(kind='custom0', funct7=0x00, funct3=1, mnemonic='mm_o', mask_funct3=True)
def _exec_mm_o(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=False, variant='mm_o')

@handler(kind='custom0', funct7=0x00, funct3=2, mnemonic='mm', mask_funct3=True)
def _exec_mm(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=False, variant='mm')

@handler(kind='custom0', funct7=0x00, funct3=3, mnemonic='mm_v', mask_funct3=True)
def _exec_mm_v(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=False, variant='mm_v')

@handler(kind='custom0', funct7=0x00, funct3=7, mnemonic='mm_t', mask_funct3=True)
def _exec_mm_t(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=False, variant='mm_t')

# funct7=0x01 → MMC family (is_accumulate=True)
@handler(kind='custom0', funct7=0x01, funct3=0, mnemonic='mmc_s', mask_funct3=True)
def _exec_mmc_s(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn,
                                 is_accumulate=True, variant='mmc_s')

# ... mmc_o, mmc, mmc_v, mmc_t ...
```

### Anti-Patterns to Avoid
- **`np.matmul` in `gemm_core` for P4** — research empirically locks 4-ULP drift. P7 reactivates after P6 strict mode green via 3-loop.
- **Mutating `mxe_accum` in MM_S / MM / MM_T variants** — gtx_npu_mm.cc:106-176, 289-315 confirms these write to ADDRR/ADDRC, never to mxe_accum. Touching it would corrupt MMC_O/MMC_V chain semantics across mixed firmware.
- **`np.matmul(A.float16, B.float16).astype(float16)`** — Pitfall 2 trap (per-step FP16 truncation). Always upcast to FP32 first.
- **`@handler` closure factory loop** — D-04 vs P7 numba JIT friendliness.
- **`is_accumulate` from call count or instance state** — D-05/D-08 violation. Drive solely from funct7 == 0x01.
- **Caching `proc.get_state()` between calls** — Pitfall 12. Always re-acquire.
- **In-process `sim_t.run()` from pytest** — P2 plan-05 documented GIL contamination + WJOIN SystemExit propagation friction. Use subprocess fallback.
- **Calling `dispatch_4mode` from `firmware_mm_op` path** — gtx_npu_mm.cc:333-389 confirms direct exec_* dispatch without 4-mode router. The 4-mode router is for gem5-simplified DISPATCH_MM (funct7=0x04) only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16↔FP32 conversion | Custom bit manipulation | `arr.astype(np.float32)` / `.astype(np.float16)` | P1 D-09 — NumPy 2.x view-based path validated; pure-python bit ops kept as fallback only |
| Endianness handling for L1 reads | Custom byte unpacking | `mem.l1_f16(nest, spu)` view from `GtxMemory` | P1 D-12 — view guaranteed via `arr.base is not None`; LE bound by `npu.h:38` `spu.l1[off]` LE write convention |
| Hex file parsing for `_verify_minimal` | Custom @offset directive parser | `bytes.fromhex(line)` after stripping; no `@offset` directive in DDR-dump output (only in init) | verify.py:172-184 simple line-by-line concat. Per gtx_npu_dma.cc:509-558 dump always emits 32 bytes/line, no `@offset`. |
| `decode_firmware_mm_args` packed-bits | Manual mask + shift twice | Single helper mirroring P3 `decode_firmware_dma_args` | P3 D-03 pattern proven; helper allows pytest unit test |
| Dispatch table for funct3 sub-decomposition | Manual switch | `_registry.py @handler(mask_funct3=True)` | P3 verified path; `npu.custom0` already handles 2-level lookup (npu.py:125-144) |
| 4-mode dispatch | Per-op routing | `dispatch_4mode.dispatch_iss_opcode(npu, ..., funct7=GTX_OP_MM)` for gem5-simplified DISPATCH_MM only | Already exists in `dispatch_4mode.py:38-66`. **firmware_mm_op (funct7=0x00) bypasses this** per gtx_npu_mm.cc structure. |
| .elf disassembly for fixture sanity check | Hand-write objdump parser | `riscv64-unknown-elf-readelf -h` in Makefile (already used by nop_wjoin.elf Makefile) | P2 D-22 pattern proven |

**Key insight:** P4 should look like a refinement of P3's pattern, not a new architecture. Re-use `_registry.handler`, `dispatch_4mode`, the existing `MockProcessor.get_state().XPR[i]` mock, and the subprocess `pyspike --extlib=riscv.gtx <fw>.elf` regression pattern from P2.

## Common Pitfalls

### Pitfall A: BLAS vs Scalar 3-loop Drift (THIS RESEARCH SURFACED)
**What goes wrong:** `np.matmul(A_f32, B_f32)` uses OpenBLAS internally. Even with both inputs upcast to FP32 from FP16, BLAS's vectorized accumulation order ≠ C++ explicit 3-loop accumulation order, producing FP16-cast results that drift by up to 4 ULP / 0.0078 abs.

**Why it happens:** OpenBLAS uses tiled SIMD reductions (different summation order); C++ `gtx_npu_mm.cc:73-79` uses straightforward `for i / for j / for k: sum += a*b`. Both are correct in IEEE 754, but accumulation order matters for non-associative FP add.

**How to avoid:** P4 `gemm_core` MUST use explicit 3-loop. Reactivate `np.matmul` only after P6 strict-mode pass via 3-loop has banked the bit-exact baseline; P7 wraps either the 3-loop with `@njit` (faster than pure Python ~100×) OR validates that BLAS path matches C++ within `--ulp 1 --atol 0.001` for the specific firmware regression suite.

**Warning signs:**
- `verify.py --ulp 1 --atol 0.001 --strict` reports "PASS within tolerance" (`within_tolerance > 0`) but D-14 fails the test
- Single-cell ULP differences in 16×16 outputs scattered randomly
- 41/500 trials drifting (with current host: NumPy 2.2.6 + scipy-openblas)

### Pitfall B: `mxe_accum` per-(NEST,SPU) Continuity Drift Across Variants
**What goes wrong:** Plan author treats all 10 MM/MMC variants uniformly w.r.t. `mxe_accum`. In reality:
- MM_O / MMC_O (gtx_npu_mm.cc:200-212) **write** mxe_accum (scalar `sum(A)` or `sum(A) + prior`)
- MM_V / MMC_V (gtx_npu_mm.cc:267-269) **write** mxe_accum (scalar `dot(A,B)` or `dot(A,B) + prior`)
- MM_S / MMC_S (gtx_npu_mm.cc:150-176) DO NOT touch mxe_accum — write FP32 to ADDRC
- MM / MMC (gtx_npu_mm.cc:106-140) DO NOT touch mxe_accum — read FP32 bias from ADDRC, write FP16 to ADDRR
- MM_T / MMC_T (gtx_npu_mm.cc:289-315) DO NOT touch mxe_accum — write FP16 (transposed) to ADDRR

**Why it happens:** CONTEXT.md surface sketch in mm_engine `firmware_mm` writes mxe_accum on every `is_accumulate=True` call. This is wrong for MMC_S, MMC, MMC_T — those use ADDRC for prior-FP32-bias and ADDRR for write.

**How to avoid:** Variant-specific dispatch in `mm_engine.firmware_mm`. Surface sketch:
```python
def firmware_mm(npu, proc, insn, *, is_accumulate: bool, variant: str) -> int:
    rs1 = proc.get_state().XPR[insn.rs1]
    args = decode_firmware_mm_args(rs1)
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu  = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM: nest = 0
    if spu >= GTX_SPU_NUM:   spu = 0

    # exec_mm_o / exec_mm_v read+write npu._mxe_accum scalar
    # exec_mm_s / exec_mm / exec_mm_t use ADDRC FP32 bias staging via mem
    if variant in ('mm_o', 'mmc_o'):
        return _exec_mm_o_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_v', 'mmc_v'):
        return _exec_mm_v_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_s', 'mmc_s'):
        return _exec_mm_s_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm', 'mmc'):
        return _exec_mm_basic_variant(npu, nest, spu, args, is_accumulate)
    if variant in ('mm_t', 'mmc_t'):
        return _exec_mm_t_variant(npu, nest, spu, args, is_accumulate)
    return 0
```
**P4 chain test (D-07) must use the `mm.s → mmc.s → mmc` triad as ROADMAP states** — these all use ADDRC FP32 bias staging, NOT mxe_accum. The chain is via L1 ADDRC bytes. mxe_accum chain is a separate test (`mm.o → mmc.o`). **CONTEXT.md D-07 conflates the two — research surfaces this distinction.**

### Pitfall C: HW Convention "0 → 65536" Per-Field, Not Global
**What goes wrong:** `firmware_mm_op` rs1 has THREE 16-bit fields (rowA / colA / colB). Per gtx_npu_mm.cc:347 `dim16` lambda, the 0→65536 conversion is **per-field independent**. A naive `if rs1 == 0: rs1 = 0x10000` (whole-word) corrupts all three fields when only one is zero.

**Why it happens:** The CONTEXT.md surface sketch happens to be correct (`rowA = rs1 & 0xFFFF or 0x10000`), but a careless port might use a single test.

**How to avoid:** Apply `dim16(v) = (v & 0xFFFF) or 0x10000` to each field individually as in §rs1 Packed Decode example.

### Pitfall D: `mm_t` Output Layout Is N×M (Transposed), Not M×N
**What goes wrong:** `exec_mm_t` (gtx_npu_mm.cc:289-315) computes `C[i*N + j] = (A×B)[i,j]` but writes to `addr_r + (i + M*j)*2` — i.e., the output L1 region is **N×M** (transposed) layout, NOT M×N. A unit test that compares `actual.view(np.float16).reshape(M, N)` against `np.matmul(A,B)` will fail dimensionality. Compare instead against `np.matmul(A,B).T.reshape(N, M)` or a flat `.T.flatten()`.

**Why it happens:** The mnemonic `mm_t` doesn't say "A^T or B^T input" — it means "output is C^T".

**How to avoid:** Document in `mm_engine` docstring; pre-compute oracle as `np.matmul(A_f32, B_f32).T.astype(np.float16)`.

### Pitfall E: `funct3=4..6` Are Reserved (Default Falls Through to Basic mm)
**What goes wrong:** gtx_npu_mm.cc:373-376 `default:` clause routes any funct3 ∈ {4, 5, 6} to `exec_mm` with a TRACE warning. A pyspike port that crashes or returns illegal_instruction on these funct3 values diverges from C++.

**Why it happens:** The disasm table only registers funct3 ∈ {0, 1, 2, 3, 7}, but the dispatcher silently widens to all unregistered.

**How to avoid:** `mm_engine.firmware_mm` `default` route to `_exec_mm_basic_variant` (i.e., variant='mm' or 'mmc' depending on is_accumulate), matching C++ default. Spike trace will lack mnemonic for funct3=4..6 but op semantics match.

### Pitfall F: `proc.get_state().XPR[insn.rs1]` Reads Garbage if rs1==0 Path Mistakenly Reaches firmware_mm_op
**What goes wrong:** funct7=0x00 + insn.rs1==0 (i.e., x0) is the WRSPR gem5 path. If `ops/mm.py` accidentally is reached for rs1==0 (e.g., dispatch table priority bug), `XPR[0]` is hardwired to 0 — `dim16(0) = 0x10000`, leading to row_A=col_A=col_B=65536 = bogus 65536×65536×65536 GEMM.

**Why it happens:** The 2-level dispatch in `npu.custom0` (npu.py:125-144) tries the `None` sentinel key first before integer funct3. If P2 `ops/spr.py` registers WRSPR under `funct7=0x00, mask_funct3=False` (sentinel `None`), it WINS over P4 `funct7=0x00, mask_funct3=True, funct3=*` registrations, regardless of rs1. **But** P2 already dispatched WRSPR via funct7=0x49 (ISS_F7_WRSPR_ISS), not 0x00 — there is NO funct7=0x00 handler currently registered. P4 is FREE to register funct7=0x00 with mask_funct3=True.

**Mitigation (research finding):** Verify that P4 mm @handler with `mask_funct3=True` does NOT collide with any existing handler. Reading `src/main/python/riscv/gtx/ops/spr.py` (P2 plan-02) confirms WRSPR is registered at funct7=0x49 ISS-only, NOT at funct7=0x00. Path is clear. **However, the funct7=0x00 + insn.rs1==0 case (gem5-simplified WRSPR) is currently unhandled in pyspike** — no @handler exists for it. Plan should:
- Either add a `funct7=0x00, mask_funct3=False` @handler entry that NOPs (keeps backwards compat, since gem5 firmware emits WRSPR via funct7=0x49 in pyspike) — **OR**
- Implement the rs1!=0 heuristic INSIDE one of the funct3-specific @handler entries via early return when `insn.rs1 == 0`.

The former is cleaner; the latter mirrors C++ closely but spreads the heuristic across all 5 mm + 5 mmc handlers.

**Recommendation:** Add `funct7=0x00 + mask_funct3=False NOP` @handler in P4 plan (single entry, with explicit comment "gem5-simplified WRSPR path — handled by funct7=0x49 in pyspike; this is a no-op safety net per Pitfall 5").

### Pitfall G: `tmu_id` / `curr_id` Without `is_ploop` / `is_tloop` Guard
**What goes wrong:** gtx_npu_mm.cc:338-339:
```cpp
int nest = is_ploop ? tmu_id : 0;
int spu  = is_tloop ? curr_id : 0;
```
A direct port that drops the conditional (`nest = npu.warp.tmu_id`) corrupts NEST=0, SPU=0 outputs when run outside a P-loop or T-loop context.

**How to avoid:** Mirror exactly:
```python
nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
spu  = npu.warp.curr_id if npu.warp.is_tloop else 0
```

## Code Examples

### `decode_firmware_mm_args` (verified against gtx_npu_mm.cc:347-355)
```python
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:347
# auto dim16 = [](uint64_t v) -> uint32_t {
#     uint32_t d = v & 0xFFFF;
#     return d ? d : 0x10000;
# };
# uint32_t row_A = dim16(rs1);
# uint32_t col_A = dim16(rs1 >> 16);
# uint32_t col_B = dim16(rs1 >> 48);

def decode_firmware_mm_args(rs1: int) -> dict:
    def dim16(v: int) -> int:
        d = v & 0xFFFF
        return d if d != 0 else 0x10000
    return {
        'row_A': dim16(rs1),
        'col_A': dim16(rs1 >> 16),
        'col_B': dim16(rs1 >> 48),
    }
```

### `_exec_mm_o_variant` (verified against gtx_npu_mm.cc:186-225)
```python
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:186-225
# - Read addr_a from spu.lspr[LSPR_SPM_ADDRA]
# - sum = sum_k(fp16_to_32(spu.l1[(addr_a + k*2) % L1_SIZE | (spu.l1[..+1]<<8)]))
# - if has_bias: sum += mxe_accum[nest][spu]
# - mxe_accum[nest][spu] = sum
# - Write fp16(sum) to L0 (BIG-endian per L0 convention!) at l0_addr*32, zero rest

def _exec_mm_o_variant(npu, nest: int, spu: int, args: dict,
                       is_accumulate: bool) -> int:
    # B is implicit all-ones (gtx_npu_mm.cc:181). Only col_A elements of A read.
    col_A = args['col_A']
    addr_a = npu.lspr[nest][spu][LSPR_SPM_ADDRA]
    l1_bytes = npu.mem.l1_byte(nest, spu)

    # Read col_A FP16 values from L1 starting at addr_a
    sum_f32 = np.float32(0.0)
    for k in range(col_A):
        off = (addr_a + k * 2) % GTX_L1_SIZE_BYTES
        raw = int(l1_bytes[off]) | (int(l1_bytes[off + 1]) << 8)
        sum_f32 += np.float32(np.frombuffer(np.uint16(raw).tobytes(), dtype=np.float16)[0])

    if is_accumulate:
        sum_f32 += np.float32(npu._mxe_accum[nest, spu])
    npu._mxe_accum[nest, spu] = sum_f32

    # L0 write — BIG-ENDIAN per gtx_npu_mm.cc:217-218 (note: differs from L1 LE!)
    l0_addr = int(npu.gspr.get(GSPR_GTX_OPERAND3, 0)) & 0x1F
    l0_off = l0_addr * 32
    fp16 = int(np.float16(sum_f32).view(np.uint16))
    l0 = npu.mem.l0_byte(nest, spu)
    l0[l0_off % GTX_L0_SIZE_BYTES]       = (fp16 >> 8) & 0xFF
    l0[(l0_off + 1) % GTX_L0_SIZE_BYTES] = fp16 & 0xFF
    # Zero remaining 15 FP16 slots (32B - 2B written)
    for i in range(1, 16):
        l0[(l0_off + i * 2)     % GTX_L0_SIZE_BYTES] = 0
        l0[(l0_off + i * 2 + 1) % GTX_L0_SIZE_BYTES] = 0
    return 0
```

**⚠️ Important:** L0 write in MM_O is **big-endian** (gtx_npu_mm.cc:217-218 swaps lo/hi vs L1 LE). MM_V (line 274-275) is **little-endian** matching L1. This asymmetry is verified in source. The plan must reproduce this exactly.

### `_verify_minimal.compare_hex` (verified against verify.py:217-284)
```python
# Source: vendor/gtx_cpp_reference/gtx/verify.py:217-284
# - Parse hex via bytes.fromhex(line) per stripped line
# - r_raw = (data[i*2] << 8) | data[i*2+1]   <- BIG-ENDIAN bit-pair (Pitfall 1!)
# - exact = sum(r_raw == g_raw)
# - within_tol if ULP <= ulp_tol OR abs_diff <= atol
# - PASS = mismatches == 0 (i.e., everything is at least within_tolerance)
# - STRICT mode (D-14): PASS = exact_matches == total_fp16

def compare_hex(actual_path: str, golden_path: str, *,
                ulp: int = 1, atol: float = 0.001, strict: bool = True) -> tuple:
    def _parse(path):
        out = bytearray()
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                # Note: dump output never has @offset, but be tolerant
                if line.startswith('@'):
                    continue
                out.extend(bytes.fromhex(line))
        return bytes(out)

    a_bytes, g_bytes = _parse(actual_path), _parse(golden_path)
    n = min(len(a_bytes), len(g_bytes)) // 2
    exact = within = failures = 0
    first_failure = None

    for i in range(n):
        # BIG-ENDIAN bit-pair per verify.py:235 (Pitfall 1)
        r_raw = (a_bytes[i*2] << 8) | a_bytes[i*2 + 1]
        g_raw = (g_bytes[i*2] << 8) | g_bytes[i*2 + 1]
        if r_raw == g_raw:
            exact += 1
            continue
        # Decode and compare ULP / abs
        r_val = float(np.frombuffer(np.uint16(r_raw).newbyteorder('<').tobytes(),
                                    dtype=np.float16)[0])
        g_val = float(np.frombuffer(np.uint16(g_raw).newbyteorder('<').tobytes(),
                                    dtype=np.float16)[0])
        ulp_dist = abs((r_raw if r_raw < 0x8000 else -(r_raw & 0x7FFF))
                       - (g_raw if g_raw < 0x8000 else -(g_raw & 0x7FFF)))
        abs_diff = abs(r_val - g_val) if not (np.isnan(r_val) or np.isnan(g_val)) else float('inf')
        if ulp_dist <= ulp or abs_diff <= atol:
            within += 1
        else:
            failures += 1
            if first_failure is None:
                first_failure = (i, r_raw, g_raw)

    stats = dict(exact_matches=exact, within_tolerance=within,
                 failures=failures, total_fp16=n, first_failure=first_failure)
    if strict:
        return (exact == n, stats)
    return (failures == 0, stats)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `np.matmul(A_f32, B_f32).astype(np.float16)` (CONTEXT D-02) | Explicit Python 3-loop accumulate (this RESEARCH) | 2026-05-06 (P4 plan) | P4 must trade ~100× speed for bit-exactness; P7 numba reactivates BLAS-equivalent perf via @njit |
| `nest, spu = npu.tmu_id, npu.curr_id` (CONTEXT surface sketch) | `nest, spu = npu.warp.tmu_id, npu.warp.curr_id` (verified vs warp_state.py:29-30) | 2026-05-06 | Trivial fix; CONTEXT was wrong about field path |
| `funct3=4 → mm_t` (CONTEXT specifics) | **`funct3=7 → mm_t`** (verified vs gtx_npu_disasm.inc:43, gtx_npu_mm.cc:370) | 2026-05-06 | Critical fix — wrong funct3 value would have made mm_t/mmc_t handlers unreachable by firmware |
| Use vendor `mm_basic.elf` + `mm_basic_n1s16.hex` (CONTEXT D-09/D-10) | Write own .S/.elf and synthesize golden | 2026-05-06 (vendor lacks mm_basic assets) | P4 absorbs ~1 day of fixture work that was assumed to be borrowing |

**Deprecated/outdated:**
- CONTEXT.md D-04 says "10 individual @handler" — still correct but funct3=4 (not 7) was wrong.
- CONTEXT.md gemm_core surface sketch using `np.matmul` — still relevant for P7 but misleading for P4 strict mode.
- ROADMAP.md `Phase 4 Plans: 5` placeholder — still valid plan count.

## Runtime State Inventory

> Phase 4 is feature-add (no rename/refactor). Skipping per template guidance. **Stored data:** None — verified by `git diff src/main/python/riscv/gtx/`. **Live service config:** None. **OS-registered state:** None. **Secrets/env vars:** Existing `GTX_NO_EXIT`, `GTX_DDR_REVERSED`, `GTX_DDR_SIZE` (all P1-P3 owned, unchanged). New `GTX_DDR_DUMP_ADDR` / `GTX_DDR_DUMP_SIZE` consumed in tests only. **Build artifacts:** New `tests/gtx/data/elf/mm_basic.elf` (committed). No package/wheel impact in P4 (PKG-01 is P6).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All P4 work | ✓ | 3.x available on host | — |
| NumPy ≥ 2.0 | gemm_core, mm_engine | ✓ | 2.2.6 | — |
| pytest | tests/gtx/test_*.py | ✓ | (existing) | — |
| `riscv64-unknown-elf-gcc` (15.2+) | mm_basic.elf build | ✗ on this researcher's box, ✓ on dev box per P2 D-22 | — | **Commit pre-built mm_basic.elf to git (P2 D-22 pattern). CI does NOT need toolchain.** |
| `pyspike` CLI on PATH | regression test subprocess invocation | ✓ when wheel installed | — | `[sys.executable, "-m", "riscv"]` per P2 test_skeleton.py:53-58 |
| `_riscv.so` (pyspike binding) | end-to-end .elf regression | ✓ when wheel built | — | `_RISCV_AVAILABLE` self-detect skip pattern (P2 plan-05 D-1) |
| C++ libgtx_npu.so (vendor) | ground-truth golden hex generation | ⚠ optional (developer env only) | — | **In-Python explicit 3-loop synthesizes golden directly — no C++ build dep at CI time** |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- riscv64-unknown-elf-gcc — pre-build .elf and commit (mirrors P2 D-22)
- C++ libgtx_npu.so — synthesize golden hex via in-Python 3-loop (self-comparison test layer; OK for P4 unit + integration; cross-validation with C++ is a P6 follow-up if/when libgtx_npu.so build env stabilized)

## `np.matmul` Bit-Exactness Analysis (NEW — empirically driven)

### Test setup
- Host: Linux x86_64 / NumPy 2.2.6 / scipy-openblas BLAS backend
- 500 random 16×16×16 GEMM trials with `seed(42)`
- Inputs: `randn(16,16).astype(np.float16)`, upcast to FP32, then both `np.matmul` (BLAS) and explicit triple-`for` loop (FP32 accum)
- Outputs cast to FP16 and compared by `view(np.uint16)` for raw bit-pattern equality

### Findings
| Metric | Value | Implication |
|--------|-------|-------------|
| Bit-exact trials | 459/500 (91.8%) | Majority match — BLAS ~RIGHT for most matrices |
| Drifting trials | 41/500 (8.2%) | Substantial minority drift |
| Max ULP drift | **4** | Exceeds `verify.py --ulp 1` |
| Max abs diff | **0.0078125** | Exceeds `verify.py --atol 0.001` |

### Implication for P4
- **`np.matmul` (BLAS) cannot pass `--strict` mode** against C++ explicit-3-loop golden hex (gtx_npu_mm.cc:73-79 default path).
- D-14 strict mode (`exact_matches == total_fp16`) would fail on ~8% of random matrices.
- P7 numba `@njit` of explicit 3-loop is the bit-exact path; P7 BLAS reactivation requires golden recomputation OR loosened acceptance gate (regression-specific decision).

### Recommendation locked
Plan P4 with **explicit triple-loop FP32 accumulate** in `gemm_core`. The CONTEXT.md D-02 surface sketch becomes:
```python
# WRONG for P4 strict mode:
# C_f32 = np.matmul(A_f32, B_f32)

# CORRECT for P4 strict mode (port of gtx_npu_mm.cc:73-79):
C_f32 = np.zeros((M, N), dtype=np.float32)
for i in range(M):
    for j in range(N):
        s = np.float32(0.0)
        for k in range(K):
            s += A_f32[i, k] * B_f32[k, j]
        C_f32[i, j] = s
```
Performance estimate: 16×16×16 GEMM in pure-Python ~5 ms (CPython 3.10), vs `np.matmul` ~50 µs. Acceptable for P4 unit tests + a small mm_basic.elf regression. A 256×256×256 GEMM at 5 ms × (256/16)³ = 21 sec — also acceptable for "tens of minutes" budget. P7 numba `@njit` would bring this back under 1 ms.

## ELF Fixture Strategy (D-09 fallback locked)

**Vendor inventory verified:**
- `vendor/gtx_cpp_reference/example/build/{add_vv_example.elf, test_printf.elf}` — neither emits `mm` opcodes
- `vendor/gtx_cpp_reference/test/MUL_MAT/n1s16/` — only `.c` source + plain-text input/ref data, no .elf, no .hex
- `vendor/gtx_cpp_reference/test/CONCAT/n1s16/data/n1s16_concat_result2.hex` — only golden hex in vendor, but for CONCAT op (P5)
- **No `mm_basic.elf` and no `mm_basic_n1s16.hex` exist anywhere in vendor.**

**Fallback (D-09 fallback path activated):**

1. Author `tests/gtx/data/elf/mm_basic.S` mirroring `nop_wjoin.S`:
```asm
# Minimal MM firmware for P4 strict mode regression.
#
# Behavior:
#   1. addi sp, sp, -16            ; valid DRAM
#   2. WRSPR LSPR_SPM_ADDRA, 0     ; A region at L1[0..]
#   3. WRSPR LSPR_SPM_ADDRB, 0x200 ; B region at L1[0x200..]
#   4. WRSPR LSPR_SPM_ADDRR, 0x400 ; result at L1[0x400..]
#   5. (load A, B into L1 via DMA OR pre-fill via init hex)
#   6. .insn r 0x0b, 0b010, 0x00, x10, x1, x2  ; custom0 funct7=0x00 funct3=2 (MM)
#                                              ; rs1 = packed dims, rs2 unused
#                                              ; rd = x10 (MM returns 0)
#   7. .insn r 0x2b, 0b101, 0, x0, x0, x0      ; WJOIN
#
# rs1 packed: colB << 48 | colA << 16 | rowA
# For 4×4×4: rs1 = (4 << 48) | (4 << 16) | 4 = 0x0004_0000_0004_0004

.section .text._start, "ax"
.global _start
_start:
    addi  sp, sp, -16

    # Set up L1 ADDR* via WRSPR (funct7=0x49 ISS-full)
    li    x1, 0x900
    li    x2, 0
    .insn r 0x0b, 0, 0x49, x0, x1, x2     # WRSPR ADDRA = 0
    li    x1, 0x901
    li    x2, 0x200
    .insn r 0x0b, 0, 0x49, x0, x1, x2     # WRSPR ADDRB = 0x200
    li    x1, 0x903
    li    x2, 0x400
    .insn r 0x0b, 0, 0x49, x0, x1, x2     # WRSPR ADDRR = 0x400

    # Pre-load A, B at L1[0], L1[0x200] via DMA-load — option A
    # OR test fixture pre-fills via ddr_init_from_file at the host setup level — option B (simpler)

    # Issue MM (funct7=0x00, funct3=2 = mm), rs1=x1, rs2=x0
    li    x1, 0x0004000000040004           # 4×4×4 GEMM packed dims
    .insn r 0x0b, 0b010, 0x00, x10, x1, x0

    # WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0

    j .
```

2. Author `tests/gtx/data/elf/Makefile` mirroring nop_wjoin/Makefile:
```make
CC       = /opt/riscv/bin/riscv64-unknown-elf-gcc
LDFLAGS  = -nostdlib -nostartfiles -static -Wl,-Ttext-segment=0x80000000

mm_basic.elf: mm_basic.S
	$(CC) $(LDFLAGS) -o $@ $<

# Existing nop_wjoin.elf rule unchanged ...
```

3. Commit `mm_basic.elf` to git (P2 D-22 pattern); CI does NOT need toolchain.

4. **Synthesize golden hex** by:
   - Pre-fill A, B in L1 at known offsets (via fixture: `mem.l1_f16(0,0)[0:16] = A_pattern; mem.l1_f16(0,0)[0x100:0x110] = B_pattern`)
   - Run mm_basic.elf via subprocess pyspike with `GTX_DDR_DUMP=/tmp/actual.hex`
   - Compute oracle in Python: `oracle_C = explicit_3loop_fp32(A_pattern, B_pattern).astype(np.float16)`
   - Pack oracle FP16 values as **big-endian bit pairs** (per verify.py:235 convention) into hex lines, write to `tests/gtx/data/golden/mm_basic_n1s16.hex`
   - Compare actual.hex vs golden.hex via `_verify_minimal.compare_hex(strict=True)`

5. **Bonus: fully-in-process unit test** (no subprocess) covers MM-01..04 without .elf:
   - `test_op_mm.py` constructs `MockProcessor`, `MockInsn(funct7=0x00, funct3=2, rs1=1, rs2=0)`, sets `XPR[1] = packed_dims`, pre-fills `npu.mem.l1_f16(0,0)`, calls `npu.custom0(...)`, reads `npu.mem.l1_f16(0,0)[result_off:]`.

This bridges the "cross-toolchain not always available" + "vendor missing assets" gap with no architectural change to the broader plan.

## Open Questions

1. **Does the C++ libgtx_npu.so actually use scalar 3-loop (default `#else` path) or OpenBLAS in production?**
   - What we know: `gtx_npu_mm.cc:65-67` gates OpenBLAS via `#if defined(GTX_USE_OPENBLAS)`. Default build flag set unknown.
   - What's unclear: If golden hex was generated under `GTX_USE_BLAS=1`, our explicit 3-loop differs from golden. If golden was generated under default (`#else` 3-loop), our 3-loop matches.
   - Recommendation: Plan a sanity round-trip — record what build flags vendor used. Or, since we're synthesizing golden in Python ourselves (per ELF Fixture Strategy step 4), this becomes moot for P4. **For P6 cross-validation against vendor goldens, this question reactivates.**

2. **Is `mm_basic.elf`'s correct firmware sequence to use ISS-full WRSPR (funct7=0x49) or gem5-simplified (funct7=0x00, rs1==0)?**
   - What we know: P2 only registered ISS-full WRSPR (funct7=0x49). gem5-simplified WRSPR via funct7=0x00 + rs1==0 is unhandled by current pyspike (will hit P4 mm @handler with mask_funct3=True and silently NOP per Pitfall F).
   - Recommendation: `mm_basic.S` SHOULD use ISS-full WRSPR (funct7=0x49) for unambiguous routing. The Pitfall F safety NOP @handler at funct7=0x00, mask_funct3=False is for backward compat only.

3. **Will pyspike `subprocess.run` propagate WJOIN `SystemExit(0)` cleanly under all CI/wheel install configurations?**
   - What we know: P2 plan-05 verified working on dev machine. P3 regression suite passes 179/179.
   - What's unclear: cibuildwheel manylinux2014 environment may have different exec semantics.
   - Recommendation: Plan covers both paths (subprocess primary; in-process `sim_t.run` documented as v2 follow-up if subprocess proves flaky in CI matrix).

4. **Does any vendor MUL_MAT firmware exist that could be ported to .S?**
   - What we know: `vendor/.../test/MUL_MAT/n1s16/n1s16_mul_mat.c` is a C source not committed as .elf. Vendor uses a `run_tests_n1s16.sh` script we have NOT inspected.
   - Recommendation: P4 plan should check for cross-compiled .elf artifacts that may be in CI/dev-box `vendor/.../test/MUL_MAT/n1s16/build/` (not in git). Otherwise ship our own minimal mm_basic.elf as primary fixture.

## Validation Architecture

> nyquist_validation = true (per .planning/config.json)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥ 8 (existing) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` (existing); per-test isolation via `--noconftest -o "addopts="` for offline mode (P2 plan-05 D-1) |
| Quick run command | `pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py tests/gtx/test_funct7_routing.py -x --noconftest -o "addopts="` |
| Full suite command | `pytest tests/gtx/ -q` (includes regression test_regression_fw_mm.py — gated on `_RISCV_AVAILABLE` + `mm_basic.elf` + `pyspike` on PATH) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MM-01 | `gemm_core` FP32 internal accumulate, FP16 cast | unit | `pytest tests/gtx/test_op_mm.py::test_gemm_core_explicit_3loop_matches_oracle -x` | ❌ Wave 0 |
| MM-01 | `gemm_core` Pitfall 2 anti-pattern (`np.float16` reduction inf-trap) | unit | `pytest tests/gtx/test_op_mm.py::test_gemm_core_fp32_internal_not_fp16 -x` | ❌ Wave 0 |
| MM-02 | All 10 MM/MMC variants @handler register correctly + funct3 mapping | unit | `pytest tests/gtx/test_op_mm.py::test_handler_registry_has_all_10_mm_variants -x` | ❌ Wave 0 |
| MM-02 | mm/mmc basic 16×16×16 bit-exact vs explicit 3-loop oracle | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_basic_bit_exact -x` | ❌ Wave 0 |
| MM-02 | mm_s/mmc_s FP32 to ADDRC bit-exact | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_s_writes_fp32_to_addrc -x` | ❌ Wave 0 |
| MM-02 | mm_o/mmc_o scalar sum(A) to L0 + mxe_accum (BIG-endian L0 layout!) | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_o_writes_scalar_to_l0_be -x` | ❌ Wave 0 |
| MM-02 | mm_v/mmc_v scalar dot(A,B) to L0 (LE) + mxe_accum | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_v_writes_dot_to_l0_le -x` | ❌ Wave 0 |
| MM-02 | mm_t/mmc_t transposed C^T to ADDRR (N×M layout!) | unit | `pytest tests/gtx/test_op_mm.py::test_exec_mm_t_writes_transposed -x` | ❌ Wave 0 |
| MM-03 | `decode_firmware_mm_args` rs1 packed bits + 0=65536 per field | unit | `pytest tests/gtx/test_op_mm.py::test_decode_firmware_mm_args -x` | ❌ Wave 0 |
| MM-03 | funct7=0x00 + rs1==0 → no MM; rs1!=0 → MM | unit | `pytest tests/gtx/test_funct7_routing.py::test_funct7_zero_collision_routing -x` | ❌ Wave 0 |
| MM-03 | funct7=0x01 always routes to MMC regardless of rs1 | unit | `pytest tests/gtx/test_funct7_routing.py::test_funct7_one_always_mmc -x` | ❌ Wave 0 |
| MM-04 | mm.s → mmc.s → mmc chain in L1 ADDRC accumulates correctly | integration | `pytest tests/gtx/test_mm_chain.py::test_mm_addrc_chain_continuity -x` | ❌ Wave 0 |
| MM-04 | mm.o → mmc.o chain on `mxe_accum[(nest=1, spu=5)]` accumulates | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_chain_continuity -x` | ❌ Wave 0 |
| MM-04 | per-(NEST,SPU) isolation — only [1,5] cell mutates, others unchanged | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_per_cell_isolation -x` | ❌ Wave 0 |
| MM-04 | mxe_accum dtype stays float32 across chain | integration | `pytest tests/gtx/test_mm_chain.py::test_mxe_accum_dtype_locked -x` | ❌ Wave 0 |
| MM-05 | First .elf GEMM regression strict mode pass | regression | `pytest tests/gtx/test_regression_fw_mm.py::test_mm_basic_strict_mode_pass -x` | ❌ Wave 0 (gated on `_RISCV_AVAILABLE` + ELF + pyspike CLI) |
| MM-05 | `_verify_minimal.compare_hex` BE FP16 bit-pair (Pitfall 1) | unit | `pytest tests/gtx/test_op_mm.py::test_verify_minimal_be_fp16_pairs -x` | ❌ Wave 0 |
| MM-05 (#5) | Mode 4 P+T dispatch routes to (tmu_id, curr_id) only | unit | `pytest tests/gtx/test_funct7_routing.py::test_mode4_routes_to_tmu_curr -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/gtx/test_op_mm.py tests/gtx/test_mm_chain.py tests/gtx/test_funct7_routing.py -x --noconftest -o "addopts="` (≤ 30 sec; pure-python; no `_RISCV_AVAILABLE` requirement)
- **Per wave merge:** `pytest tests/gtx/ -q` (full P3 + P4 regression, ~1-2 min)
- **Phase gate:** Full suite green INCLUDING `test_regression_fw_mm.py::test_mm_basic_strict_mode_pass` before `/gsd:verify-work 4`. If `_RISCV_AVAILABLE` is False or the toolchain-built `mm_basic.elf` is missing, the regression test must skip cleanly (NOT fail) — but the ship gate is "skip-or-pass; never fail".

### Wave 0 Gaps
- [ ] `tests/gtx/test_op_mm.py` — covers MM-01, MM-02, MM-03 (decode), MM-05 (verify_minimal unit)
- [ ] `tests/gtx/test_mm_chain.py` — covers MM-04
- [ ] `tests/gtx/test_funct7_routing.py` — covers MM-03 (routing matrix), MM-05 (Mode 4)
- [ ] `tests/gtx/test_regression_fw_mm.py` — covers MM-05 strict-mode .elf
- [ ] `tests/gtx/_verify_minimal.py` — D-13 mini port
- [ ] `tests/gtx/data/elf/{mm_basic.S, mm_basic.elf}` — D-09 fallback fixture (ELF committed)
- [ ] `tests/gtx/data/elf/Makefile` — extend existing or add mm_basic.elf rule
- [ ] `tests/gtx/data/golden/mm_basic_n1s16.hex` — D-10 fallback synthesized golden (FP16 BE bit-pair format per verify.py:235)
- [ ] `src/main/python/riscv/gtx/gemm_core.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/mm_engine.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/ops/mm.py` — module exists check (will fail until Wave 1)
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — `from . import mm` line check
- [ ] Test framework: existing pytest infra is sufficient; no install needed.

## Sources

### Primary (HIGH confidence)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc:1-389` — gemm_core (27-94), exec_mm (106-140), exec_mm_s (150-176), exec_mm_o (186-225), exec_mm_v (233-281), exec_mm_t (289-315), firmware_mm_op (333-389)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:38-50` — exact mnemonic ↔ funct7+funct3 mapping for all 10 MM/MMC variants
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:56-72` — funct7=0x00/0x01 collision heuristic (`if insn.rs1 != 0 → firmware_mm_op`)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:151-275` — dispatch_iss_opcode body (separate from firmware_mm_op path)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:412-416` — GTX_MM_BASIC=0..GTX_MM_T=4 constants (used by gem5-simplified DISPATCH_MM only, NOT by funct3 in firmware_mm_op)
- `vendor/gtx_cpp_reference/gtx/verify.py:217-284` — compare_fp16 BE bit-pair logic, ULP/atol semantics
- `src/main/python/riscv/gtx/warp_state.py:25-42` — `tmu_id` / `curr_id` exact field names
- `src/main/python/riscv/gtx/npu.py:33-178` — GtxNpu structure, mxe_accum init, custom0 2-level dispatch, flush_deferred_ddr_stores
- `src/main/python/riscv/gtx/dispatch_4mode.py:38-121` — dispatch_iss_opcode P3 stub + dispatch_4mode 4-mode router
- `src/main/python/riscv/gtx/_registry.py:27-99` — @handler decorator + 2-level collect_for_kind('custom0')
- `src/main/python/riscv/gtx/dma_engine.py:66-99` — `decode_firmware_dma_args` mirror pattern for MM
- `src/main/python/riscv/gtx/ops/dma.py:1-100` — @handler + `_select_nest`/`_select_spu` pattern for P4 to mirror
- `src/main/python/riscv/gtx/encoding.py` — funct7/funct3 + GSPR/LSPR address constants (no MM constants yet — P4 should add `GTX_F7_MM=0x00`, `GTX_F7_MMC=0x01`, `GTX_MM_S/O/V/T=0/1/2/3/7` funct3 names; or use literals)
- `src/main/python/riscv/gtx/memory.py:36-86` — GtxMemory l0/l1/l2 byte+f16 view API
- `src/main/python/riscv/gtx/ddr.py:97-169` — ddr_init_from_file / ddr_dump_to_file (consumed by D-12 explicit dump in regression test)
- `tests/gtx/test_skeleton.py:30-176` — subprocess pyspike pattern + `_RISCV_AVAILABLE` self-detect (P4 regression mirrors)
- `tests/gtx/data/elf/{nop_wjoin.S, Makefile}` — P2 D-22 fixture pattern (P4 mm_basic mirrors)
- `.planning/STATE.md` "Phase 2 Plan 01 Decisions" — `mxe_accum` is 2D (NEST, SPU) float32 (locked correction from CONTEXT D-06 4D)
- `.planning/research/PITFALLS.md` Pitfalls 1, 2, 3, 4, 5 — direct relevance to P4

### Secondary (MEDIUM confidence)
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` (system-reminder) — confirms FP16 LE byte order on L1, BE/LE asymmetry implied by gtx_npu_mm.cc:217-218 (MM_O writes BE to L0 — research surfaces)
- `vendor/gtx_cpp_reference/example/build/{add_vv_example.elf, test_printf.elf}` — confirms mm_basic.elf NOT in vendor (vendor inventory)
- `vendor/gtx_cpp_reference/test/MUL_MAT/n1s16/{n1s16_mul_mat.c, data/}` — C source + plain-text data, NO .elf (vendor inventory)

### Tertiary (LOW confidence — empirical only, single-host measurement)
- `np.matmul` vs explicit 3-loop bit-exact across 500 trials: 459/500 match, 41/500 drift up to 4 ULP. Single-host measurement (NumPy 2.2.6 + scipy-openblas). **Recommendation: cross-verify on CI manylinux2014 environment as a P4 plan-validation task. Different BLAS backends may show different drift profiles.**

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already on disk; numpy 2.2.6 verified
- Architecture: HIGH — direct port from C++ ground-truth; existing P2/P3 patterns reused
- Pitfalls: HIGH — 5 of 6 surfaced pitfalls are source-verified; pitfall A (BLAS drift) empirically measured on this host
- ELF/golden fixture strategy: HIGH for fallback path (D-09 fallback explicitly documented in CONTEXT.md); MEDIUM for primary "vendor borrow" path (vendor lacks asset — fallback activated)
- In-process .elf load: LOW — feasible per pyspike binding capability but NOT validated; subprocess fallback is HIGH confidence path

**Research date:** 2026-05-06
**Valid until:** 2026-06-05 (30 days; stable C++ vendor, stable pyspike binding API). Earlier expiry if pyspike upstream spike bumps `rocc_insn_t` layout (Pitfall 15).

## RESEARCH COMPLETE
