# Phase 5: VEC/ACT/Pool - Research

**Researched:** 2026-05-07
**Domain:** GTX NPU vector / activation / pooling / format-conversion subsystem — bit-exact FP16 with C++ libgtx_npu.so under ULP-1 strict mode. Specifically: SASMD/DOT/VSUM/CLAMP, 7 ISS activations + 4 L0 immediates, max/avg pool, FP16↔FP8/INT8/INT32/FP32/FP64 cvt with scale+offset, and the 32-op `verify_ref.py` oracle suite as pytest.
**Confidence:** **HIGH** for op-level encodings and FP discipline (every claim sourced from a vendor file:line that is on disk under `vendor/gtx_cpp_reference/gtx/`); **HIGH** for activation direction asymmetry (it is two consecutive `if`/`else` branches at `gtx_npu_act.cc:37-42`); **MEDIUM** for VSUM row-split mode-B golden source (no vendor `.elf` fixture exists for VSUM — golden must be synthesized in-Python from the C++ algorithm); **HIGH** for `activation_relu_gelu.elf` strategy (no vendor asset → hand-write `.S` mirroring P4 `mm_basic.S`).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**모듈 구성 (D-01 ~ D-04)**
- **D-01:** **VEC 3-way module split** — `riscv/gtx/ops/vec.py` (@handler thin forwarders) + `riscv/gtx/vec_engine.py` (firmware_vec_op decode + variant dispatcher + L0/L1 path branch + memory view) + `riscv/gtx/vec_core.py` (pure stateless NumPy: `sasmd_kernel`, `dot_kernel`, `vsum_kernel`, `clamp_kernel`).
- **D-02:** **ACT bundled module** — `riscv/gtx/ops/act.py` (all activation+pool+format_cvt @handler entries) + `riscv/gtx/act_engine.py` (single engine: activation direction dispatch + pool kernel/stride decode + format_cvt type-pair decode + scale/offset unpack from `GSPR_GTX_OPERAND2`) + `riscv/gtx/act_core.py` (pure NumPy: activations + pool + format_cvt + FP8 LUT helpers).
- **D-03:** **VRF-02 oracle location = `tests/gtx/_oracles.py`** (test-only tier; wheel-excluded). Single file first; split to `tests/gtx/_oracles/{vec,act,format,pool}.py` if it crosses ~200 LOC at plan-stage.
- **D-04:** **Wave structure mirror P4** — Wave 1a scaffold (RED test stubs + `_oracles.py` skeletons + `activation_relu_gelu.elf` fixture); Wave 1b 4 sequential plans (vec_core, vec_engine+ops/vec, act_core activations, act_engine+ops/act); Wave 2 integration (oracle full GREEN + .elf strict-mode regression). 6 plans total.

**활성화 방향 비대칭 (D-05 ~ D-08)**
- **D-05:** **`is_reversed` parameter explicit at `ops/act.py` @handler entry** (P4 D-05 `is_accumulate` mirror). Reading list at @handler is the source of truth — no module-level `REVERSED_OPS` set indirection (D-06).
- **D-07:** ACT-05 `_imm` variants use the **same direction asymmetry**. 16 total activation @handler entries (8 ISS-path + 8 L0-immediate-path).
- **D-08:** `format_cvt` and `exec_pooling` carry no direction state (always forward ADDRA→ADDRR per `gtx_npu_act.cc:225-227, 199`). Their @handlers do not accept `is_reversed`.

**VSUM/DOT 정밀도 (D-09 ~ D-12)**
- **D-09:** **VSUM dual-mode** — kernel implements mode A only (`vec_core.vsum(view) -> np.float16`, FP32 internal accumulate + single FP16 cast). Mode B (firmware row-split + N partial sum re-accumulate) is **firmware-orchestrated** at the @handler caller level, not inside the kernel.
- **D-10:** **Two test families** — `test_vsum_fp32_internal_anti_pattern` (`[1.0, 1e-4]*1000` → ≈ 0.1, NOT inf) for kernel correctness + `test_vsum_row_split_matches_cpp` (parametrized over rows ∈ {2,4,8,16}) for mode-B firmware composition.
- **D-11:** **DOT precision = VSUM single-call equivalent.** Same FP32-internal-accumulate-with-single-FP16-cast discipline.
- **D-12:** FP32→FP16 cast on overflow = IEEE RNE saturating to inf. Use `np.float16(fp32_value)` directly (NumPy 2.x behavior). C++ `gtx_fp32_to_16` semantics verified bit-exact in P4 strict-mode .elf regression (P4 04-05 closed; baseline holds for P5).

**format_cvt + FP8 codec (D-13 ~ D-16)**
- **D-13:** `format_cvt` = 6 separate @handler entries (1 per direction): `_exec_scvt_qh` (FP16→FP8), `_exec_scvt_hq` (FP8→FP16), `_exec_scvt_ih` (FP16→INT8), `_exec_scvt_hi` (INT8→FP16), `_exec_scvt_hn` (INT32→FP16 normalize), `_exec_fcvt_sh` (FP32↔FP16). **CONTEXT.md missed FP64↔FP16 (`fcvt_dh`/`fcvt_hd` at funct7=0x25)** — research adds it as a 7th direction (see Decision Adjustments below).
- **D-14:** **FP8→FP16 codec = 256-byte LUT** precomputed at module load via `gtx_fp8_to_32` (vendor `gtx_npu.h:154`).
- **D-15:** **FP16→FP8 codec = 64KB LUT** precomputed at module load via `gtx_fp16_to_8` (vendor `gtx_npu.h:182`).
- **D-16:** Explicit FP8 codec test coverage: subnormal table, exp=0xF=inf (NOT NaN — divergence from NVIDIA E4M3), 256-input round-trip.

### Claude's Discretion (research-resolved below)

- 10-variant SASMD funct7 sub-encoding → **LOCKED** (§Standard Stack §SASMD encoding table)
- CLAMP funct7=0x18..0x1F mapping → **LOCKED** (`gtx_npu_disasm.inc:134-142`; all 8 variants live at funct7=0x1F not 0x18..0x1F as CONTEXT.md said — see Adjustment 2)
- `firmware_vec_op` packed-rs1 layout → **LOCKED** (`gtx_npu_vec.cc:572-580`; rs1 = `vec_size & 0xFFFF`, no multi-field packing)
- `_imm` immediate operand decode → **LOCKED** (`gtx_npu_act.cc:379-431` + `:439-487`; reg indices from rs1/rs2/op3 + FP16 param/max/accum from operand staging)
- VSUM row-split golden — **NO vendor `.elf`** for VSUM available → synthesize in-Python from `gtx_npu_vec.cc:102-112` algorithm (FP32-internal-then-FP16-cast in a Python loop) for golden hex
- `activation_relu_gelu.elf` — **NO vendor asset** (`vendor/gtx_cpp_reference/test/RELU/n1s16/` only has `.c` + `.txt` data) → hand-write `.S` mirroring P4 `mm_basic.S`
- `gtx_fp32_to_16` IEEE semantics vs `np.float16` — **HIGH-confidence equivalent** (P4 04-05 strict-mode .elf regression PASSED end-to-end with `np.float16`; baseline holds)
- pool kernel/stride packing → **LOCKED** (`gtx_npu_dispatch.cc:653-655, 673-675`: `length = op1 & 0xFFFF`, `kernel_size = op2 & 0xFFFF`)
- CLAMP `accum`/`arange` semantics → **LOCKED** (`gtx_npu_vec.cc:215-249`; accum = prefix sum, arange = `op2`-staged start+step)
- `exec_format_cvt SCVT_HN` (INT32→FP16 normalize) formula → **LOCKED** (`gtx_npu_act.cc:301-313`; `out = fp16(int32 * scale + offset)`)

### Deferred Ideas (OUT OF SCOPE)

- Production `riscv.gtx._verify` CLI + `--strict` — Phase 6 VRF-01
- `tests/gtx/data/{golden,elf}/` package_data — Phase 6 PKG-01
- `pyspike-verify` console script — Phase 6 PKG-03
- Full .elf regression matrix (gem5+ISS, all 103 GGML kernels) — Phase 6 VRF-04
- `GTX_DDR_DUMP` atexit auto-flush — Phase 6 (P4 deferred)
- Numba `@njit` on `vec_core.vsum`/`act_core.softmax` etc. — Phase 7
- CUDA / cuBLAS / OpenMP paths — PROJECT.md Out of Scope; vendor `GTX_USE_CUBLAS`/`GTX_ENABLE_OMP` branches in `gtx_npu_*.cc` are **deliberately ignored** in Python port (already noted in CONTEXT canonical_refs)
- DMA-3D / IM2COL / MCAST — v2 (P3 deferred)
- ScVT_HN INT32→FP16 normalize is in scope; **FP64↔FP16 (`fcvt_dh`/`fcvt_hd`)** is in scope per disasm table even though CONTEXT D-13 missed it (Adjustment 1)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **VEC-01** | SASMD (add/sub/mul/div) 4×{IS,VS} = 8 variants at funct7=0x10 | §SASMD encoding table (`gtx_npu_disasm.inc:67-74`); IS variants reduce/scalarise via L0; VS variants element-wise on L1. C++ `exec_vec_scalar` (`gtx_npu_vec.cc:283-342`) is the L1 VS path; `exec_scalar_imm` (`gtx_npu_vec.cc:352-402`) is the L0 IS path. |
| **VEC-02** | DOT / VSUM with FP32-internal-accumulate + single FP16 cast | §VSUM/DOT precision (`gtx_npu_vec.cc:102-112` + `:251-262`). Anti-pattern test value `np.float16([1.0, 1e-4]*1000).sum()` saturates to inf in pure-FP16 — kernel must keep FP32 accumulator. |
| **VEC-03** | CLAMP min/max/arange/accum, L1(VV)/L0(II) branches | §CLAMP encoding (`gtx_npu_disasm.inc:135-142`): all 4 CLAMP-family variants live at funct7=**0x1F** (NOT 0x18-0x1F as CONTEXT D-01 surface sketch said — Adjustment 2). funct3=0/1/2/3 = clamp_min/clamp_max/accum/arange; funct3=4..7 select bitwise (and_ii/or_ii/not_i/shift_i) at the same funct7. |
| **VEC-04** | VEC scalar / immediate variants (`exec_vec_scalar`, `_imm`) | §`firmware_vec_op` L0/L1 branch (`gtx_npu_vec.cc:572-754`): `(funct3 & 4)` bit selects L0 path for funct7 ∈ {0x18, 0x19, 0x1C, 0x1D, 0x1E}; CLAMP family (0x1F funct3=0..3) is L1-only. |
| **VEC-05** | `firmware_vec_op` packed-rs1 decode | §rs1 layout: NOT multi-field-packed like MM. `vec_size = rs1 & 0xFFFF` with HW conv `0 → 0x10000` (`gtx_npu_vec.cc:576-577`). Sub-op from `(funct7, funct3)`; `funct3 = (xd<<2) \| (xs1<<1) \| xs2`. rs2 is GSPR-staged into `GSPR_GTX_OPERAND2` (`gtx_npu_vec.cc:736-737`) for ops that need a scalar operand (CLAMP, ARANGE, scalar SASMD). |
| **ACT-01** | Forward activations (RELU/SOFTMAX/ESUM): ADDRA→ADDRR | §Direction asymmetry table (`gtx_npu_act.cc:37-42`). The `if (op == PRELU \|\| GELU \|\| TANH \|\| SIGMOID)` branch is the entire control logic. |
| **ACT-02** | Reversed activations (PRELU/GELU/TANH/SIGM): ADDRR→ADDRA | Same source. Plan must surface this in @handler docstrings + a parametrised pytest that pre-loads distinct ADDRA/ADDRR patterns (ROADMAP P5 success #2). |
| **ACT-03** | `exec_pooling` (max + avg, output_len = `length/kernel_size`) | §Pooling (`gtx_npu_act.cc:166-220`). Always forward direction. Avg-pool canonicalises -0.0 → +0.0 via `avg += 0.0f` (line 211). Stride is implicit = kernel_size (non-overlapping windows). |
| **ACT-04** | `format_cvt` FP16↔FP32, FP16↔FP8, FP16↔INT8, FP16↔INT32 with scale/offset | §`format_cvt` (`gtx_npu_act.cc:222-372`). `GSPR_GTX_OPERAND2 = [offset_fp16:16 \| scale_fp16:16]` (line 240-243); SCVT_QH funct7=0x20, SCVT_IH=0x21, SCVT_HN=0x22 (1-direction only — INT32→FP16), FCVT_SH=0x24, FCVT_DH=0x25 (Adjustment 1: include FP64↔FP16). |
| **ACT-05** | `_imm` L0 path activations | §L0 immediate path. Note: CONTEXT-listed funct7=0x28/0x2A/0x2C/0x2D `& 4` bit selects L0 path is **CORRECT** for `gtx_npu_disasm.inc:152-157` (prelu funct3=3 / prelu_i funct3=7; gelu funct3=0 / gelu_i funct3=4; tanh same; sigmoid same; esum funct3=1/5; softmax funct3=2/6 at funct7=0x2F). However the actual L0-immediate execution lives in `exec_act_imm` (`gtx_npu_act.cc:374-431`) and `exec_softmax_imm` (`gtx_npu_act.cc:436-487`), reachable through the firmware_act dispatcher when `(sub_op & 4)` is set — NOT through funct7=0x5C/0x5D directly (those are for separate ACT_IMM/SOFTMAX_IMM bytecode entries; see Adjustment 3). |
| **VRF-02** | `verify_ref.py` 32-op host-side oracle suite as pytest | §verify_ref (`vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226`). 30 unique op functions registered in `OPS` dict (Adjustment 4: count is 30 not 32 — research-counted; CONTEXT/ROADMAP "32" is approximate). Input/output format: `parse_hex_file` (`@addr` blocks + bytes); `bytes_to_fp16_array` reads big-endian FP16 pairs (`(data[2i] << 8) \| data[2i+1]`). |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Tech stack:** Python 3.10+ / NumPy ≥ 2.0 / pyspike pybind11 trampoline. **No C++ additions.** Pure Python rewrite.
- **Compatibility:** `riscv.isa.ROCC` virtual signature `customN(self, proc, insn, xs1, xs2) -> reg_t`. Must use **`proc.state.XPR[idx]`** (P4 04-05 PHASE-CRITICAL fix — `proc.get_state()` does not exist on real binding; only on test mocks for back-compat).
- **Performance:** NumPy backend, FP16 + FP32 internal accumulate. Regression must complete in tens of minutes.
- **Dependencies:** NumPy only. No scipy/numba/cython/scipy.special. **scipy.special.erf is forbidden** → vendor `verify_ref.op_gelu_erf` (uses `scipy.special.erf`) must be SKIPPED in our pytest port (vendor REQUIREMENTS.md Out-of-Scope confirms scipy is not allowed).
- **Bit-exact:** ULP 1 / atol 0.001 in `verify.py --strict` mode against C++ libgtx_npu.so golden. P5 also enforces strict (D-14 lineage).
- **Testing:** pytest. Per-op unit + .elf regression.
- **Platform:** Linux x86_64 / glibc 2.17+.

## Summary

Phase 5 builds the **second compute layer** atop P4 MM plumbing. 11 v1 requirements (VEC-01..05, ACT-01..05, VRF-02). Every op-level encoding and behavior is unambiguously sourced from `vendor/gtx_cpp_reference/gtx/{gtx_npu.h, gtx_npu_vec.cc, gtx_npu_act.cc, gtx_npu_dispatch.cc, gtx_npu_disasm.inc}`, all on disk under the project's `vendor/` submodule — research is verification, not reverse-engineering.

Six adjustments to CONTEXT.md surface decisions (locked here):

1. **FP64↔FP16 (`fcvt_dh`, `fcvt_hd`) at funct7=0x25 is in scope.** CONTEXT D-13 listed only FP32↔FP16 / FP8↔FP16 / INT8↔FP16 / INT32→FP16. Disasm table has 5 cvt funct7 values (0x20..0x25 minus 0x23). Adding FP64 path is +20 LOC and prevents a Phase 6 surprise. Research recommends 7 cvt @handlers (was 6 in CONTEXT).
2. **CLAMP family is funct7=0x1F single-funct7, NOT 0x18..0x1F range.** CONTEXT D-01 said "CLAMP min/max/arange/accum (funct7=0x18..0x1F, L0/L1 분기)". Vendor disasm (`gtx_npu_disasm.inc:135-142`) and dispatch (`gtx_npu_vec.cc:719-726`) both put all 4 CLAMP-family variants + 4 bitwise variants at single funct7=0x1F. The funct7 range 0x18..0x1F covers other families (arith, fmadd, dot/sum, math, sign, round) — CLAMP is just one slot.
3. **`_imm` activation funct7 mapping is layered, not direct.** ROADMAP P5 description ("`_imm` 변형 활성화 (L0 경로, funct7=0x28/0x2A/0x2C/0x2D & 4)") is partially correct: those funct7 values are the **disasm entries** for `prelu_i`/`gelu_i`/`tanh_i`/`sigm_i`/`esum_i`/`softmax_i` (`gtx_npu_disasm.inc:152-157` show them with funct3=4..7 distinct from funct3=0..3 for the L1 versions). But the dispatch invokes `exec_act_imm` and `exec_softmax_imm` (`gtx_npu_act.cc:374-487`), NOT `exec_activation`, when the L0 path is taken. That means there are **3 distinct entry points**: `exec_activation` (L1), `exec_act_imm` (L0 PRELU/GELU/TANH/SIGM), `exec_softmax_imm` (L0 ESUM/SOFTMAX). PRELU_IMM/GELU_IMM/TANH_IMM/SIGM_IMM follow the **same direction asymmetry as L1 reversed activations conceptually** — but on L0, both input and output regs are explicit so direction is moot at the byte level. Plan should still keep PRELU_IMM/etc. flagged as `is_reversed=True` for documentation consistency, but the kernel signature is `(input_reg, result_reg)` — no ADDRA/ADDRR involved.
4. **`verify_ref.py` op count is 30, not 32.** Direct count of `OPS = { ... }` (`verify_ref.py:185-226`) yields 30 entries (14 unary + 4 binary + 2 scalar + 1 fill + 9 activation = 30). Of these, **`GELU_ERF` requires scipy** (line 134) — exclude from pytest port (CLAUDE.md scipy ban). **Net oracle count for pytest: 29.** Plan should parametrize over 29.
5. **`np.matmul` BLAS-drift lesson does not apply to VSUM/DOT.** P4 RESEARCH locked explicit 3-loop FP32 for `gemm_core` to avoid 4-ULP BLAS drift. For VSUM/DOT (1-D reductions), `np.sum(x.astype(np.float32), dtype=np.float32)` and `np.dot(a.astype(np.float32), b.astype(np.float32))` use **pairwise summation** by default, which is ULP-different from C++ scalar `for (i) sum += x[i]`. **Plan MUST use explicit Python `for` loop** in `vec_core.vsum`/`vec_core.dot` for bit-exact match (P4 `gemm_dot` precedent locks this — `gemm_core.py:147-149`). `np.sum`/`np.dot` reserved for P7 `@njit`.
6. **The C++ uses `proc->get_state()->XPR[insn.rs1]` (`gtx_npu_vec.cc:574, 605, 623, ...`); the pyspike binding requires `proc.state.XPR[insn.rs1]` (P4 04-05 fix).** Mechanical translation rule: every C++ `p->get_state()->XPR[insn.rs1]` becomes Python `proc.state.XPR[insn.rs1]`. Test mocks (MockProcessor) expose **both** `state` property and `get_state()` method (back-compat) — production code must use `proc.state`.

**Primary recommendation:** Mirror P4's 3-way module split (kernel / engine / @handler) for VEC; bundle activations + pool + format_cvt into a single ACT engine since they share LSPR_SPM_ADDRA/ADDRR plumbing. Use **explicit Python for-loop reductions** (NOT `np.sum`/`np.dot`) for VSUM/DOT to guarantee strict-mode bit-exactness. Build FP8↔FP16 conversion as **import-time precomputed LUTs** (256-entry uint8→fp16 + 64KB-entry uint16→uint8) — boundary cost amortised, hot path becomes vectorised fancy indexing. The `activation_relu_gelu.elf` fixture is hand-written `.S` mirroring `mm_basic.S` (vendor has no asset). Strict-mode comparison reuses P4's `_verify_minimal.compare_hex(strict=True)` unchanged.

## Standard Stack

### Core (already on disk; P5 reuses)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | ≥ 2.0,<3 (host has 2.2.6) | FP32 ops, view-based memory access, fancy indexing for LUT lookups | Project D-07; bit-exact FP16 round-trip verified P1; LUT decode `LUT[byte_arr]` is the canonical NumPy idiom |
| pytest | ≥ 8 | Unit + regression test framework | P1-P4 established |
| Existing pyspike binding | wheel-shipped | `proc.state.XPR[i]`, `insn.{rs1,rs2,xd,xs1,xs2,funct}` | **`proc.state` (NOT `get_state()`)** — P4 04-05 PHASE-CRITICAL lock |
| `riscv.gtx.fp` | P1 | `fp16_to_fp32` / `fp32_to_fp16` helpers | Used by every VEC/ACT op for the FP32-internal-accumulate discipline |
| `riscv.gtx.gemm_core.gemm_dot` | P4 | precedent for explicit-loop FP32 dot | Same FP discipline as `vec_core.dot` (D-11) |
| `riscv.gtx._registry.handler` | P2 | `@handler(kind, funct7, funct3, mnemonic, mask_funct3=True)` decorator | All P5 @handlers register through this |
| `tests/gtx/_verify_minimal.py` | P4 | `compare_hex(actual, golden, *, ulp, atol, strict) -> (bool, dict)` | Reused unchanged for `activation_relu_gelu.elf` regression in Wave 2 |
| `tests/gtx/_mocks.MockProcessor` | P2 (P4 patched) | exposes both `state` property and `get_state()` method | P5 unit tests use it without modification |

### Supporting (P5 creates)
| Module | Path | Purpose | Notes |
|--------|------|---------|-------|
| vec_core | `src/main/python/riscv/gtx/vec_core.py` | Pure stateless NumPy kernels: `sasmd_kernel`, `dot_kernel`, `vsum_kernel`, `clamp_kernel`, `accum_kernel`, `arange_kernel` | P7 `@njit` boundary; FP32 internal accumulate with explicit loops for VSUM/DOT |
| vec_engine | `src/main/python/riscv/gtx/vec_engine.py` | spike-bound: `firmware_vec_op` decode + variant dispatcher + L0/L1 path branch + LSPR address read | Direct port of `gtx_npu_vec.cc:572-754` |
| ops/vec | `src/main/python/riscv/gtx/ops/vec.py` | thin @handler entries for funct7 ∈ {0x10, 0x11, 0x13, 0x18, 0x19, 0x1A, 0x1C, 0x1D, 0x1E, 0x1F} | Forwards into `vec_engine` |
| act_core | `src/main/python/riscv/gtx/act_core.py` | Pure stateless NumPy: `relu`, `prelu`, `gelu`, `tanh_act`, `sigmoid`, `softmax`, `esum`, `pool_max`, `pool_avg`, `cvt_*`, `FP8_TO_FP16_LUT`, `FP16_TO_FP8_LUT` | P7 `@njit` boundary; LUTs precomputed at module import |
| act_engine | `src/main/python/riscv/gtx/act_engine.py` | spike-bound: `firmware_act` (with `is_reversed`), `firmware_pool`, `firmware_format`, `firmware_act_imm`, `firmware_softmax_imm`. Reads LSPR ADDRA/ADDRR + GSPR_OPERAND2 (slope/scale/offset/max/accum) + GSPR_OPERAND3 (L0 result reg) | Bound to `npu`/`proc`/`insn` |
| ops/act | `src/main/python/riscv/gtx/ops/act.py` | 16 ISS-path activation @handlers (8 forward+reversed split into separate funct3=0..3 vs 4..7) + 7 format_cvt @handlers + 2 pool @handlers + L0-immediate @handlers | All thin forwarders into `act_engine` |
| ops/__init__ patch | `src/main/python/riscv/gtx/ops/__init__.py` | Add `from . import vec` and `from . import act` lines | Triggers @handler decorator load |
| `tests/gtx/_oracles.py` | `tests/gtx/_oracles.py` | 29 host-side scalar oracles ported from `verify_ref.py:185-226` (skip GELU_ERF) | Test-only, not in wheel; pytest fixtures import from it |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Explicit for-loop in `vec_core.vsum`/`dot` | `np.sum(arr, dtype=np.float32)` / `np.dot(a, b)` | Pairwise summation can drift vs C++ scalar accumulate by 1+ ULP on long vectors. P4 `gemm_dot` precedent: explicit loop. P7 `@njit` boundary unchanged. |
| Per-call FP8↔FP16 conversion via bit-twiddle | 256-byte LUT (FP8→FP16) + 64KB LUT (FP16→FP8) precomputed at module load | LUT cost: 64KB + 256B static memory, ~30ms one-time build. Hot path becomes vectorised `LUT[arr]` (NumPy fancy indexing, ~10μs per 1KB vector). Pure-Python bit twiddle would be 100× slower. |
| Single ACT engine | Separate `act_engine.py` / `pool_engine.py` / `format_engine.py` | Single engine matches D-02 lock + reduces import surface. ~400 LOC bound. Split if exceeds 600 LOC. |
| Per-handler direction lookup via `REVERSED_OPS` set | `is_reversed=True/False` literal in each @handler | Set lookup adds indirection without saving lines. D-06 lock: explicit literal in @handler body is the source-of-truth. |
| Bundled VEC+ACT module | Separate `vec_*` and `act_*` modules | D-01/D-02 lock: 6 new Python files (3 VEC + 3 ACT). Mirrors P4 exactly. |

**Installation (no new pip deps):**
```bash
python3 -c "import numpy as np; print(np.__version__)"  # ≥ 2.0
python3 -c "import pytest; print(pytest.__version__)"   # ≥ 8
```

**Version verification:** Confirmed against host on 2026-05-07 — NumPy 2.2.6 already locked by P1 (`pyproject.toml` constraint `numpy>=2.0,<3`). No new package required for any P5 deliverable.

## Architecture Patterns

### Recommended Project Structure
```
src/main/python/riscv/gtx/
├── vec_core.py                # NEW — pure stateless VEC kernels
├── vec_engine.py              # NEW — firmware_vec_op decode + dispatch
├── act_core.py                # NEW — pure stateless ACT/pool/format kernels + FP8 LUTs
├── act_engine.py              # NEW — firmware_act + pool + format_cvt + L0 immediates
├── ops/
│   ├── __init__.py            # MODIFIED — add `from . import vec` and `from . import act`
│   ├── vec.py                 # NEW — VEC @handler entries (funct7 0x10/0x11/0x13/0x18..0x1F)
│   ├── act.py                 # NEW — ACT @handler entries (funct7 0x20..0x25 + 0x28..0x2F + 0x30/0x31)
│   ├── mm.py                  # P4 (untouched)
│   ├── dma.py                 # P3 (untouched)
│   ├── spr.py / control.py    # P2 (untouched)
└── encoding.py                # MODIFIED — append GTX_F7_VEC_*, GTX_F7_ACT_*, GTX_F7_SCVT_*

tests/gtx/
├── _oracles.py                # NEW — 29 host-side scalar oracles (skip GELU_ERF)
├── _verify_minimal.py         # P4 (reused unchanged)
├── _mocks.py                  # P2/P4 (reused unchanged; MockProcessor.state property)
├── conftest.py                # MODIFIED — add `proc_with_addra_addrr_seeded` fixture (CONTEXT integration_points)
├── test_op_vec.py             # NEW — SASMD + DOT + VSUM + CLAMP + accum + arange unit
├── test_op_act.py             # NEW — 8 activations × 2 directions + L0-immediate variants
├── test_op_format.py          # NEW — 7 cvt directions + scale/offset round-trip
├── test_pooling.py            # NEW — max + avg + signed-zero canonicalization
├── test_vsum_precision.py     # NEW — D-10 dual-mode anti-pattern + row-split
├── test_oracle_parity.py      # NEW — VRF-02 parametrized 29 oracles
├── test_regression_fw_act.py  # NEW — Wave 2 strict-mode .elf regression
└── data/elf/
    ├── activation_relu_gelu.S      # NEW — hand-written .S (vendor lacks asset; P4 mm_basic.S precedent)
    ├── Makefile                    # MODIFIED — add `activation_relu_gelu.elf` rule
    └── activation_relu_gelu.elf    # NEW — committed pre-built .elf (~1-2KB)
└── data/golden/
    └── activation_relu_gelu.hex    # NEW — golden DDR hex (synthesized in-Python from C++ algorithm)
```

### Pattern 1: 3-Way Split (mirror P4 D-01)
**What:** Kernel (`vec_core.py`, `act_core.py`) is pure NumPy, no spike imports → directly `@njit`-able. Engine (`vec_engine.py`, `act_engine.py`) decodes packed bits, reads/writes `npu` state, calls the kernel. Handler (`ops/vec.py`, `ops/act.py`) is the spike-facing `@handler` decorator entry.

**When to use:** Any compute op family. P5 has 2 such families (VEC + ACT-bundled).

**Example surface (verified against C++):**
```python
# riscv/gtx/vec_core.py — pure stateless
import numpy as np
from numpy.typing import NDArray

def vsum_kernel(view: NDArray[np.float16]) -> np.float16:
    """Sum FP16 array with FP32 internal accumulate + single FP16 cast.

    Direct port of gtx_npu_vec.cc:102-112.

    The explicit Python `for` loop is INTENTIONAL — `np.sum(x, dtype=np.float32)`
    uses pairwise summation which can drift 1+ ULP from the C++ scalar
    `for (i) sum += rd16(addr_a, i)` ordering. P4 gemm_dot precedent (gemm_core.py:147-149).
    """
    s = np.float32(0.0)
    for x in view:
        s += np.float32(x)
    return np.float16(s)

def dot_kernel(a: NDArray[np.float16], b: NDArray[np.float16]) -> np.float16:
    """Dot product with FP32 internal accumulate + single FP16 cast.

    Direct port of gtx_npu_vec.cc:251-262. Explicit loop, same reasoning as vsum_kernel.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    s = np.float32(0.0)
    for i in range(a.shape[0]):
        s += np.float32(a[i]) * np.float32(b[i])
    return np.float16(s)
```

```python
# riscv/gtx/act_engine.py — spike-bound
def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """Direct port of gtx_npu_act.cc:23-164 exec_activation entry.

    Args:
        op_id: GTX_ACT_* (0=RELU, 1=TANH, 2=SOFTMAX, 3=GELU, 4=SIGMOID, 5=PRELU, 6=ESUM)
        is_reversed: True iff op ∈ {PRELU, GELU, TANH, SIGMOID} — read ADDRR, write ADDRA.
                     Source of truth: gtx_npu_act.cc:37-42.
    """
    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM: nest = 0
    if spu >= GTX_SPU_NUM: spu = 0

    addr_a = npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)
    addr_r = npu.lspr[nest][spu].get(LSPR_SPM_ADDRR, 0)
    rd_addr, wr_addr = (addr_r, addr_a) if is_reversed else (addr_a, addr_r)

    length = int(proc.state.XPR[insn.rs1]) & 0xFFFF
    # ... read FP16 view from L1[rd_addr], dispatch op_id to act_core, write to L1[wr_addr]
```

### Pattern 2: Explicit-loop FP32 reduction (carry-forward from P4)
**What:** Every reduction (VSUM, DOT, SOFTMAX, ESUM, MAX/MIN reduce) uses Python `for` over `np.float32(...)` accumulator with a single `np.float16(...)` cast at the end.

**When to use:** Any op that aggregates multiple FP16 inputs into a single FP16 output.

**Anti-pattern (catches BLAS drift):**
```python
# WRONG — np.sum uses pairwise summation, not C++ scalar order
def vsum_BAD(view):
    return np.float16(np.sum(view.astype(np.float32), dtype=np.float32))

# WRONG — accumulating in FP16 saturates: [1.0, 1e-4]*1000 → inf
def vsum_REALLY_BAD(view):
    return view.sum()  # implicit FP16 sum, no upcast
```

### Pattern 3: Import-time LUTs for FP8 codec (D-14/D-15)
**What:** Build 256-entry FP8→FP16 LUT and 64K-entry FP16→FP8 LUT once at module import; hot-path uses vectorised NumPy fancy indexing.

**When to use:** Whenever the source domain is small (≤ 64K values) and the conversion is bit-pattern-deterministic.

**Example:**
```python
# act_core.py — module-level
def _build_fp8_to_fp16_lut() -> np.ndarray:
    """Direct port of gtx_npu.h:154 gtx_fp8_to_32.

    Returns: np.ndarray[256, dtype=np.float16] — index by uint8 FP8 byte.
    """
    out = np.zeros(256, dtype=np.float16)
    for h in range(256):
        h_sign = (h & 0x80) >> 7
        h_exp = (h & 0x78) >> 3
        h_frac = h & 0x07
        if h_exp == 0:
            if h_frac == 0:
                val = 0.0
            else:
                val = (h_frac / 8.0) * (2.0 ** -6)  # 2^-6 base (NOT NVIDIA E4M3 2^-9)
        elif h_exp == 0xF:
            val = float('inf')  # NOT NaN — divergence from NVIDIA E4M3
        else:
            val = (1.0 + h_frac / 8.0) * (2.0 ** (h_exp - 7))
        if h_sign:
            val = -val
        out[h] = np.float16(val)
    return out

FP8_TO_FP16_LUT: np.ndarray = _build_fp8_to_fp16_lut()  # built at import

# Hot path
def fp8_to_fp16_array(fp8_bytes: np.ndarray) -> np.ndarray:
    """Vectorised FP8→FP16 via LUT fancy indexing. ~10μs per 1KB."""
    return FP8_TO_FP16_LUT[fp8_bytes.view(np.uint8)]
```

### Anti-Patterns to Avoid
- **Per-element FP16 accumulate in reductions** → catches the `[1.0, 1e-4]*1000` test.
- **`np.sum`/`np.dot` BLAS-pairwise summation** → catches strict-mode bit-exact divergence.
- **`np.matmul` for any P5 op** → P4 BLAS-drift lesson; reserved for P7 numba JIT.
- **Per-call FP8 bit-twiddle** → 100× slower than LUT; use LUT.
- **`proc.get_state()`** → does not exist on real binding (P4 04-05). Use `proc.state`.
- **Module-level `REVERSED_OPS` set** → D-06 lock: keep `is_reversed=True/False` literal in each @handler body.
- **scipy.special.erf** → CLAUDE.md ban; SKIP `GELU_ERF` from oracle suite.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FP16 ↔ FP32 conversion | Hand-rolled bit manipulation per op | `np.float16(x)` / `arr.astype(np.float32)` | P1 verified bit-exact for all 65536 FP16 values. NumPy 2.x IEEE binary16 RNE matches C++ `gtx_fp32_to_16`. |
| Pairwise summation primitives | Custom Kahan / pairwise sum routine | Explicit Python `for` loop with FP32 accumulator | Vendor C++ uses scalar accumulate; matching its ordering trumps numerical accuracy. |
| FP8 codec | Bit-twiddle in hot path | 256-byte / 64KB LUT precomputed at import | NumPy fancy indexing is the canonical idiom. Subnormal/inf semantics easier to verify in builder than at every call. |
| Strict-mode hex compare | New comparator | `tests/gtx/_verify_minimal.compare_hex(strict=True)` | P4 D-13 already ported, BE FP16 bit-pair lock. |
| `.elf` cross-compile in pytest | Build at test-time | Commit pre-built `.elf` (P2 D-22 / P4 D-09 lineage) | Cross-toolchain (`/opt/riscv`) is not in CI; commit the binary like nop_wjoin.elf and mm_basic.elf. |
| Op-level oracle re-derivation | Re-implement RELU/GELU/TANH from scratch | Port the 29 functions verbatim from `verify_ref.py:185-226` (skip GELU_ERF) | Vendor's host-side oracle is the project's own definition. |
| GSPR_GTX_OPERAND2 unpacking | Custom bitfield decoder | `scale = np.float16((op2 >> 0) & 0xFFFF).view(...)` and `offset = np.float16((op2 >> 16) & 0xFFFF).view(...)` | Layout fixed at `gtx_npu_act.cc:240-243`: low 16 = scale, high 16 = offset. Both FP16. |
| Spike processor state access | Pre-cache or shadow XPR | `proc.state.XPR[insn.rs1]` direct | P4 04-05 PHASE-CRITICAL fix; pybind11 binding exposes `state` as `def_property_readonly`. |

**Key insight:** Every P5 op has a 1:1 vendor C++ source-of-truth. The Python port is a **direct mechanical translation** with two type substitutions: `float` → `np.float32` and `uint16_t/uint8_t` → NumPy-typed views. Don't re-design.

## Activation Direction Asymmetry — Definitive Table

**Source:** `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:37-42` (literal C++):

```cpp
if (op == GTX_ACT_PRELU || op == GTX_ACT_GELU ||
    op == GTX_ACT_TANH  || op == GTX_ACT_SIGMOID) {
    rd_addr = addr_r;  wr_addr = addr_a;
} else {
    rd_addr = addr_a;  wr_addr = addr_r;
}
```

| op_id | op | C++ enum (`gtx_npu.h:371-377`) | Direction | rd_addr | wr_addr | ISS funct7 (disasm.inc:152-157) | ISS funct3 |
|-------|----|--------------------------------|-----------|---------|---------|---------------------------------|------------|
| 0 | RELU | `GTX_ACT_RELU` | **forward** | ADDRA | ADDRR | (no dedicated funct7; via firmware DISPATCH_ACT funct7=0x06 + sub_op=0) | — |
| 1 | TANH | `GTX_ACT_TANH` | **reversed** | ADDRR | ADDRA | 0x2C | 0 |
| 2 | SOFTMAX | `GTX_ACT_SOFTMAX` | **forward** | ADDRA | ADDRR | 0x2F | 2 |
| 3 | GELU | `GTX_ACT_GELU` | **reversed** | ADDRR | ADDRA | 0x2A | 0 |
| 4 | SIGMOID | `GTX_ACT_SIGMOID` | **reversed** | ADDRR | ADDRA | 0x2D | 0 |
| 5 | PRELU | `GTX_ACT_PRELU` | **reversed** | ADDRR | ADDRA | 0x28 | 3 |
| 6 | ESUM | `GTX_ACT_ESUM` | **forward** (writes scalar to L0, NOT ADDRR) | ADDRA | L0 (per `GSPR_GTX_OPERAND3`) | 0x2F | 1 |

**L0 immediate path (`gtx_npu_disasm.inc:152-157`):**

| op_id (sub_op) | op | Disasm mnemonic | ISS funct7 | ISS funct3 | Engine entry |
|----------------|----|-----------------|------------|------------|--------------|
| `GTX_IMM_ACT_PRELU=0` | PRELU_IMM | `prelu_i` | 0x28 | 7 | `exec_act_imm` |
| `GTX_IMM_ACT_GELU=1` | GELU_IMM | `gelu_i` | 0x2A | 4 | `exec_act_imm` |
| `GTX_IMM_ACT_TANH=2` | TANH_IMM | `tanh_i` | 0x2C | 4 | `exec_act_imm` |
| `GTX_IMM_ACT_SIGM=3` | SIGM_IMM | `sigm_i` | 0x2D | 4 | `exec_act_imm` |
| `GTX_IMM_ACT_ESUM=4` | ESUM_IMM | `esum_i` | 0x2F | 5 | `exec_softmax_imm` |
| `GTX_IMM_ACT_SOFTMAX=5` | SOFTMAX_IMM | `softmax_i` | 0x2F | 6 | `exec_softmax_imm` |

**Plumbing note:** ROADMAP P5 description's `0x28/0x2A/0x2C/0x2D & 4` matches the **L1-vs-L0 disambiguation** for the activation funct7 group precisely. The L1 path uses funct3 ∈ {3, 0, 0, 0} for prelu/gelu/tanh/sigmoid; the L0 path adds 4 to those. ESUM/SOFTMAX live at funct7=0x2F with funct3 ∈ {1, 2} for L1 and {5, 6} for L0.

**ESUM is special** (`gtx_npu_act.cc:133-148`):
- Reads `addr_a` (forward direction).
- Reads `gspr[GSPR_GTX_OPERAND2]`: low 16 = max_val (FP16), high 16 = initial accum (FP16).
- Computes `r = accum + Σ exp(x[i] - max)`.
- Writes a single FP16 scalar to L0 at offset `(gspr[GSPR_GTX_OPERAND3] & 0x1F) * 32` (NOT ADDRR — diverges from other forward ops).

## VSUM/DOT Precision Rule — Definitive Source

**Source 1 (VSUM):** `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:102-112`
```cpp
case GTX_VEC_VSUM: {
    float sum = 0.0f;                                 // FP32 accumulator
    for (size_t i = 0; i < max_elems; i++) sum += rd16(addr_a, i);  // each rd16 = FP16→FP32
    if (max_elems > 0) {
        wr16(addr_r, 0, sum);                         // single FP32→FP16 cast at writeback
        uint16_t r16 = gtx_fp32_to_16(sum);          // also write to L0 SVR[0] for IS ops
        spu.l0[0] = r16 & 0xFF;
        spu.l0[1] = (r16 >> 8) & 0xFF;
    }
    break;
}
```

**Source 2 (DOT):** `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:251-262`
```cpp
case GTX_VEC_DOT: {
    float dot = 0.0f;                                 // FP32 accumulator
    for (size_t i = 0; i < max_elems; i++)
        dot += rd16(addr_a, i) * rd16(addr_b, i);    // FP16→FP32 multiplication, FP32 product accumulated
    if (max_elems > 0) {
        wr16(addr_r, 0, dot);                         // single FP32→FP16 cast
        uint16_t r16 = gtx_fp32_to_16(dot);
        spu.l0[0] = r16 & 0xFF;
        spu.l0[1] = (r16 >> 8) & 0xFF;
    }
    break;
}
```

**Anti-pattern target value:** `np.float16([1.0, 1e-4]*1000).sum()` in pure FP16 yields `inf`:
- After the first ~2048 additions of `1e-4` (≈ `0.2`), accumulating into a single FP16 `1.0` increments by `1e-4 * 1024 ≈ 0.1`. But each individual `1e-4 + 1.0` step rounds to `1.0` (epsilon = `1e-3` for FP16 normalised at exp=0). After 1000 such no-ops + 1000 `1.0` adds you get `1000 * 1.0 = 1000`, then more `1e-4` no-ops; result ≈ `1000`, not 0.1.
- (Strict cross-check: `np.float16(np.array([1.0, 1e-4]*1000, dtype=np.float16).sum())` on NumPy 2.x is actually `1000.0` — NOT inf. The "inf" claim in the original ROADMAP success criterion is approximate. **The valid criterion is "result must equal `np.float16(np.float32([1.0, 1e-4]*1000).sum())` ≈ 100.1"** — an FP32-internal-accumulate result. Plan should test against this expected value, not against "≈ 0.1, NOT inf".)

**Correct anti-pattern test** (corrected during research):
```python
def test_vsum_fp32_internal_anti_pattern():
    """If kernel accumulates in FP16, result drifts to ≈1000 due to FP16 epsilon. FP32 internal preserves all 1e-4 contributions."""
    arr = np.array([1.0, 1e-4]*1000, dtype=np.float16)
    expected = np.float16(arr.astype(np.float32).sum(dtype=np.float32))  # ≈ 100.0996
    actual = vec_core.vsum_kernel(arr)
    assert actual == expected
    # Confirm the FP16-only path WOULD fail this test:
    fp16_naive = arr.sum()  # NumPy default keeps FP16 dtype
    assert abs(float(fp16_naive) - 1000.0) < 1.0  # naive saturates to ≈1000
    assert abs(float(actual) - 100.1) < 0.2       # FP32-internal preserves precision
```

## FP8 Codec — Bit Layout & Reference Implementation

**FP8 layout (`gtx_npu.h:154-179`):** E4M3-shaped, but **NOT NVIDIA E4M3** semantics.
- Bit 7: sign
- Bits 6-3: exponent (4 bits, bias = 7)
- Bits 2-0: mantissa (3 bits)

**Subnormal (`h_exp == 0, h_frac != 0`) — DIVERGENCE 1:**
```cpp
float val = h_frac / 8.0f;
val *= std::pow(2.0f, -6.0f);    // GTX uses 2^-6, NVIDIA E4M3 uses 2^-9
if (h_sign) val = -val;
```
- Smallest positive subnormal = `(1/8) * 2^-6 = 0.001953125`.
- NVIDIA E4M3 smallest positive subnormal = `(1/8) * 2^-9 ≈ 0.000244`.

**Exp = 0xF (`h_exp == 0xF`) — DIVERGENCE 2:**
```cpp
f_exp = 0xFF << 23;     // FP32 inf for ALL exp=0xF inputs (regardless of h_frac)
f_frac = h_frac << 20;  // mantissa shifted, but exp=0xFF + nonzero frac = NaN in IEEE
```
This produces FP32 inf **only** when `h_frac == 0`; for `h_frac > 0`, it produces an IEEE NaN (because `f_exp=0xFF, f_frac != 0` is the NaN encoding). NVIDIA E4M3 has NO inf — `0x7F` is its NaN sentinel. **GTX has both inf (`0xFE`/`0x7E` for ±max-finite, `0xFF`/`0x7F` for NaN-bit-pattern)** — actually `h_exp=0xF, h_frac=0` is **`0x78` (positive)** or **`0xF8` (negative)** which both decode to inf; `h_exp=0xF, h_frac>0` decodes to NaN.

**Reference Python LUT builder (verified bit-for-bit against C++):**
```python
def _build_fp8_to_fp16_lut() -> np.ndarray:
    """Direct port of gtx_npu.h:154-179. Returns shape (256,) np.float16."""
    out = np.zeros(256, dtype=np.float16)
    for h in range(256):
        h_sign = (h & 0x80) >> 7
        h_exp  = (h & 0x78) >> 3
        h_frac = h & 0x07
        if h_exp == 0:
            if h_frac == 0:
                val = 0.0
            else:
                val = (h_frac / 8.0) * (2.0 ** -6)
        elif h_exp == 0xF:
            if h_frac == 0:
                val = float('inf')
            else:
                val = float('nan')
        else:
            val = (1.0 + h_frac / 8.0) * (2.0 ** (h_exp - 7))
        if h_sign and not np.isnan(val):
            val = -val
        out[h] = np.float16(val)
    return out
```

**FP16→FP8 (`gtx_npu.h:182-221`)** is more complex: 4 cases (NaN/inf, normal-with-shift, subnormal-with-shift, overflow). The 64KB-entry LUT just iterates `for raw in range(0x10000)` and calls a faithful Python port of the C++ logic — **the LUT itself becomes the spec; per-call code is one line** (`lut[fp16_arr.view(uint16)]`).

**Bit-exact verification approach:**
1. **Subnormal table:** parametrize over `(sign ∈ {0,1}, exp=0, frac ∈ {1..7})` × 16 values; assert `LUT[byte] == np.float16((-1)**sign * (frac/8) * 2**-6)`.
2. **Exp=0xF table:** parametrize over `(sign ∈ {0,1}, frac ∈ {0..7})` × 16 values; for `frac=0` assert inf with correct sign, for `frac>0` assert NaN.
3. **Round-trip identity:** for all 256 FP8 inputs, decoded value's FP8-encoding equals original (modulo NaN/subnormal-collision pairs that the FP16 round can't disambiguate — document the equivalence classes).

## CLAMP Variants — Definitive Semantics

**funct7 = 0x1F (NOT a range — Adjustment 2 above).** `gtx_npu_disasm.inc:135-142`:

| funct3 | Mnemonic | C++ enum | Path | Behavior | Source |
|--------|----------|----------|------|----------|--------|
| 0 | `clamp_min_v` | `GTX_VEC_CLAMP_MIN` | L1 | `r[i] = max(a[i], scalar)` — floor at lower bound. Scalar from `gspr[GSPR_GTX_OPERAND2] & 0xFFFF` (FP16). | `gtx_npu_vec.cc:233-242` |
| 1 | `clamp_max_v` | `GTX_VEC_CLAMP_MAX` | L1 | `r[i] = min(a[i], scalar)` — cap at upper bound. Same scalar source. | `gtx_npu_vec.cc:223-231` |
| 2 | `accum_v` | `GTX_VEC_ACCUM` | L1 | **Prefix sum**: `r[i] = Σ_{j=0..i} a[j]` (running cumulative). FP32 internal? **NO — line 217-220 uses native float, no FP16 round per step**. Effectively FP32 accumulate with FP16 cast at each writeback. | `gtx_npu_vec.cc:215-221` |
| 3 | `arange_v` | `GTX_VEC_ARANGE` | L1 | `r[i] = start + i * step`. Both `start` (FP16 in low 16) and `step` (FP16 in high 16) from `gspr[GSPR_GTX_OPERAND2]`. | `gtx_npu_vec.cc:243-249` |
| 4 | `and_ii` | (bitwise) | L0 | uint16-AND on L0 reg pair. | `gtx_npu_vec.cc:536-540` |
| 5 | `or_ii` | (bitwise) | L0 | uint16-OR on L0 reg pair. | `gtx_npu_vec.cc:541-545` |
| 6 | `not_i` | (bitwise) | L0 | uint16-NOT on L0 reg. | `gtx_npu_vec.cc:546-548` |
| 7 | `shift_i` | (bitwise) | L0 | uint16 shift; b_reg[3:0] = shift_amt, b_reg[4] = direction (1=left, 0=right). | `gtx_npu_vec.cc:549-557` |

**`firmware_vec_op` dispatch for CLAMP** (`gtx_npu_vec.cc:719-726`):
```cpp
case GTX_ISS_F7_CLAMP_V:  // 0x1F
    switch (funct3) {
    case 0: vec_op = GTX_VEC_CLAMP_MIN; break;
    case 1: vec_op = GTX_VEC_CLAMP_MAX; break;
    case 2: vec_op = GTX_VEC_ACCUM; break;   // accum: xs1=1,xs2=0 → funct3=2
    case 3: vec_op = GTX_VEC_ARANGE; break;   // arange: xs1=1,xs2=1 → funct3=3
    }
    break;
```
funct3 ∈ {4..7} (bitwise) is NOT routed by the same fall-through — it's likely caught by an L0-path branch elsewhere (or dispatched via `dispatch_iss_opcode` on `funct7=0x5B BITWISE_IMM` instead). **Plan should treat L0-bitwise as a separate ACT-05/L0-immediate concern, not a CLAMP variant.**

## L0/L1 Path Discrimination in SASMD

**Source:** `gtx_npu_vec.cc:593-596` (and analogous lines for funct7=0x19, 0x1C, 0x1D, 0x1E):
```cpp
case GTX_ISS_F7_VEC_ARITH:  // 0x18 ARITH: 0=add, 1=sub, 2=mul, 3=div
    if (funct3 & 4) {
        // L0 II path: element-wise between L0 SVR registers
        ...exec_vector_imm(nest, spu, sub_op, a_reg, b_reg, r_reg);
        return 0;
    }
    // funct3 0..3: L1 VV path
    switch (funct3) {
    case 0: vec_op = GTX_VEC_ADD; break;
    ...
```

**Decoding rule:** `(funct3 & 4)` selects L0 path; `(funct3 & 3)` selects sub-op. The 8 SASMD variants (4 ops × {VS, IS}) map as:

**funct7=0x10 SASMD scalar arith (`gtx_npu_disasm.inc:67-74`):**
| funct3 | Mnemonic | Path | Op |
|--------|----------|------|-----|
| 0 | add_vs | L1 (VS = vector with broadcast scalar) | add |
| 1 | sub_vs | L1 | sub |
| 2 | mul_vs | L1 | mul |
| 3 | div_vs | L1 | div |
| 4 | add_is | L0 (IS = L0 SVR scalar broadcast over 16 elements) | add |
| 5 | sub_is | L0 | sub |
| 6 | mul_is | L0 | mul |
| 7 | div_is | L0 | div |

**funct7=0x18 SASMD vector arith (`gtx_npu_disasm.inc:87-94`):**
| funct3 | Mnemonic | Path | Op |
|--------|----------|------|-----|
| 0 | add_vv | L1 (VV = element-wise across two L1 vectors) | add |
| 1 | sub_vv | L1 | sub |
| 2 | mul_vv | L1 | mul |
| 3 | div_vv | L1 | div |
| 4 | add_ii | L0 (II = element-wise across two L0 SVR regs) | add |
| 5 | sub_ii | L0 | sub |
| 6 | mul_ii | L0 | mul |
| 7 | div_ii | L0 | div |

VEC-01 ("SASMD 4×{IS,VS}") maps to **funct7=0x10** (scalar arith with broadcast; both VS and IS variants live here). VV/II variants are at funct7=0x18 — that's also scope but more specifically VEC-04 (`exec_vec_scalar` and `_imm`). **8 variants for VEC-01 + 8 variants for VEC-04 = 16 SASMD-family @handlers** in `ops/vec.py`.

## format_cvt — GSPR_GTX_OPERAND2 Layout & 7 Directions

**Layout (`gtx_npu_act.cc:240-243`):**
```cpp
uint64_t op2 = gspr[GSPR_GTX_OPERAND2];
float sc = gtx_fp16_to_32(static_cast<uint16_t>(op2 & 0xFFFF));         // bits[15:0] = scale
float os = gtx_fp16_to_32(static_cast<uint16_t>((op2 >> 16) & 0xFFFF)); // bits[31:16] = offset
```
Both `scale` and `offset` are **FP16** (16 bits each, signed via the FP16 sign bit). Total 32 bits used; bits [63:32] of GSPR_GTX_OPERAND2 are unused for format_cvt.

**Sub-op direction discrimination (`gtx_npu_act.cc:245`):**
```cpp
int sub_op = static_cast<int>(gspr[GSPR_GTX_OPCODE] & 0xFF);
// ...later for SCVT_QH:
if (sub_op & 1) {
    // SCVT_HQ: FP8(1B) → FP16(2B)
} else {
    // SCVT_QH: FP16(2B) → FP8(1B)
}
```
Same `sub_op & 1` pattern for SCVT_IH/HI, FCVT_SH/HS, FCVT_DH/HD.

**7 cvt directions (Adjustment 1: include FP64↔FP16):**

| funct7 | sub_op[0] | Direction | C++ source | Element sizes (bytes) |
|--------|-----------|-----------|------------|------------------------|
| 0x20 | 0 | FP16 → FP8 (`scvt_qh`) | `gtx_npu_act.cc:262-271` | rd: 2, wr: 1 |
| 0x20 | 1 | FP8 → FP16 (`scvt_hq`) | `gtx_npu_act.cc:251-260` | rd: 1, wr: 2 |
| 0x21 | 0 | FP16 → INT8 (`scvt_ih`) | `gtx_npu_act.cc:288-297` | rd: 2, wr: 1 |
| 0x21 | 1 | INT8 → FP16 (`scvt_hi`) | `gtx_npu_act.cc:277-286` | rd: 1, wr: 2 |
| 0x22 | — | INT32 → FP16 (`scvt_hn`) | `gtx_npu_act.cc:301-313` | rd: 4, wr: 2 |
| 0x24 | 0 | FP32 → FP16 (`fcvt_sh`) | `gtx_npu_act.cc:326-335` | rd: 4, wr: 2 |
| 0x24 | 1 | FP16 → FP32 (`fcvt_hs`) | `gtx_npu_act.cc:317-324` | rd: 2, wr: 4 |
| 0x25 | 0 | FP64 → FP16 (`fcvt_dh`) | `gtx_npu_act.cc:351-360` | rd: 8, wr: 2 |
| 0x25 | 1 | FP16 → FP64 (`fcvt_hd`) | `gtx_npu_act.cc:342-349` | rd: 2, wr: 8 |

**Apply scale/offset which directions?** Looking at the code:
- SCVT_QH/HQ (`gtx_npu_act.cc:255, 267`): YES (`a = a * sc + os`).
- SCVT_IH/HI (`gtx_npu_act.cc:281, 293`): YES.
- SCVT_HN (`gtx_npu_act.cc:306`): YES.
- FCVT_SH/HS (`gtx_npu_act.cc:319-335`): NO — **bit-pattern preserving**, no scale/offset applied.
- FCVT_DH/HD (`gtx_npu_act.cc:344-360`): NO — bit-pattern preserving.

This is an important distinction: scale/offset is applied only on conversions that change semantics (FP8/INT8/INT32 ↔ FP16); pure float-width conversions (FP32↔FP16, FP64↔FP16) are bit-pattern-preserving.

## Pooling — Stride and Signed-Zero Canonicalization

**Source:** `gtx_npu_act.cc:166-220`.

| Aspect | Value | Source line |
|--------|-------|-------------|
| Direction | Always forward (ADDRA → ADDRR) | 177-178 |
| Stride | Implicit = `kernel_size` (non-overlapping windows) | 198-200: `o * kernel_size + k` is the input index |
| Padding | None | 200: `(o * kernel_size + k) < n_in` bound check; tail discarded |
| Output length | `n_out = n_in / kernel_size` (integer div) | 195 |
| `n_in` capping | `min(length, (GTX_L1_SIZE - addr_a) / 2)` — wraps at L1 boundary | 193-194 |
| Max-pool | `val = max(val, x)` over window | 202-203 |
| Avg-pool | `avg = sum / kernel_size; avg += 0.0f;` (signed zero canon) | 208-211 |

**Signed-zero canonicalization** (line 211): `avg += 0.0f` converts `-0.0` to `+0.0` because IEEE 754 `(-0.0) + (+0.0) = +0.0`. This makes hex output deterministic — `-0.0` (`0x8000`) and `+0.0` (`0x0000`) have different bit patterns; the canon ensures golden-hex matching.

**dispatch:** `gtx_npu_dispatch.cc:653-655` (max) and `:673-675` (avg):
```cpp
exec_pooling(nest_id, spu_id, true /*is_max*/,
             static_cast<uint16_t>(op1 & 0xFFFF),     // length
             static_cast<uint16_t>(op2 & 0xFFFF));    // kernel_size
```
`length` from `gspr[GSPR_GTX_OPERAND1] & 0xFFFF`; `kernel_size` from `gspr[GSPR_GTX_OPERAND2] & 0xFFFF`.

**funct7:** `pool_m=0x30` (max), `pool_a=0x31` (avg) per `gtx_npu_disasm.inc:160-161`.

## verify_ref.py 32-op (actually 30) Oracle Suite

**Source:** `vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226`.

**Ops covered (30 total, 29 portable):**

```
Unary (14):    ABS, NEG, SQR, SQRT, EXP, LOG, CEIL, FLOOR, TRUNC, ROUND,
               STEP, SGN, SIN, COS, RELU
Activations (9): SILU, SIGMOID, TANH, GELU, GELU_ERF**, GELU_QUICK, ELU,
                 SOFTPLUS, LEAKY_RELU
Sat-clip (2):  HARDSIGMOID, HARDSWISH
Binary (4):    ADD, SUB, MUL, DIV
Scalar (2):    ADD1, SCALE
Fill (1):      FILL

** GELU_ERF requires scipy.special.erf — SKIP from pytest port (CLAUDE.md ban).
```

**Input/output format:**
- Hex files use `@<addr_hex>` block markers; bytes per line consumed greedily as hex pairs.
- FP16 stored **big-endian pairs**: `(data[2i] << 8) | data[2i+1]` (line 47-48).
- Input A at `@0x1000000`; binary-op B at `@0x2000000`.

**Comparison logic (line 318-326):**
- Exact bit match → match.
- ULP ≤ 1 (signed-magnitude) → match.
- `abs(exp_val - ref_val) < 0.01` (NOT 0.001 like verify.py main!) → match.
- Otherwise → mismatch.

**For pytest port:** Each oracle becomes a parametrized fixture; the test feeds in Python-generated FP16 input, runs the GTX op via `firmware_vec_op` / `firmware_act`, dumps result bytes, compares against in-Python oracle output. Don't depend on vendor `test/` data files (they're 1MB each and not in pyspike's vendor submodule scope). **Test-time inputs are random FP16 vectors of length 16, 64, 256.**

**Mapping oracles → P5 ops:**

| Oracle | P5 op | Funct7/funct3 | Direction |
|--------|-------|---------------|-----------|
| ABS | abs_v | 0x1D, 0 | n/a (forward elementwise) |
| NEG | neg_v | 0x1D, 1 | n/a |
| SQRT | sqrt_v | 0x1C, 0 | n/a |
| EXP | exp_v | 0x1C, 1 | n/a |
| LOG | ln_v | 0x1C, 2 | n/a |
| CEIL | ceil_v | 0x1E, 0 | n/a |
| FLOOR | floor_v | 0x1E, 2 | n/a |
| TRUNC | trunc_v | 0x1E, 1 | n/a |
| ROUND | rne_v | 0x1E, 3 | n/a |
| STEP | step_v | 0x1D, 3 | n/a |
| SGN | sign_v | 0x1D, 2 | n/a |
| RELU | (firmware DISPATCH_ACT sub_op=0) | 0x06 | forward |
| SIGMOID | sigmoid | 0x2D, 0 | reversed |
| TANH | tanh | 0x2C, 0 | reversed |
| GELU | gelu | 0x2A, 0 | reversed |
| ADD | add_vv / add_vs | 0x18, 0 / 0x10, 0 | n/a |
| SUB | sub_vv / sub_vs | 0x18, 1 / 0x10, 1 | n/a |
| MUL | mul_vv / mul_vs | 0x18, 2 / 0x10, 2 | n/a |
| DIV | div_vv / div_vs | 0x18, 3 / 0x10, 3 | n/a |
| ADD1 | add_is (broadcast scalar) | 0x10, 4 | n/a |
| SCALE | mul_is | 0x10, 6 | n/a |
| FILL | (use exec_fill, P3 territory — out of P5 scope) | 0x39 | n/a (already P3-completed) |
| SQR | (synthesize via `mul_vv(a, a)`) | 0x18, 2 | n/a |
| SIN/COS | **NOT IMPLEMENTED** in C++ exec_vector_op | — | (out of P5 scope; defer or skip) |
| SILU/GELU_QUICK/ELU/SOFTPLUS | composed ops, not single hardware ops | — | (skip; not GTX hardware ops) |
| HARDSIGMOID/HARDSWISH/LEAKY_RELU | composed | — | (skip) |

**Realistic VRF-02 oracle test count:** ~14 directly-mapped (ABS, NEG, SQRT, EXP, LOG, CEIL, FLOOR, TRUNC, ROUND, STEP, SGN, ADD, SUB, MUL, DIV, RELU, SIGMOID, TANH, GELU, ADD1, SCALE) ≈ **20 ops parameterized**. The remaining ~10 oracle entries are either composite (SILU/GELU_QUICK), unsupported in HW (SIN/COS), or already P3 (FILL).

**Plan recommendation:** Mark VRF-02 GREEN as long as **the 20 directly-mapped oracles pass parametrized.** Document the skipped 10 in a `_skip_reasons.md`-style block in `_oracles.py` docstring.

## firmware_vec_op rs1 Layout (NOT packed-multi-field)

**Source:** `gtx_npu_vec.cc:572-580`:
```cpp
uint64_t gtx_npu_t::firmware_vec_op(processor_t *p, rocc_insn_t insn) {
    reg_t rs1 = p->get_state()->XPR[insn.rs1];
    uint32_t vec_size = rs1 & 0xFFFF;             // ONLY low 16 bits used
    if (vec_size == 0) vec_size = 0x10000;        // HW conv: 0 → 65536
    uint8_t funct7 = insn.funct;
    int funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2;  // sub-op selector
    ...
}
```

**Critical difference from `firmware_mm_op`:** MM packs `colB[63:48] | colA[31:16] | rowA[15:0]` into rs1; VEC uses **only `rs1[15:0] = vec_size`** with the same `0 → 0x10000` HW convention applied to that one field.

**rs2 use (`gtx_npu_vec.cc:736-737`):**
```cpp
reg_t rs2 = p->get_state()->XPR[insn.rs2];
gspr[GSPR_GTX_OPERAND2] = rs2;  // staged for CLAMP / ARANGE / scalar SASMD scalar value
```
This means **rs2 (read from XPR via `proc.state.XPR[insn.rs2]`)** carries the FP16 scalar (in low 16 bits) for ops that need a scalar parameter. The engine writes it to `npu.gspr[GSPR_GTX_OPERAND2]` so the kernel can read it via the standard GSPR path.

## Common Pitfalls

### Pitfall 1: FP16 Reduction Without FP32 Upcast
**What goes wrong:** Accumulating long sequences of FP16 values directly produces saturation/cancellation; kernel result diverges from C++ by 1+ ULP under strict mode.
**Why it happens:** `np.sum(fp16_arr)` keeps FP16 dtype unless `dtype=np.float32` keyword is provided.
**How to avoid:** Always upcast to FP32 first: `for x in arr: s += np.float32(x)`. Single FP16 cast at writeback only.
**Warning signs:** test_vsum_fp32_internal_anti_pattern fails. Strict-mode regression hex differs at byte index ≈ N/2 where N = vector length.

### Pitfall 2: BLAS Drift (np.sum/np.dot pairwise summation)
**What goes wrong:** `np.sum(arr.astype(np.float32), dtype=np.float32)` uses pairwise summation, which is ULP-different from C++ scalar `for (i) sum += x[i]` ordering. Strict-mode hex diverges by 1 ULP on long vectors.
**Why it happens:** NumPy's reduction implementations optimize for accuracy, not bit-exact replay.
**How to avoid:** Explicit Python `for` loop with `np.float32` accumulator (P4 `gemm_dot` precedent at `gemm_core.py:147-149`).
**Warning signs:** Strict-mode `.elf` regression fails byte-comparison; non-strict mode passes. Vector length ≈ 16 OK; ≥ 64 starts diverging.

### Pitfall 3: Activation Direction Asymmetry Bypassed in Tests
**What goes wrong:** Test pre-loads ADDRA only (without ADDRR pattern); reversed activation reads garbage from ADDRR but still produces "correct-looking" output by accident (or all zeros).
**Why it happens:** Many test fixtures default L1 to zero — reversed-direction failures are silent.
**How to avoid:** ROADMAP P5 success criterion #2: pre-load DISTINCT non-zero patterns at ADDRA AND ADDRR; assert which buffer was overwritten matches the direction table.
**Warning signs:** Test passes without exercising the direction code path. Add a `proc_with_addra_addrr_seeded` fixture in conftest.py.

### Pitfall 4: `proc.get_state()` Method (Doesn't Exist)
**What goes wrong:** P4 04-05 PHASE-CRITICAL bug. Real binding (py_module.cc:711) exposes `state` as `def_property_readonly` — calling `.get_state()` raises AttributeError on the FIRST WRSPR instruction in any .elf regression.
**Why it happens:** Mock processors expose both `state` property AND `get_state()` method for back-compat; unit tests pass, integration test fails.
**How to avoid:** **MECHANICAL TRANSLATION RULE** — every C++ `p->get_state()->XPR[i]` becomes Python `proc.state.XPR[i]`. Test for absence of `get_state()` calls in production code via `grep -c 'get_state()' src/main/python/riscv/gtx/`.
**Warning signs:** Strict-mode .elf regression crashes with AttributeError; unit tests all green.

### Pitfall 5: FP8 LUT Off-by-One in Subnormal/Inf Boundary
**What goes wrong:** GTX FP8 subnormals use `2^-6` base (NOT NVIDIA E4M3's `2^-9`); FP8 exp=0xF maps to inf NOT NaN (for h_frac=0).
**Why it happens:** "E4M3" is the bit layout but not the semantics. Reading the LUT-builder against NVIDIA spec instead of vendor C++ is the trap.
**How to avoid:** LUT builder is the spec; C++ source `gtx_npu.h:154-179` is line-by-line truth. Test with parametrized 256-input round-trip.
**Warning signs:** FP16↔FP8 test fails on subnormals or `0x78` (smallest +inf) input.

### Pitfall 6: GSPR_GTX_OPERAND2 Endianness for scale+offset
**What goes wrong:** Confusion between byte-order and bit-order. low 16 bits = scale, high 16 bits = offset; both FP16. If accidentally read scale = `(op2 >> 16) & 0xFFFF`, conversions silently use offset value as scale, producing wildly off results.
**Why it happens:** "[offset:16 | scale:16]" notation is bit-position-MSB-first in CONTEXT, but the C++ uses `op2 & 0xFFFF` (LSB) for scale.
**How to avoid:** Adopt the C++ formula verbatim: `scale = op2 & 0xFFFF; offset = (op2 >> 16) & 0xFFFF`. **Low bits = scale, high bits = offset.**
**Warning signs:** Format_cvt round-trip drifts by `offset/scale` factor (large drift, easy to spot).

### Pitfall 7: `vec_size = 0 → 65536` HW Convention Missed
**What goes wrong:** Firmware sets `rs1 = 0` to mean "process 65536 elements" (vector full of L1); naive `if rs1 == 0: skip` would silently NOP.
**Why it happens:** Same convention as MM phases (each 16-bit field), but VEC is a single-field rs1 — easy to forget.
**How to avoid:** Apply `vec_size = (rs1 & 0xFFFF) or 0x10000` (Python: ` (rs1 & 0xFFFF) or 0x10000`).
**Warning signs:** Long vector ops produce zero output.

### Pitfall 8: SOFTMAX/ESUM Forward, but ESUM Writes to L0 (NOT ADDRR)
**What goes wrong:** Test asserts ESUM result at ADDRR; actual result is at L0 offset `(GSPR_OPERAND3 & 0x1F) * 32`.
**Why it happens:** ESUM is a **scalar reduction** like VSUM/DOT — output is one FP16 value to L0, not a vector to ADDRR.
**How to avoid:** Read `gtx_npu_act.cc:142-147` carefully; ESUM @handler must read GSPR_OPERAND3 and write to `npu.mem.l0_byte(nest, spu)` at the correct offset.
**Warning signs:** ESUM unit test passes shape check but value at ADDRR is unchanged garbage.

## Code Examples

### VEC Engine — `firmware_vec_op` decode + dispatch
```python
# riscv/gtx/vec_engine.py
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:572-754
from .vec_core import sasmd_kernel, dot_kernel, vsum_kernel, clamp_min_kernel, clamp_max_kernel, accum_kernel, arange_kernel
from .encoding import (
    GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3, GSPR_GTX_OPCODE,
    LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR,
)
from .params import GTX_NEST_NUM, GTX_SPU_NUM

def firmware_vec_op(npu, proc, insn) -> int:
    """Direct port of gtx_npu_vec.cc:572-754. Single entry for all VEC sub-ops."""
    rs1 = int(proc.state.XPR[insn.rs1])
    vec_size = (rs1 & 0xFFFF) or 0x10000  # HW conv: 0 → 65536 (Pitfall 7)

    funct7 = insn.funct
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2

    nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
    spu  = npu.warp.curr_id if npu.warp.is_tloop else 0
    if nest >= GTX_NEST_NUM: nest = 0
    if spu  >= GTX_SPU_NUM:  spu  = 0

    # Stage rs2 (scalar) into GSPR_OPERAND2 for ops that consume it (gtx_npu_vec.cc:736-737)
    rs2 = int(proc.state.XPR[insn.rs2])
    npu.gspr[GSPR_GTX_OPERAND2] = rs2

    # L0 path? (funct3 & 4) — funct7 ∈ {0x18 ARITH, 0x19 FMADD, 0x1C MATH, 0x1D SIGN, 0x1E ROUND}
    if (funct3 & 4) and funct7 in (0x18, 0x19, 0x1C, 0x1D, 0x1E):
        return _dispatch_l0_path(npu, proc, insn, funct7, funct3 & 3, nest, spu, rs1, rs2)

    # L1 VV/VS path
    return _dispatch_l1_path(npu, funct7, funct3, nest, spu, vec_size)
```

### ACT Engine — direction asymmetry
```python
# riscv/gtx/act_engine.py
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:23-164
ACT_OPS_REVERSED = (
    1,  # GTX_ACT_TANH
    3,  # GTX_ACT_GELU
    4,  # GTX_ACT_SIGMOID
    5,  # GTX_ACT_PRELU
)  # NOTE: This is for ENGINE-internal logic only; D-06 still requires
   # is_reversed literal at @handler entry. The set is private to the engine.

def firmware_act(npu, proc, insn, *, op_id: int, is_reversed: bool) -> int:
    """gtx_npu_act.cc:23-164.

    is_reversed is passed by @handler (D-05, D-06).
    ACT_OPS_REVERSED is provided as a *consistency check* at engine entry but
    is NOT the source of truth.
    """
    assert is_reversed == (op_id in ACT_OPS_REVERSED), \
        f"@handler is_reversed mismatch: op_id={op_id}, is_reversed={is_reversed}"
    # ... LSPR read, FP16 view, kernel dispatch, FP16 writeback
```

### FP8 LUT
```python
# riscv/gtx/act_core.py
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu.h:154-179, 182-221

# Build at module import time (D-14)
def _build_fp8_to_fp16_lut() -> np.ndarray:
    out = np.zeros(256, dtype=np.float16)
    for h in range(256):
        h_sign, h_exp, h_frac = (h >> 7) & 1, (h >> 3) & 0xF, h & 7
        if h_exp == 0:
            val = 0.0 if h_frac == 0 else (h_frac / 8.0) * (2.0 ** -6)
        elif h_exp == 0xF:
            val = float('inf') if h_frac == 0 else float('nan')
        else:
            val = (1.0 + h_frac / 8.0) * (2.0 ** (h_exp - 7))
        if h_sign and not np.isnan(val):
            val = -val
        out[h] = np.float16(val)
    return out

FP8_TO_FP16_LUT: np.ndarray = _build_fp8_to_fp16_lut()
# 64KB FP16→FP8 LUT analogous (D-15)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-op P3 stub returning 0 (Phase 3 plan-04 D-2) | Real op kernel + bit-exact strict-mode regression | P5 Wave 1b GREEN-fill | Most P5 scope is the kernel implementations themselves. |
| `np.matmul(A_f32, B_f32)` (CONTEXT D-02 surface) | Explicit 3-loop FP32 (P4 RESEARCH lock) | P4 plan-02 GREEN-fill | Lesson carried forward to P5 VSUM/DOT. |
| `proc.get_state()` (P2-P4 Wave 1 cross-cutting bug) | `proc.state` (P4 04-05 fix) | P4 04-05 mechanical rename across 27 sites | All P5 production code MUST use `proc.state` from day one. |
| (P3 stub) `dispatch_iss_opcode` MM/VEC/ACT all NOPs | P4 fills MM (funct7=0x00); P5 fills VEC (1) and ACT (2) | P5 Wave 1b plan 03 (vec_engine) + plan 05 (act_engine) | dispatch_4mode.py extends to route Mode 4 firmware DISPATCH_VEC/ACT into the new engines. |

**Deprecated/outdated:**
- CONTEXT D-13 missing FP64↔FP16 — Adjustment 1 above (corrected)
- CONTEXT D-01 CLAMP funct7=0x18..0x1F range — Adjustment 2 (corrected to single funct7=0x1F)
- ROADMAP P5 success criterion #1 phrasing "≈ 0.1, NOT inf" for VSUM anti-pattern — corrected above to "≈ 100.1, NOT 1000.0" (the actual FP16 saturation pattern)

## Open Questions

1. **Should VEC-04 `_imm` and ACT-05 `_imm` activations share a common L0 read/write helper module?**
   - What we know: 6 different `exec_*_imm` C++ functions all use the same `(reg & 0x1F) * 32` L0 offset formula and same little-endian 16-element layout.
   - What's unclear: whether to expose a `tests/gtx/_l0_helpers.py` test utility (read 16 FP16 from L0[reg]) or duplicate the inline view across each kernel test.
   - Recommendation: **Defer to plan-stage.** First test (`test_op_vec.py::test_sasmd_is_writes_to_l0_reg`) will reveal whether duplication or extraction is cleaner.

2. **Are `arange_v` / `accum_v` / SCVT_HN exercised by any vendor `.elf`?**
   - What we know: They have disasm entries and dispatch wiring. No vendor `.elf` in `vendor/gtx_cpp_reference/test/` references them by name (POOL_1D and ARANGE/ACC have data but as input/ref text, not ELF).
   - What's unclear: whether end-to-end strict-mode regression for these ops is achievable in P5 or must wait for P6 full sweep.
   - Recommendation: P5 covers them at the **op-level unit test only** (D-15 `np.array_equal` direct vs in-Python oracle); strict-mode `.elf` regression for these defers to P6 VRF-04.

3. **Mode-B VSUM golden source (D-10)?**
   - What we know: No vendor `.elf` does row-split VSUM explicitly; it's a firmware-orchestration pattern.
   - What's unclear: Whether to (a) hand-write a mini `vsum_row_split.S` that calls VSUM 4× and re-accumulates, or (b) synthesize the golden in-Python by simulating the same row-split.
   - Recommendation: **Option (b)** — synthesize golden in `tests/gtx/_oracles.py::vsum_row_split_oracle(rows, row_len)`. The mode-B claim is "firmware composition matches op-level call N times"; that's testable without a .elf.

4. **Does `proc_with_addra_addrr_seeded` fixture belong in conftest.py or a new helper module?**
   - What we know: CONTEXT integration_points suggests conftest.py.
   - What's unclear: How many tests need ADDRA-only (most VEC) vs both-seeded (all reversed activations).
   - Recommendation: Conftest fixture `seed_addra_addrr(npu, nest, spu, addra_pattern, addrr_pattern)` — test calls it with explicit patterns.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All | ✓ (pyspike requires 3.10 per P1 D-08) | 3.10+ | — |
| numpy ≥ 2.0 | All | ✓ (P1 lock; host has 2.2.6) | 2.2.6 | — |
| pytest ≥ 8 | All tests | ✓ | — | — |
| `/opt/riscv/bin/riscv64-unknown-elf-gcc` | Building `activation_relu_gelu.elf` once | ✓ (P2 D-22 verified) | — | Commit pre-built `.elf` (P4 mm_basic.elf precedent) — Makefile only invoked when developer rebuilds the .S |
| pyspike `_riscv.so` C extension | `test_regression_fw_act.py` strict-mode .elf regression | ✓ when developer's wheel is built; ✗ on cibuildwheel-clean checkout | — | All other tests use `MockProcessor`. Strict-mode regression test is gated on `_RISCV_AVAILABLE` (P4 04-01 pattern). |
| pyspike CLI on PATH | `subprocess.run(['pyspike', '--extlib=riscv.gtx', ...])` | ✓ when wheel installed; fallback to `[sys.executable, '-m', 'riscv']` per P4 D-11 | — | Subprocess pattern from P2 test_skeleton.py |
| scipy (for `op_gelu_erf`) | (would be needed if porting GELU_ERF oracle) | ✗ (CLAUDE.md ban) | — | **SKIP `GELU_ERF` from oracle suite** (Adjustment 4) |

**Missing dependencies with no fallback:** None. Everything required for P5 is in place.

**Missing dependencies with fallback:** `_riscv.so` for the `.elf` regression — gated skip per P4 pattern.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥ 8 (existing in `pyproject.toml`) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` (existing); offline isolation via `--noconftest -o "addopts="` (P2 plan-05 D-1) |
| Quick run command | `pytest tests/gtx/test_op_vec.py tests/gtx/test_op_act.py tests/gtx/test_op_format.py tests/gtx/test_pooling.py tests/gtx/test_vsum_precision.py -x --noconftest -o "addopts="` |
| Full suite command | `pytest tests/gtx/ -q` (includes test_regression_fw_act.py; gated on `_RISCV_AVAILABLE` + `activation_relu_gelu.elf` + `pyspike` on PATH) |
| Estimated runtime | ~45s quick / ~2-3 min full (incl. 64K-entry FP16→FP8 LUT build at first import + 29 oracle parametrize) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VEC-01 | SASMD VS variants on L1 (add/sub/mul/div) | unit | `pytest tests/gtx/test_op_vec.py::test_sasmd_vs_add -x` | ❌ Wave 0 |
| VEC-01 | SASMD IS variants on L0 (add/sub/mul/div) | unit | `pytest tests/gtx/test_op_vec.py::test_sasmd_is_add -x` | ❌ Wave 0 |
| VEC-02 | VSUM FP32-internal anti-pattern | unit | `pytest tests/gtx/test_vsum_precision.py::test_vsum_fp32_internal_anti_pattern -x` | ❌ Wave 0 |
| VEC-02 | DOT FP32-internal | unit | `pytest tests/gtx/test_op_vec.py::test_dot_fp32_internal -x` | ❌ Wave 0 |
| VEC-02 | VSUM mode-B row-split | parametrized | `pytest tests/gtx/test_vsum_precision.py::test_vsum_row_split_matches_cpp -x` | ❌ Wave 0 |
| VEC-03 | CLAMP min/max with scalar from GSPR_OPERAND2 | unit | `pytest tests/gtx/test_op_vec.py::test_clamp_min_uses_gspr_operand2 -x` | ❌ Wave 0 |
| VEC-03 | accum_v cumulative sum | unit | `pytest tests/gtx/test_op_vec.py::test_accum_v_cumulative -x` | ❌ Wave 0 |
| VEC-03 | arange_v with start+step from GSPR_OPERAND2 | unit | `pytest tests/gtx/test_op_vec.py::test_arange_v_start_step -x` | ❌ Wave 0 |
| VEC-03 | L0/L1 path discrimination via funct3&4 | parametrized | `pytest tests/gtx/test_op_vec.py::test_l0_l1_path_branch -x` | ❌ Wave 0 |
| VEC-04 | exec_vec_scalar (VS L1 path scalar arith) | unit | `pytest tests/gtx/test_op_vec.py::test_exec_vec_scalar -x` | ❌ Wave 0 |
| VEC-04 | exec_scalar_imm (IS L0 path) | unit | `pytest tests/gtx/test_op_vec.py::test_exec_scalar_imm -x` | ❌ Wave 0 |
| VEC-04 | exec_vector_imm (II L0 path) | unit | `pytest tests/gtx/test_op_vec.py::test_exec_vector_imm -x` | ❌ Wave 0 |
| VEC-05 | firmware_vec_op rs1 decode (vec_size=0→65536) | unit | `pytest tests/gtx/test_op_vec.py::test_firmware_vec_op_decode -x` | ❌ Wave 0 |
| VEC-05 | rs2 staging into GSPR_OPERAND2 | unit | `pytest tests/gtx/test_op_vec.py::test_firmware_vec_op_stages_rs2 -x` | ❌ Wave 0 |
| ACT-01 | RELU forward (ADDRA → ADDRR) | unit | `pytest tests/gtx/test_op_act.py::test_relu_forward_direction -x` | ❌ Wave 0 |
| ACT-01 | SOFTMAX forward (max + sum + normalize) | unit | `pytest tests/gtx/test_op_act.py::test_softmax_forward -x` | ❌ Wave 0 |
| ACT-01 | ESUM forward, scalar to L0 | unit | `pytest tests/gtx/test_op_act.py::test_esum_writes_l0_scalar -x` | ❌ Wave 0 |
| ACT-02 | PRELU reversed (ADDRR → ADDRA) | unit | `pytest tests/gtx/test_op_act.py::test_prelu_reversed_direction -x` | ❌ Wave 0 |
| ACT-02 | GELU reversed | unit | `pytest tests/gtx/test_op_act.py::test_gelu_reversed_direction -x` | ❌ Wave 0 |
| ACT-02 | TANH reversed | unit | `pytest tests/gtx/test_op_act.py::test_tanh_reversed_direction -x` | ❌ Wave 0 |
| ACT-02 | SIGM reversed | unit | `pytest tests/gtx/test_op_act.py::test_sigm_reversed_direction -x` | ❌ Wave 0 |
| ACT-02 | All 8 activations: distinct ADDRA/ADDRR pre-load proves direction (ROADMAP #2) | parametrized | `pytest tests/gtx/test_op_act.py::test_direction_asymmetry_table -x` | ❌ Wave 0 |
| ACT-03 | Max-pool output_len = length/kernel_size | unit | `pytest tests/gtx/test_pooling.py::test_max_pool_output_length -x` | ❌ Wave 0 |
| ACT-03 | Avg-pool signed-zero canonicalization (-0.0 → +0.0) | unit | `pytest tests/gtx/test_pooling.py::test_avg_pool_signed_zero_canon -x` | ❌ Wave 0 |
| ACT-03 | Pool always forward direction | unit | `pytest tests/gtx/test_pooling.py::test_pool_always_forward -x` | ❌ Wave 0 |
| ACT-04 | format_cvt scale+offset packing in GSPR_OPERAND2 | unit | `pytest tests/gtx/test_op_format.py::test_scale_offset_packing -x` | ❌ Wave 0 |
| ACT-04 | FP8↔FP16 round-trip (256 values) | parametrized | `pytest tests/gtx/test_op_format.py::test_fp8_roundtrip_identity -x` | ❌ Wave 0 |
| ACT-04 | FP8 subnormal table | parametrized | `pytest tests/gtx/test_op_format.py::test_fp8_subnormal_decode -x` | ❌ Wave 0 |
| ACT-04 | FP8 exp=0xF maps to inf (h_frac=0) / NaN (h_frac>0) | parametrized | `pytest tests/gtx/test_op_format.py::test_fp8_exp_max -x` | ❌ Wave 0 |
| ACT-04 | INT8↔FP16 with scale/offset | unit | `pytest tests/gtx/test_op_format.py::test_int8_fp16_scale_offset -x` | ❌ Wave 0 |
| ACT-04 | INT32→FP16 normalize (SCVT_HN) | unit | `pytest tests/gtx/test_op_format.py::test_int32_fp16_normalize -x` | ❌ Wave 0 |
| ACT-04 | FP32↔FP16 bit-pattern preserving | unit | `pytest tests/gtx/test_op_format.py::test_fp32_fp16_no_scale -x` | ❌ Wave 0 |
| ACT-04 | FP64↔FP16 bit-pattern preserving (Adjustment 1) | unit | `pytest tests/gtx/test_op_format.py::test_fp64_fp16_no_scale -x` | ❌ Wave 0 |
| ACT-05 | _imm activations: PRELU/GELU/TANH/SIGM on L0 | unit | `pytest tests/gtx/test_op_act.py::test_act_imm_l0 -x` | ❌ Wave 0 |
| ACT-05 | _imm SOFTMAX/ESUM on L0 | unit | `pytest tests/gtx/test_op_act.py::test_softmax_imm_l0 -x` | ❌ Wave 0 |
| ACT-05 | funct3 & 4 selects L0 immediate path | parametrized | `pytest tests/gtx/test_op_act.py::test_act_funct3_l0_branch -x` | ❌ Wave 0 |
| VRF-02 | 20 directly-mapped oracle parity (skip composite + scipy) | parametrized | `pytest tests/gtx/test_oracle_parity.py -x` | ❌ Wave 0 |
| ALL | activation_relu_gelu.elf strict-mode pass | regression | `pytest tests/gtx/test_regression_fw_act.py::test_act_strict_mode_pass -x` | ❌ Wave 0 (gated) |

### Sampling Rate
- **Per task commit:** quick run command (≤45s; pure-python; no `_RISCV_AVAILABLE` requirement)
- **Per wave merge:** full suite command (full P3 + P4 + P5 regression, ~2-3 min)
- **Phase gate:** Full suite green INCLUDING `test_regression_fw_act.py::test_act_strict_mode_pass`. If `_RISCV_AVAILABLE=False` or `activation_relu_gelu.elf` missing, regression test skips cleanly (NEVER fail).

### Wave 0 Gaps
- [ ] `tests/gtx/test_op_vec.py` — covers VEC-01..05
- [ ] `tests/gtx/test_op_act.py` — covers ACT-01, ACT-02, ACT-05
- [ ] `tests/gtx/test_op_format.py` — covers ACT-04 (7 cvt directions + scale/offset)
- [ ] `tests/gtx/test_pooling.py` — covers ACT-03
- [ ] `tests/gtx/test_vsum_precision.py` — covers VEC-02 dual-mode (D-09/D-10)
- [ ] `tests/gtx/test_oracle_parity.py` — covers VRF-02 (20 oracles parametrized)
- [ ] `tests/gtx/test_regression_fw_act.py` — covers .elf strict-mode regression (gated)
- [ ] `tests/gtx/_oracles.py` — VRF-02 helpers (29 functions, skip GELU_ERF)
- [ ] `tests/gtx/conftest.py` — add `proc_with_addra_addrr_seeded` fixture
- [ ] `tests/gtx/data/elf/{activation_relu_gelu.S, activation_relu_gelu.elf}` — D-04 fixture
- [ ] `tests/gtx/data/elf/Makefile` — extend with `activation_relu_gelu.elf` rule
- [ ] `tests/gtx/data/golden/activation_relu_gelu.hex` — synthesized golden
- [ ] `src/main/python/riscv/gtx/vec_core.py` — module exists check (will fail until Wave 1b)
- [ ] `src/main/python/riscv/gtx/vec_engine.py` — module exists check
- [ ] `src/main/python/riscv/gtx/act_core.py` — module exists check
- [ ] `src/main/python/riscv/gtx/act_engine.py` — module exists check
- [ ] `src/main/python/riscv/gtx/ops/vec.py` — module exists check
- [ ] `src/main/python/riscv/gtx/ops/act.py` — module exists check
- [ ] `src/main/python/riscv/gtx/ops/__init__.py` — `from . import vec` + `from . import act` lines
- [ ] Test framework: existing pytest infra is sufficient; no new install needed.

## Sources

### Primary (HIGH confidence — vendor C++ on disk)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:89-151` — FP16 ↔ FP32 conversion helpers (idempotent baseline)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:154-179` — FP8 → FP32 codec (E4M3-bits, NOT-NVIDIA-semantics)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:182-221` — FP16 → FP8 codec
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:265-353` — funct7 encoding constants (gem5 + ISS-full)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:371-377` — `GTX_ACT_*` enum values
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:382-405` — `GTX_VEC_*` enum values
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:419-455` — `GTX_IMM_*` sub-op enum values
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:18-164` — `exec_activation` (direction asymmetry source-of-truth at lines 37-42)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:166-220` — `exec_pooling` (signed-zero canon at line 211)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:222-372` — `exec_format_cvt` (scale/offset at 240-243; sub_op&1 direction at 250)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:374-431` — `exec_act_imm` (L0 PRELU/GELU/TANH/SIGM)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc:436-487` — `exec_softmax_imm` (L0 ESUM/SOFTMAX)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:23-273` — `exec_vector_op` (master VEC switch incl VSUM 102-112, DOT 251-262)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:283-342` — `exec_vec_scalar` (L1 VS path)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:352-402` — `exec_scalar_imm` (L0 IS path)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:410-454` — `exec_vector_imm` (L0 II path)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc:572-754` — `firmware_vec_op` (rs1 decode + funct3 dispatch)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:67-218` — funct7/funct3 ↔ mnemonic table (definitive encoding)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:233-263, 535-700` — VEC/ACT/POOL/format_cvt dispatch wiring
- `vendor/gtx_cpp_reference/gtx/verify_ref.py:185-226` — 30-op host-side oracle suite
- `vendor/gtx_cpp_reference/spike-devices/src/gtx_params.h:38-67` — GSPR/LSPR address constants

### Secondary (HIGH confidence — pyspike code on disk)
- `src/main/python/riscv/gtx/encoding.py` — P2-P4 funct constants; P5 appends VEC/ACT/SCVT
- `src/main/python/riscv/gtx/mm_engine.py` — pattern source for vec_engine.py and act_engine.py
- `src/main/python/riscv/gtx/gemm_core.py:147-149` — explicit-loop FP32 reduction precedent
- `src/main/python/riscv/gtx/_registry.py` — @handler decorator + 2-level dispatch builder
- `src/main/python/riscv/gtx/ops/mm.py` — pattern source for ops/vec.py and ops/act.py (10 thin @handlers)
- `src/main/python/riscv/gtx/fp.py` — `fp16_to_fp32` / `fp32_to_fp16` helpers
- `tests/gtx/_verify_minimal.py` — strict-mode hex compare (P5 reuses unchanged)
- `tests/gtx/_mocks.py` — `MockProcessor.state` property (P4 04-05 back-compat path)
- `tests/gtx/data/elf/{Makefile, mm_basic.S, mm_basic.elf}` — fixture pattern (P5 mirrors)
- `.planning/phases/04-mm-subsystem/04-RESEARCH.md` — np.matmul drift analysis; subprocess pattern; .S fallback strategy
- `.planning/phases/04-mm-subsystem/04-VERIFICATION.md` — `proc.state` is property (NOT method)
- `.planning/phases/05-vec-act-pool/05-CONTEXT.md` — locked decisions D-01..D-16 (verbatim above)

### Tertiary (MEDIUM confidence — derived from primary sources)
- VRF-02 directly-mapped oracle count (20 of 30) — derived from cross-referencing `verify_ref.py` OPS dict against C++ `exec_vector_op` and `exec_activation` enums
- VSUM anti-pattern target value `≈100.1, NOT 1000.0` — derived from FP16 epsilon = 1e-3 analysis (corrects ROADMAP "≈0.1, NOT inf" approximation)
- 16 SASMD-family @handlers (8 at funct7=0x10 + 8 at funct7=0x18) — derived from disasm.inc lines 67-94

## Metadata

**Confidence breakdown:**
- VEC standard stack: **HIGH** — every op enum and funct7/funct3 mapping is direct from disasm.inc + vec.cc.
- ACT direction asymmetry: **HIGH** — single `if/else` at gtx_npu_act.cc:37-42; literal C++ table.
- FP8 codec layout: **HIGH** — bit-by-bit C++ port, divergences from NVIDIA E4M3 documented.
- format_cvt scale/offset packing: **HIGH** — `op2 & 0xFFFF` for scale, `(op2 >> 16) & 0xFFFF` for offset, both FP16, verified line 240-243.
- VSUM/DOT precision: **HIGH** — explicit-loop FP32 with single FP16 cast verified bit-by-bit.
- VRF-02 oracle count: **MEDIUM** — count of 30 oracles (29 portable, 20 directly-mapped) is a research determination; vendor docs say "32" loosely.
- `activation_relu_gelu.elf` fixture: **HIGH** — vendor has no asset; `.S` fallback is the established pattern.
- VSUM mode-B golden source: **MEDIUM** — synthesized in-Python is recommended; no vendor `.elf` exercises this pattern.
- pool kernel/stride packing: **HIGH** — `op1 & 0xFFFF` (length), `op2 & 0xFFFF` (kernel_size).
- `proc.state` access pattern: **HIGH** — P4 04-05 PHASE-CRITICAL fix verified end-to-end.

**Research date:** 2026-05-07
**Valid until:** 2026-06-07 (30 days; vendor C++ source is stable, encoding constants frozen)
