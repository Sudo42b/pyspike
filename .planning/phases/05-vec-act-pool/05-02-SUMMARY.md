---
phase: 05-vec-act-pool
plan: 02
subsystem: vec
tags: [vec, sasmd, dot, vsum, clamp, accum, arange, fp32-internal, l0-l1-dispatch, py-rocc]

# Dependency graph
requires:
  - phase: 05-vec-act-pool
    plan: 01
    provides: "encoding constants (GTX_F7_VEC_*, GTX_VEC_*) + vec_core.py + vec_engine.py + ops/vec.py importable stubs + test_op_vec.py / test_vsum_precision.py RED scaffolds"
  - phase: 04-mm-subsystem
    provides: "_registry.@handler decorator + ops/__init__.py auto-import + MockProcessor.state property + gemm_core FP32-internal explicit-loop precedent (gemm_core.py:147-149)"
provides:
  - "vec_core.py 7 stateless FP32-internal kernels (sasmd/dot/vsum/clamp_min/clamp_max/accum/arange) with explicit Python for-loop discipline (no np.sum/np.dot/np.matmul/np.einsum on FP16)"
  - "vec_engine.firmware_vec_op full body: rs1[15:0]→vec_size (Pitfall 7 0→0x10000) + rs2→GSPR_OPERAND2 + funct7-keyed L0/L1 path branch covering 0x10/0x18/0x1A/0x1C/0x1D/0x1E/0x1F"
  - "ops/vec.py 22 thin @handler entries (8 SASMD-VS/IS at funct7=0x10, 2 dot/vsum at funct7=0x1A, 8 SASMD-VV/II at funct7=0x18, 4 CLAMP family at funct7=0x1F)"
  - "test_op_vec.py 15 GREEN op-level unit tests covering VEC-01..05"
  - "test_vsum_precision.py 5 GREEN tests (anti-pattern + 4 parametrized row-split)"
  - "GTX_F7_VEC_DOT_SUM correction: 0x13→0x1A (vendor disasm.inc:101-104 authoritative); GTX_F7_VEC_MINMAX=0x13 added"
affects: [05-03-act, 05-04-pool-format, 05-05-oracle, 05-06-regression]

# Tech tracking
tech-stack:
  added: []  # zero new runtime deps; pure Python + NumPy
  patterns:
    - "Explicit Python for-loop FP32 accumulator for FP16 reductions (vsum/dot) -- NOT np.sum/np.dot/np.matmul/np.einsum (RESEARCH Pitfall 2)"
    - "L0 SVR access via 16-FP16 block at byte offset (reg & 0x1F) * 32 (gtx_npu_vec.cc:360, 417)"
    - "rs2 carries FP16 scalar in low 16 bits for SASMD/CLAMP/ARANGE; arange decodes start (low) + step (high) from same rs2"
    - "Vendor C++ source is authoritative for funct7 values -- correct mid-flight when scaffold seeded a draft value (Rule 1 deviation pattern)"

key-files:
  created: []
  modified:
    - src/main/python/riscv/gtx/vec_core.py
    - src/main/python/riscv/gtx/vec_engine.py
    - src/main/python/riscv/gtx/ops/vec.py
    - src/main/python/riscv/gtx/encoding.py
    - tests/gtx/test_op_vec.py
    - tests/gtx/test_vsum_precision.py

key-decisions:
  - "vec_core kernels follow gemm_core (P4 D-01) leaf-module discipline -- zero `riscv.gtx.*` imports beyond .encoding constants; P7 numba @njit boundary clean."
  - "vec_engine.firmware_vec_op unifies the SASMD funct7=0x10 path (which C++ dispatches via dispatch_iss_opcode) with the firmware_vec_op funct7={0x18,0x1A,0x1C-0x1F} path. Single Python entry keeps @handler funct7-routing uniform; no semantic divergence from C++."
  - "Anti-pattern test redesigned: original [1.0, 1e-4]*1000 input rounds identically in FP16 across both naive and FP32-internal paths (FP16 has only ~3 decimal digits at 1000+ range). Replaced with [1024.0]+5000*[0.4] which genuinely diverges (explicit-FP16-cumulative=1024.0 vs FP32-internal=3024.0)."
  - "GTX_F7_VEC_DOT_SUM corrected from 0x13 to 0x1A. funct7=0x13 is vendor MIN/MAX scalar arith (max_vs/min_vs/max_is/min_is); 0x1A is DOT/SUM with funct3=0→dot_vvs and funct3=1→sum_vs (vsum). Vendor authoritative: gtx_npu.h:308 + disasm.inc:101-104 + gtx_npu_vec.cc:632-637 all align."
  - "DOT/VSUM scalar writeback writes FP16 to L1[ADDRR][0] AND L0[0..1] in LE byte order (gtx_npu_vec.cc:108-110). MM_O writes BE; VEC writes LE -- documented asymmetry."

patterns-established:
  - "FP32-internal-then-FP16-cast discipline as a layer-wide invariant: every reduction (vsum, dot, accum, arange) accumulates in np.float32 and casts to np.float16 only at writeback. Plans 03/04 must follow."
  - "rs2 staging convention: firmware_vec_op writes proc.state.XPR[insn.rs2] -> npu.gspr[GSPR_GTX_OPERAND2] BEFORE any sub-op runs. Sub-ops then read low/high 16 of rs2 via _fp16_low16/_fp16_high16 helpers."
  - "L0 register addressing: byte-offset (reg & 0x1F) * 32; 16 FP16 elements per register; LE bit-pattern; access via npu.mem.l0_byte(nest, spu).view(np.float16) slice."

requirements-completed: [VEC-01, VEC-02, VEC-03, VEC-04, VEC-05]

# Metrics
duration: 16min
completed: 2026-05-07
---

# Phase 5 Plan 02: VEC Subsystem Summary

**VEC-01..05 GREEN: 22 @handler entries + 7 stateless FP32-internal NumPy kernels + spike-bound L0/L1 dispatch covering SASMD (8 VS/IS + 8 VV/II), DOT/VSUM with FP32-internal precision discipline, and the CLAMP family (clamp_min_v/clamp_max_v/accum_v/arange_v) -- 20 op-level unit tests GREEN, 0 regressions across the 219-test P3+P4+P5 suite.**

## Performance

- **Duration:** ~16 min (5 atomic commits)
- **Tasks:** 3
- **Files modified:** 6
- **Test surface change:** +20 GREEN (15 test_op_vec + 5 test_vsum_precision); 25 skipped overall (down from 45 baseline)

## Accomplishments

- VEC-01 (SASMD VS+IS): 8 variants × 2 paths covered. test_sasmd_vs_add + test_sasmd_is_add + sub/mul/div_vs all GREEN.
- VEC-02 (DOT/VSUM FP32 internal): test_vsum_fp32_internal_anti_pattern + test_dot_fp32_internal + 4-parametrize test_vsum_row_split_matches_cpp all GREEN.
- VEC-03 (CLAMP): clamp_min_v + clamp_max_v + accum_v + arange_v GREEN with explicit GSPR_OPERAND2 staging assertion.
- VEC-04 (exec_vec_scalar/_imm): VS L1 + IS L0 + II L0 paths GREEN via test_exec_vec_scalar / test_exec_scalar_imm / test_exec_vector_imm.
- VEC-05 (firmware_vec_op decode + rs2 staging): test_firmware_vec_op_decode + test_firmware_vec_op_stages_rs2 GREEN.
- 22 VEC @handlers registered (collect_disasms() returns 66 total entries, +22 over Plan 01 baseline of 44).

## Task Commits

1. **Task 1 prep (RED VSUM tests):** `7186e23` (test) -- replaced 5 pytest.skip stubs in test_vsum_precision.py with executable assertions that fail RED before kernel landed.
2. **Task 1 GREEN (vec_core kernels):** `bd0256e` (feat) -- 7 kernels with explicit FP32-internal accumulate; 5/5 vsum precision tests GREEN.
3. **Task 2 prep (RED VEC tests):** `a766c90` (test) -- replaced 13 pytest.skip stubs in test_op_vec.py with full executable scaffolds against vec_engine.firmware_vec_op + GtxNpu fixtures.
4. **Task 2 GREEN (vec_engine):** `28d2ba6` (feat) -- firmware_vec_op full body + L0/L1 dispatch + encoding correction (GTX_F7_VEC_DOT_SUM=0x1A).
5. **Task 3 (ops/vec.py):** `d3d7a2b` (feat) -- 22 @handler entries.

**Plan metadata commit:** added below.

## Files Created/Modified

| File | LOC | Role |
|------|-----|------|
| `src/main/python/riscv/gtx/vec_core.py` | 146 | 7 pure stateless FP32-internal kernels (sasmd/dot/vsum/clamp_min/clamp_max/accum/arange) |
| `src/main/python/riscv/gtx/vec_engine.py` | 302 | firmware_vec_op + 4 sub-dispatchers (sasmd, arith_l0_ii, unary_l0, _apply_unary) + L1/L0 view helpers |
| `src/main/python/riscv/gtx/ops/vec.py` | 178 | 22 thin @handler shims |
| `src/main/python/riscv/gtx/encoding.py` | (modified) | GTX_F7_VEC_DOT_SUM corrected 0x13→0x1A; GTX_F7_VEC_MINMAX=0x13 added |
| `tests/gtx/test_op_vec.py` | 320 | 15 GREEN VEC op tests |
| `tests/gtx/test_vsum_precision.py` | 78 | 5 GREEN VSUM precision tests |

**Total source delta:** +626 LOC across vec_core/vec_engine/ops/vec.

## Anti-pattern Test Result (numbers from actual test execution)

For input `arr = np.array([1024.0] + [0.4]*5000, dtype=np.float16)`:

| Path | Result | Notes |
|------|--------|-------|
| Explicit FP16 cumulative `for x: s = np.float16(s + x)` (would-be naive) | **1024.0** | All `0.4` additions absorbed by FP16 ULP at 1024+ accumulator (FP16 ULP ≥ 1.0 for values ≥ 1024) |
| `vec_core.vsum_kernel(arr)` (FP32 internal then single FP16 cast) | **3024.0** | Preserves all 5000*0.4 = 2000 contribution: 1024 + 2000 ≈ 3024 |
| Oracle `np.float16(arr.astype(np.float32).sum(dtype=np.float32))` | **3024.0** | Matches kernel exactly |

The test asserts both `actual == oracle` AND `actual != naive_fp16` to prove the kernel ISN'T silently doing the naive thing.

For overflow input `np.full(70000, 1.0, np.float16)` (D-12 corner): kernel returns `np.float16(inf)` because the FP32 sum 70000.0 exceeds FP16 max 65504 → FP16 cast on overflow yields inf per IEEE round-to-nearest (NumPy 2.x).

The original ROADMAP-cited input `[1.0, 1e-4]*1000` is documented but NOT used as the divergence test -- both naive and FP32-internal end up at FP16 1000.0 because FP16 has only ~3 decimal digits at the 1000-magnitude (FP32 sum 1000.10004 rounds to FP16 1000.0).

## Decisions Made

1. **Anti-pattern test input swap:** plan body's `[1.0, 1e-4]*1000` → `[1024.0] + 5000*[0.4]`. Both inputs prove the same precision discipline; the new one is genuinely divergent at FP16 cast time. Documented in test docstring.
2. **GTX_F7_VEC_DOT_SUM = 0x1A (vendor-authoritative correction):** Plan 01 seeded 0x13 from a draft note. Vendor disasm.inc:80-84 has `max_vs/min_vs/max_is/min_is` at funct7=0x13; DOT/SUM lives at 0x1A per disasm.inc:101-104 and gtx_npu_vec.cc:632-637. Added GTX_F7_VEC_MINMAX=0x13 for future plan 03 reference.
3. **Plan body funct3 ordering correction:** plan body says "vsum funct3=0, dot funct3=1" but vendor source `gtx_npu_vec.cc:632-637` has the reverse (`case 0: GTX_VEC_DOT; case 1: GTX_VEC_VSUM`). Implementation follows vendor; mnemonic registration uses `dot_vvs` at funct3=0 and `sum_vs` at funct3=1 per disasm.inc:101-102.
4. **SASMD funct7=0x10 routed via firmware_vec_op:** in C++ this funct7 is dispatched by `dispatch_iss_opcode` (separate path), not `firmware_vec_op`. In pyspike the @handler funct7-routing layer hits the engine entry point uniformly, so we extended firmware_vec_op to handle 0x10 too. No semantic divergence; just a Python-side plumbing collapse.
5. **L0 result-reg source = GSPR_OPERAND3 with insn.rd fallback:** vendor `exec_scalar_imm` takes result_reg as a parameter; the dispatch upstream reads from `gspr[GSPR_GTX_OPERAND3] & 0x1F`. Engine reads OPERAND3 first, falls back to `insn.rd & 0x1F` if OPERAND3 not set -- mirrors vendor `gtx_npu_vec.cc:659` pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] GTX_F7_VEC_DOT_SUM seeded with wrong value 0x13 (correct: 0x1A)**
- **Found during:** Task 2 (vec_engine firmware_vec_op dispatch routing)
- **Issue:** Plan 01 SUMMARY and current encoding.py declared `GTX_F7_VEC_DOT_SUM: int = 0x13`. Cross-checking against vendor `gtx_npu.h:308` (`GTX_ISS_F7_DOT_SUM = 0b0011010 = 0x1A`), `gtx_npu_disasm.inc:80-84` (funct7=0x13 is scalar MIN/MAX, NOT DOT/SUM), and `gtx_npu_vec.cc:632-637` (DOT/SUM dispatch lives in case `GTX_ISS_F7_DOT_SUM = 0x1A`) all show the correct value is 0x1A.
- **Fix:** Updated `GTX_F7_VEC_DOT_SUM` to `0x1A`; added `GTX_F7_VEC_MINMAX = 0x13` (with disasm.inc:80-84 reference) for future Plan 03/04 reference. Documented the correction in encoding.py comments. test_op_vec.py + ops/vec.py now reference the corrected value.
- **Files modified:** src/main/python/riscv/gtx/encoding.py, src/main/python/riscv/gtx/vec_engine.py, src/main/python/riscv/gtx/ops/vec.py, tests/gtx/test_op_vec.py
- **Verification:** All 15 test_op_vec tests GREEN with correct DOT/VSUM dispatch. `python3 -c "from riscv.gtx.encoding import GTX_F7_VEC_DOT_SUM; assert GTX_F7_VEC_DOT_SUM == 0x1A"` passes.
- **Committed in:** `28d2ba6`

**2. [Rule 1 - Bug] Plan body's funct3 ordering for dot/vsum was reversed**
- **Found during:** Task 2 (vec_engine routing)
- **Issue:** Plan body says `if funct3 == 0:  # vsum` and `if funct3 == 1:  # dot`. Vendor `gtx_npu_vec.cc:632-637` switch has `case 0: vec_op = GTX_VEC_DOT; case 1: vec_op = GTX_VEC_VSUM;`. disasm.inc:101-102 confirms: `dot_vvs` is at funct3=0, `sum_vs` is at funct3=1.
- **Fix:** Implementation follows vendor (funct3=0 → dot_kernel, funct3=1 → vsum_kernel). Test test_dot_fp32_internal exercises funct3=0 → DOT. test_firmware_vec_op_stages_rs2 uses funct3=1 → vsum (no scalar arg needed).
- **Files modified:** src/main/python/riscv/gtx/vec_engine.py
- **Verification:** test_dot_fp32_internal GREEN; computed dot([1.0]*256, [0.01]*256) = 2.56 matches vec_core.dot_kernel oracle.
- **Committed in:** `28d2ba6`

**3. [Rule 1 - Bug] VSUM anti-pattern test asserted `actual ≈ 100.1` (impossible at FP16)**
- **Found during:** Task 1 GREEN run (test_vsum_fp32_internal_anti_pattern failed)
- **Issue:** Plan body wrote `assert abs(float(actual) - 100.1) < 0.2` but the same input also forces `expected = np.float16(arr.astype(np.float32).sum())`. The FP32 sum is 1000.10004; FP16 cast of that is 1000.0 (FP16 has ~3 decimal digits at 1000-magnitude). Both naive-FP16 and FP32-internal round to identical FP16 1000.0 -- the test as written can never distinguish them. RESEARCH §VSUM Precision (lines 401-403) actually notes "≈ 100.1" refers to the FP32 INTERNAL sum BEFORE FP16 cast.
- **Fix:** Replaced the divergence input with `[1024.0] + 5000*[0.4]` (FP32-internal=3024 vs explicit-FP16-cumulative=1024 -- ~2000 unit divergence, well above FP16 ULP at 1024). Added explicit cumulative FP16 oracle to assert `naive_fp16 != expected` so the test fails LOUDLY if the kernel silently regresses to naive accumulation. Kept the original ROADMAP-cited input documented in the test docstring with its reasoning.
- **Files modified:** tests/gtx/test_vsum_precision.py
- **Verification:** All 5 vsum precision tests GREEN. Anti-pattern test now correctly distinguishes FP32-internal from FP16-naive on a single input.
- **Committed in:** `bd0256e`

---

**Total deviations:** 3 auto-fixed (3 × Rule 1 bug fixes — 2 vendor-truth corrections, 1 unsatisfiable assertion).
**Impact on plan:** All 3 are corrections to scaffold/plan-body bugs surfaced by execution; net result is the plan delivers MORE precision (vendor-authoritative funct7 + a genuinely divergent anti-pattern test). No scope expansion. No new files. No new dependencies.

## Issues Encountered

None — apart from the 3 deviations above (which were planned-work bugs auto-fixed per Rule 1, not unplanned issues).

## User Setup Required

None — pure code change, no env vars / external services.

## Next Phase Readiness

**Wave 2 unblock signal: VEC ops bit-exact with C++ algorithm. Plan 03 (act) can now build on the same 3-way module pattern (act_core stateless kernels + act_engine spike-bound dispatch + ops/act.py @handler shims).**

Ready for parallel landing:
- Plan 03 owns: act_core.py {relu/prelu/gelu/tanh/sigmoid/softmax/esum kernels}, act_engine.py firmware_act, ops/act.py {16 activation @handlers}, test_op_act.py 11 RED → GREEN.
- Plan 04 owns: act_core.py {pool + cvt + FP8 LUTs}, act_engine.py {firmware_pool + firmware_format}, ops/act.py {7 cvt + 2 pool @handlers}, test_op_format.py + test_pooling.py.

**Pattern handoff for Plan 03:** the `_l0_block_view` and `_l1_view_addr` helpers in vec_engine.py are reusable for act_engine.py (same 16-FP16 L0 block addressing + L1 byte-offset view).

**Pattern handoff for Plan 04:** when GTX_F7_VEC_FMADD (0x11), GTX_F7_VEC_FMADD_VV (0x19), GTX_F7_VEC_MINMAX (0x13) need to land later (out of P5 critical path), the same firmware_vec_op + sub-dispatcher structure scales.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| 5 task commits present (`7186e23`, `bd0256e`, `a766c90`, `28d2ba6`, `d3d7a2b`) | All in `git log --oneline -5` |
| vec_core.py importable + 7 kernels callable | PASS (smoke test) |
| vec_engine.py uses `proc.state.XPR` (not `proc.get_state()`) | PASS (`grep -c proc.get_state` = 1, all in docstring; `grep -c proc.state.XPR` = 2 in production code lines 100, 115) |
| No `np.sum`/`np.dot`/`np.matmul`/`np.einsum` calls in vec_core.py production code | PASS (matches only docstring "NEVER np.dot/..." warnings) |
| 22 VEC @handlers registered | PASS (`len([e for e in _HANDLER_REGISTRY if e['mnemonic'] and e['funct7'] in (0x10, 0x1A, 0x18, 0x1F)]) == 22`) |
| collect_disasms() returns ≥66 (was 44 in Plan 01) | PASS (66 total, +22 VEC) |
| All 15 test_op_vec tests GREEN | PASS |
| All 5 test_vsum_precision tests GREEN | PASS |
| Full P3+P4+P5 suite: 219 passed / 25 skipped / 0 failed (was 199/45/0 baseline) | PASS — no regressions, +20 new GREEN |
| GTX_F7_VEC_DOT_SUM == 0x1A (vendor-authoritative) | PASS |
| LOC: vec_core (146 ≥ 90) + vec_engine (302 ≥ 150) + ops/vec (178 ≥ 80) | PASS |

All 11 verification checks pass.

---
*Phase: 05-vec-act-pool*
*Plan: 02 (vec)*
*Completed: 2026-05-07*
